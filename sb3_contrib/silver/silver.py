import copy
from typing import Any, Dict, List, Optional, Tuple, Union

import gym
import numpy as np
import torch as th
from gym import Env, spaces
from sklearn.neighbors import KernelDensity
from stable_baselines3 import DDPG, HerReplayBuffer
from stable_baselines3.common.base_class import maybe_make_env
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.noise import ActionNoise
from stable_baselines3.common.type_aliases import DictReplayBufferSamples, GymEnv, Schedule
from stable_baselines3.common.vec_env import VecEnv, VecNormalize
from stable_baselines3.her.goal_selection_strategy import GoalSelectionStrategy


def softmax(logits: np.ndarray) -> np.ndarray:
    """
    Softmax function.

    :param logits: Input logits as array
    :return: An array the same shape as x, the result will sum to 1
    """
    y = np.exp(logits - logits.max())
    f_x = y / np.sum(y)
    return f_x


def log_prob(samples: np.ndarray, x: np.ndarray) -> np.ndarray:
    """
    Use Kernel Density Estimation to estimate the density of a distribution from samples.
    Uses Scott's Rule to choose the bandwidth.

    :param samples: The samples from the distribution to estimate
    :param x: The value to get the log probabilities
    :return: The log probabilties of x, as array
    """
    bandwidth = samples.shape[0] ** (-1.0 / (samples.shape[1] + 4))  # Scott's rule
    kde = KernelDensity(bandwidth=bandwidth)
    kde.fit(samples)
    return kde.score_samples(x)


class NonParametricSkewedHerReplayBuffer(HerReplayBuffer):
    """
    HER replay buffer that uses a non parametric estimator (KDE) to sample
    goals from low-density areas.

    Idea from "Behavior From the Void: Unsupervised Active Pre-Training"
    Non parametric density estimation with KDE used as sampling weight.

    Like HerReplay buffer with two major changes:
    - The density of the achieved goal distribution is estimated using KDE.
    - A method ``sample_goal`` that uses the density estimation to sample goals
        from the less dense regions.
    """

    def __init__(
        self,
        buffer_size: int,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        env: VecEnv,
        update_logits_freq: int = 1000,
        power: int = -1.0,
        device: Union[th.device, str] = "cpu",
        n_envs: int = 1,
        optimize_memory_usage: bool = False,
        handle_timeout_termination: bool = True,
        n_sampled_goal: int = 4,
        goal_selection_strategy: Union[GoalSelectionStrategy, str] = "future",
        online_sampling: bool = True,
    ) -> None:
        super().__init__(
            buffer_size,
            observation_space,
            action_space,
            env,
            device,
            n_envs,
            optimize_memory_usage,
            handle_timeout_termination,
            n_sampled_goal,
            goal_selection_strategy,
            online_sampling,
        )
        # Logits are the log probabilities estimated with KDE. The more dense, the higher.
        self.logits = 10 * np.ones((self.buffer_size, self.n_envs))
        self.update_logits_freq = update_logits_freq
        self.power = power

    def add(
        self,
        obs: Dict[str, np.ndarray],
        next_obs: Dict[str, np.ndarray],
        action: np.ndarray,
        reward: np.ndarray,
        done: np.ndarray,
        infos: List[Dict[str, Any]],
        is_virtual: bool = False,
    ) -> None:
        super().add(obs, next_obs, action, reward, done, infos, is_virtual)
        if self.pos % self.update_logits_freq == 0:
            self.update_logits()

    def sample(self, batch_size: int, env: Optional[VecNormalize] = None, power: float = -0.5) -> DictReplayBufferSamples:
        """
        Sample elements from the replay buffer.

        :param batch_size: Number of element to sample
        :param env: associated gym VecEnv
            to normalize the observations/rewards when sampling
        :return: Samples
        """
        env_indices = np.random.randint(self.n_envs, size=batch_size)
        batch_inds = np.zeros_like(env_indices)
        # When the buffer is full, we rewrite on old episodes. We don't want to
        # sample incomplete episode transitions, so we have to eliminate some indexes.
        all_inds = np.tile(np.arange(self.buffer_size), (self.n_envs, 1)).T
        is_valid = self.ep_length > 0
        # Special case when using the "future" goal sampling strategy, we cannot
        # sample all transitions, we restrict the sampling domain to non-final transitions
        if self.goal_selection_strategy == GoalSelectionStrategy.FUTURE:
            is_last = all_inds == (self.ep_start + self.ep_length - 1) % self.buffer_size
            is_valid = np.logical_and(np.logical_not(is_last), is_valid)

        valid_inds = [np.arange(self.buffer_size)[is_valid[:, env_idx]] for env_idx in range(self.n_envs)]
        p = [softmax(power * self.logits[:, env_idx][valid_ind].T) for env_idx, valid_ind in enumerate(valid_inds)]
        for env_idx in range(self.n_envs):
            size = np.sum(env_indices == env_idx)
            batch_inds[env_indices == env_idx] = np.random.choice(valid_inds[env_idx], size=size, p=p[env_idx], replace=False)

        # Split the indexes between real and virtual transitions.
        nb_virtual = int(self.her_ratio * batch_size)
        virtual_batch_inds, real_batch_inds = np.split(batch_inds, [nb_virtual])
        virtual_env_indices, real_env_indices = np.split(env_indices, [nb_virtual])

        # get real and virtual data
        real_data = self._get_real_samples(real_batch_inds, real_env_indices, env)
        virtual_data = self._get_virtual_samples(virtual_batch_inds, virtual_env_indices, env)

        # Concatenate real and virtual data
        observations = {
            key: th.cat((real_data.observations[key], virtual_data.observations[key]))
            for key in virtual_data.observations.keys()
        }
        actions = th.cat((real_data.actions, virtual_data.actions))
        next_observations = {
            key: th.cat((real_data.next_observations[key], virtual_data.next_observations[key]))
            for key in virtual_data.next_observations.keys()
        }
        dones = th.cat((real_data.dones, virtual_data.dones))
        rewards = th.cat((real_data.rewards, virtual_data.rewards))

        return DictReplayBufferSamples(
            observations=observations,
            actions=actions,
            next_observations=next_observations,
            dones=dones,
            rewards=rewards,
        )

    def sample_goal(self) -> np.ndarray:
        """
        Sample goal in the skewed manner.

        :return: A goal
        """
        is_empty = (self.ep_length == 0).all()
        if not is_empty:
            sample = self.sample(1, power=self.power)
            goal = sample.next_observations["achieved_goal"][0].detach().cpu().numpy()
        else:
            goal = self.observation_space["desired_goal"].sample()
        return goal

    def update_logits(self) -> None:
        """
        Update the weights of the buffer.
        """
        upper_bound = self.buffer_size if self.full else self.pos
        data = self.next_observations["achieved_goal"][:upper_bound]  # (n, n_envs, obs_shape)
        data = data.reshape(-1, data.shape[-1])  # (n * n_envs, obs_shape)
        samples = super().sample(1024).next_observations["achieved_goal"].detach().cpu().numpy()
        logits = log_prob(samples, data)
        self.logits[:upper_bound] = logits.reshape(upper_bound, -1)


class Goalify(gym.GoalEnv, gym.Wrapper):
    """
    Wrap the env into a GoalEnv.

    :param env: The environment
    :param distance_threshold: Success if euclidian distance between desired and achived
    :param weights: Weights applied before the distance computation. If a dimension is not relevant,
        set its weight to 0.0.
    :param nb_random_exploration_steps: Number of random exploration steps after the goal is reached, defaults to 30
    :param window_size: Agent can skip goals in the goal trajectory within the limit of ``window_size``
        goals ahead, defaults to 10
    """

    def __init__(
        self,
        env: Env,
        distance_threshold: float = 0.5,
        weights: Optional[np.ndarray] = None,
        nb_random_exploration_steps: int = 30,
        window_size: int = 10,
    ) -> None:
        super().__init__(env)
        # Set a goal-conditionned observation space
        self.observation_space = spaces.Dict(
            {
                "observation": copy.deepcopy(self.env.observation_space),
                "desired_goal": copy.deepcopy(self.env.observation_space),
                "achieved_goal": copy.deepcopy(self.env.observation_space),
            }
        )
        # Initial goal; will be re-sampled every time the env is reset
        self.goal = self.observation_space["desired_goal"].sample()  # type: np.ndarray
        # The buffer is required to sample goals;
        # When the env is created, the buffer does notalready exists
        # It has to be set after the model creation
        self.buffer = None  # type: NonParametricSkewedHerReplayBuffer
        # Parameters used for reward computation
        # Weights used to copute the reward: all dimension are not relevant
        self.distance_threshold = distance_threshold
        if weights is None:
            obs_dim = env.observation_space.shape[0]
            self.weights = np.ones(obs_dim)
        # Go-explore param
        self.nb_random_exploration_steps = nb_random_exploration_steps
        self.window_size = window_size

    def set_buffer(self, buffer: NonParametricSkewedHerReplayBuffer) -> None:
        """
        Set the buffer. Must be done before the first reset.

        :param buffer: The buffer
        """
        self.buffer = buffer

    def reset(self) -> Dict[str, np.ndarray]:
        obs = self.env.reset()
        self.goal = self.buffer.sample_goal()
        self.done_countdown = self.nb_random_exploration_steps
        self._is_goal_reached = False  # useful flag
        obs = self._get_obs(obs)  # turn into dict
        return obs

    def _get_obs(self, obs: np.ndarray) -> Dict[str, np.ndarray]:
        return {
            "observation": obs.copy(),
            "achieved_goal": obs.copy(),
            "desired_goal": self.goal.copy(),
        }

    def step(self, action: np.ndarray) -> Tuple[Dict[str, np.ndarray], float, bool, Dict[str, Any]]:
        wrapped_obs, reward, done, info = self.env.step(action)

        # Compute reward (has to be done before moving to next goal)
        reward = self.compute_reward(wrapped_obs, self.goal, {})
        if reward == 0:
            self._is_goal_reached = True

        # When the last goal is reached, delay the done to allow some random actions
        if self._is_goal_reached:
            if self.done_countdown > 0:
                info["use_random_action"] = True
                self.done_countdown -= 1
            else:  # self.done_countdown == 0:
                done = True

        obs = self._get_obs(wrapped_obs)
        return obs, reward, done, info

    def compute_reward(
        self, achieved_goal: np.ndarray, desired_goal: np.ndarray, info: Dict[str, Any]
    ) -> Union[float, np.ndarray]:
        """
        Returns 0.0 if achieved_goal and desired_goal are close anough, -1.0 otherwise.

        This method is vectorized, which means that if you pass an array of desired and
        achieved goals as input, the method returns an array of rewards.

        :param achieved_goal: Achieved goal or array of achieved goals
        :param desired_goal: Desired goal or array of desired goals
        :param info: For consistency, unused
        :return: Reward or array of rewards, depending on the input
        """
        d = np.linalg.norm(self.weights * (achieved_goal - desired_goal), axis=-1)
        return np.array(d < self.distance_threshold, dtype=np.float64) - 1.0


class Silver(DDPG):
    def __init__(
        self,
        env: Union[GymEnv, str],
        n_envs: int = 1,
        update_logits_freq: int = 1000,
        power: int = -1.0,
        learning_rate: Union[float, Schedule] = 0.001,
        buffer_size: int = 1000000,
        learning_starts: int = 100,
        batch_size: int = 100,
        tau: float = 0.005,
        gamma: float = 0.99,
        train_freq: Union[int, Tuple[int, str]] = 1,
        gradient_steps: int = -1,
        action_noise: Optional[ActionNoise] = None,
        replay_buffer_kwargs: Optional[Dict[str, Any]] = None,
        optimize_memory_usage: bool = False,
        tensorboard_log: Optional[str] = None,
        create_eval_env: bool = False,
        policy_kwargs: Optional[Dict[str, Any]] = None,
        verbose: int = 0,
        seed: Optional[int] = None,
        device: Union[th.device, str] = "auto",
        _init_setup_model: bool = True,
    ):
        # Wrap the env
        def env_func():
            return Goalify(maybe_make_env(env, verbose), 0.1)

        env = make_vec_env(env_func, n_envs=n_envs)
        replay_buffer_kwargs = {} if replay_buffer_kwargs is None else replay_buffer_kwargs
        replay_buffer_kwargs.update({"update_logits_freq": update_logits_freq, "power": power})
        super().__init__(
            "MultiInputPolicy",
            env,
            learning_rate,
            buffer_size,
            learning_starts,
            batch_size,
            tau,
            gamma,
            train_freq,
            gradient_steps,
            action_noise,
            NonParametricSkewedHerReplayBuffer,
            replay_buffer_kwargs,
            optimize_memory_usage,
            tensorboard_log,
            create_eval_env,
            policy_kwargs,
            verbose,
            seed,
            device,
            _init_setup_model,
        )
        for _env in self.env.envs:
            _env.set_buffer(self.replay_buffer)
