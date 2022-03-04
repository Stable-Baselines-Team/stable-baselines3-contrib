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
from stable_baselines3.common.type_aliases import GymEnv, Schedule
from stable_baselines3.common.vec_env import VecEnv
from stable_baselines3.common.vec_env.base_vec_env import VecEnv
from stable_baselines3.her.goal_selection_strategy import GoalSelectionStrategy


def log_prob(samples: np.ndarray, x: np.ndarray) -> np.ndarray:
    """
    Estimate the density of a distribution from samples, and return the logprob of x.

    :param samples: The samples from the distribution to estimate
    :param x: The value to get the logprob
    :return: The log probabilties of x, as array
    """

    kde = KernelDensity(bandwidth=0.2)
    kde.fit(samples)
    return kde.score_samples(x)


class NonParametricSkewedHerReplayBuffer(HerReplayBuffer):
    """
    Idea from "Behavior From the Void: Unsupervised Active Pre-Training"
    Non parametric density estimation with KDE used as sampling weight.
    """

    def __init__(
        self,
        buffer_size: int,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        env: VecEnv,
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
        self.weights = np.ones((self.buffer_size, self.n_envs))

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
        if self.pos % 1000 == 0:
            self.update_weights()

    def sample_goal(self) -> np.ndarray:
        """
        Sample goal in the skewed manner.

        :return: A goal
        """
        is_empty = self.pos == 0 and not self.full
        if not is_empty:
            goal = self._sample_goal()
        else:
            goal = self.observation_space["desired_goal"].sample()
        return goal

    def _sample_goal(self) -> np.ndarray:
        """
        Sample a goal goals, using Sample Importance Resampling.

        :return: Goal as an array
        """
        upper_bound = self.buffer_size if self.full else self.pos
        data = self.next_observations["achieved_goal"][:upper_bound]  # (n, n_envs, obs_shape)
        weights = self.weights[:upper_bound]  # (n * n_envs, obs_shape)
        data = data.reshape(-1, data.shape[-1])  # (n * n_envs, obs_shape)
        weights = weights.flatten()  # (n * n_envs, obs_shape)
        p = weights / np.sum(weights)
        goal_idx = np.random.choice(p.shape[0], p=p)
        goal = data[goal_idx]
        return goal

    def update_weights(self) -> None:
        """
        Update the weights of the buffer.
        """
        upper_bound = self.buffer_size if self.full else self.pos
        data = self.next_observations["achieved_goal"][:upper_bound]  # (n, n_envs, obs_shape)
        data = data.reshape(-1, data.shape[-1])  # (n*n_envs, obs_shape)
        samples = super().sample(1024).next_observations["achieved_goal"].detach().cpu().numpy()
        _log_prob = log_prob(samples, data)
        _log_prob = _log_prob - _log_prob.mean()
        _log_prob = np.clip(_log_prob, a_min=-5, a_max=None)
        weights = np.exp(-1.2 * _log_prob)
        weights[weights < np.quantile(weights, 0.5)] = 0
        self.weights = weights.reshape(upper_bound, -1)


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
