import copy
from typing import Any, Dict, Optional, Tuple, Union

import gym
import numpy as np
import torch as th
import torch.distributions as D
from gym import Env, spaces
from stable_baselines3 import SAC, HerReplayBuffer
from stable_baselines3.common.noise import ActionNoise
from stable_baselines3.common.surgeon import Surgeon
from stable_baselines3.common.type_aliases import DictReplayBufferSamples, GymEnv, Schedule
from stable_baselines3.common.vec_env import VecEnv, VecNormalize
from stable_baselines3.her.goal_selection_strategy import GoalSelectionStrategy
from torch import optim


def density(data: th.Tensor, nb_neighbors: int = 5) -> th.Tensor:
    """
    Estimate the density of a batch of tensors using k-nearest neigbors mean distance.

    :param data: The tensors
    :param nb_neighbors: Number of neighbors used for density estimation, defaults to 5
    :return: Density estimation
    """
    dist = th.cdist(data, data, p=2)
    values, _ = dist.topk(k=nb_neighbors, dim=1, largest=False)
    return th.mean(values, dim=1)


EPSILON = 1e-6


class GaussianMixtureModel(D.MixtureSameFamily):
    """
    Gaussian Mixture Model.

    Batch of n Gaussian Mixture Model, each consisting of k random weighted p-variate normal distributions
    self.locs and self.scales should have dimensions (n, k, p), (k, p) if n == 1 or (k,) if n == p == 1.

    :param self.locs: Means of the distributions, dimensions can be either (n, k, p), (k, p) if n == 1 or (k,) if n == p == 1
    :param self.scales: Standard deviation of the distribution, dimenstion must be equal to self.locs
    """

    def __init__(self, probs: th.Tensor, locs: th.Tensor, scales: th.Tensor) -> None:
        assert locs.shape == scales.shape, "self.locs and scale should have the same dimensions"
        mix = D.Categorical(probs)
        comp = D.Independent(D.Normal(locs, scales), 1)
        super().__init__(mix, comp)


# Modified Her with a a method to sample goal in the skew-fit manner.
class ParametricSkewedHerReplayBuffer(HerReplayBuffer):
    """
    Idea from SkewFit, parametric estiamtion of the density.
    """

    def __init__(
        self,
        buffer_size: int,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        env: VecEnv,
        nb_models: int = 100,
        gradient_steps: int = 100,
        batch_size: int = 2048,
        learning_rate: float = 0.01,
        power: float = -1,
        num_presampled_goals: int = 500,
        device: Union[th.device, str] = "cpu",
        n_envs: int = 1,
        optimize_memory_usage: bool = False,
        handle_timeout_termination: bool = True,
        n_sampled_goal: int = 4,
        goal_selection_strategy: Union[GoalSelectionStrategy, str] = "future",
        online_sampling: bool = True,
    ):
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
        # Parameters for distribution model
        self.gradient_steps = gradient_steps
        self.batch_size = batch_size
        # Parameters for goal sampling
        self.power = power
        self.num_presampled_goals = num_presampled_goals
        obs_dim = self.observation_space["achieved_goal"].shape[0]
        self.probs = th.rand(nb_models, requires_grad=True, device=self.device)
        self.locs = th.rand((nb_models, obs_dim), requires_grad=True, device=self.device)
        self.scales = th.rand((nb_models, obs_dim), requires_grad=True, device=self.device)
        self.distribution_optimizer = optim.Adam([self.locs, self.probs, self.scales], lr=learning_rate)

    def sample(self, batch_size: int, env: Optional[VecNormalize] = None) -> DictReplayBufferSamples:
        data = super().sample(5 * batch_size, env)
        goal_distribution = self._get_goal_distribution()
        goals = data.next_observations["achieved_goal"]
        # Importance
        weights = self._get_weights(goals, goal_distribution, self.power)
        p = weights / th.sum(weights)
        # Resampling
        index = p.multinomial(num_samples=batch_size)
        # Normalize if needed and remove extra dimension (we are using only one env for now)

        return DictReplayBufferSamples(
            observations={key: data.observations[key][index] for key in data.observations.keys()},
            next_observations={key: data.next_observations[key][index] for key in data.next_observations.keys()},
            actions=data.actions[index],
            dones=data.dones[index],
            rewards=data.rewards[index],
        )

    def sample_goal(self) -> np.ndarray:
        """
        Sample a goal in the skew-fit manner

        :return: A goal
        """
        is_empty = self.pos == 0 and not self.full
        if not is_empty:
            self._fit_goal_distribution()
            goal_distribution = self._get_goal_distribution()  # estimate goal_distribution using MLE
            goal = self._sample_goal_from_distribution(goal_distribution, self.power, self.num_presampled_goals)
        else:
            goal = self.observation_space["desired_goal"].sample()
        return goal

    def _fit_goal_distribution(self) -> None:
        """
        Fit the goal distribution using Maximum Likelyhood Estimation.
        """
        for _ in range(self.gradient_steps):
            # Sample goals from replay buffer
            samples = self._sample_wo_her(self.batch_size)
            next_achieved_goal = samples.next_observations["achieved_goal"]
            # Create goal distribution
            distribution = GaussianMixtureModel(th.relu(self.probs) + EPSILON, self.locs, th.relu(self.scales) + EPSILON)
            neg_log_likelihood = -th.mean(distribution.log_prob(next_achieved_goal))
            # Optimize the distribution
            self.distribution_optimizer.zero_grad()
            neg_log_likelihood.backward()
            self.distribution_optimizer.step()

    def _sample_wo_her(self, batch_size: int, env: Optional[VecNormalize] = None) -> DictReplayBufferSamples:
        """
        Sample only real elements from the replay buffer (ie no virtual samples).

        :param batch_size: Number of element to sample
        :param env: associated gym VecEnv
            to normalize the observations/rewards when sampling
        :return: The replay data
        """
        upper_bound = self.buffer_size if self.full else self.pos
        env_indices = np.random.randint(self.n_envs, size=batch_size)
        batch_inds = np.random.randint(upper_bound, size=batch_size)
        replay_data = self._get_real_samples(batch_inds, env_indices, env)
        return replay_data

    def _get_goal_distribution(self) -> GaussianMixtureModel:
        """
        Return the estimate goal distribution.
        """
        goal_distribution = GaussianMixtureModel(th.relu(self.probs) + EPSILON, self.locs, th.relu(self.scales) + EPSILON)
        return goal_distribution

    def _sample_goal_from_distribution(
        self, goal_distribution: GaussianMixtureModel, power: float, num_presampled_goals: int
    ) -> th.Tensor:
        """
        Sample a batch of goals, using Sample Importance Resampling.

        :param goal_distribution: The goal distribution
        :param power: Power applied on weights: 0.0 means no SkewFit, -1.0 means uniform sampling
        :param num_presampled_goals: Number of pre-sampled goals used for Sampling Importance Resampling
        :return: Tensor of goals
        """
        # Sampling
        samples = self._sample_wo_her(batch_size=num_presampled_goals)
        goals = samples.next_observations["achieved_goal"]
        # Importance
        weights = self._get_weights(goals, goal_distribution, power)
        p = weights / th.sum(weights)
        # Resampling
        index = p.multinomial(num_samples=1)[0]
        goal = goals[index]
        return goal.detach().cpu().numpy()

    def _get_weights(self, goals: th.Tensor, goal_distribution: GaussianMixtureModel, power: float = -1.0) -> th.Tensor:
        log_prob = goal_distribution.log_prob(goals).squeeze()
        weight = th.exp(log_prob)
        weight = weight**power
        return weight


class NonParametricSkewedHerReplayBuffer(HerReplayBuffer):
    """
    Idea from "Behavior From the Void: Unsupervised Active Pre-Training"
    Non parametric density estimation used as sampling weight.
    """

    def __init__(
        self,
        buffer_size: int,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        env: VecEnv,
        num_presampled_goals: int = 500,
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
        self.num_presampled_goals = num_presampled_goals

    def sample(self, batch_size: int, env: Optional[VecNormalize] = None) -> DictReplayBufferSamples:
        data = super().sample(5 * batch_size, env)
        goals = data.next_observations["achieved_goal"]
        # Importance
        weights = self._get_weights(goals)
        p = weights / th.sum(weights)
        # Resampling
        index = p.multinomial(num_samples=batch_size)
        # Normalize if needed and remove extra dimension (we are using only one env for now)

        return DictReplayBufferSamples(
            observations={key: data.observations[key][index] for key in data.observations.keys()},
            next_observations={key: data.next_observations[key][index] for key in data.next_observations.keys()},
            actions=data.actions[index],
            dones=data.dones[index],
            rewards=data.rewards[index],
        )

    def sample_goal(self) -> np.ndarray:
        """
        Sample a goal in the skew-fit manner

        :return: A goal
        """
        is_empty = self.pos == 0 and not self.full
        if not is_empty:
            goal = self._sample_goal(self.num_presampled_goals)
        else:
            goal = self.observation_space["desired_goal"].sample()
        return goal

    def _sample_goal(self, num_presampled_goals: int) -> th.Tensor:
        """
        Sample a batch of goals, using Sample Importance Resampling.

        :param goal_distribution: The goal distribution
        :param power: Power applied on weights: 0.0 means no SkewFit, -1.0 means uniform sampling
        :param num_presampled_goals: Number of pre-sampled goals used for Sampling Importance Resampling
        :return: Tensor of goals
        """
        # Sampling
        samples = self._sample_wo_her(num_presampled_goals)
        goals = samples.next_observations["achieved_goal"]
        # Importance
        weights = self._get_weights(goals)
        p = weights / th.sum(weights)
        # Resampling
        index = p.multinomial(num_samples=1)[0]
        goal = goals[index]
        return goal.detach().cpu().numpy()

    def _sample_wo_her(self, batch_size: int, env: Optional[VecNormalize] = None) -> DictReplayBufferSamples:
        """
        Sample only real elements from the replay buffer (ie no virtual samples).

        :param batch_size: Number of element to sample
        :param env: associated gym VecEnv
            to normalize the observations/rewards when sampling
        :return: The replay data
        """
        upper_bound = self.buffer_size if self.full else self.pos
        env_indices = np.random.randint(self.n_envs, size=batch_size)
        batch_inds = np.random.randint(upper_bound, size=batch_size)
        replay_data = self._get_real_samples(batch_inds, env_indices, env)
        return replay_data

    def _get_weights(self, goals: th.Tensor) -> th.Tensor:
        ent = density(goals)
        return ent


class Goalify(gym.GoalEnv, gym.Wrapper):
    """
    Wrap the env into a GoalEnv.

    :param env: The environment
    :param cell_factory: The cell factory
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
        # Initial goal; will be re-sampled every time the env is reset
        obs_dim = env.observation_space.shape[0]
        # Set a goalconditionned observation space
        self.observation_space = spaces.Dict(
            {
                "observation": copy.deepcopy(self.env.observation_space),
                "desired_goal": copy.deepcopy(self.env.observation_space),
                "achieved_goal": copy.deepcopy(self.env.observation_space),
            }
        )
        self.goal = self.observation_space["desired_goal"].sample()  # type: np.ndarray
        # The buffer is required to sample goals;
        # When the env is created, the buffer does notalready exists
        # It has to be set after the model creation
        self.buffer = None  # type: _HerReplayBuffer
        # Parameters used for reward computation
        # Weights used to copute the reward: all dimension are not relevant
        self.distance_threshold = distance_threshold
        if weights is None:
            self.weights = np.ones(obs_dim)
        # Go-explore param
        self.nb_random_exploration_steps = nb_random_exploration_steps
        self.window_size = window_size

    def set_buffer(self, buffer: NonParametricSkewedHerReplayBuffer) -> None:
        """
        Set the buffer. Must be done before reset.

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
            if self.done_countdown != 0:
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


class Silver(SAC):
    def __init__(
        self,
        env: Union[GymEnv, str],
        num_presampled_goals: int = 2048,
        learning_rate: Union[float, Schedule] = 0.0003,
        buffer_size: int = 1000000,
        learning_starts: int = 100,
        batch_size: int = 256,
        tau: float = 0.005,
        gamma: float = 0.99,
        train_freq: Union[int, Tuple[int, str]] = 1,
        gradient_steps: int = 1,
        action_noise: Optional[ActionNoise] = None,
        replay_buffer_kwargs: Optional[Dict[str, Any]] = None,
        optimize_memory_usage: bool = False,
        ent_coef: Union[str, float] = "auto",
        target_update_interval: int = 1,
        target_entropy: Union[str, float] = "auto",
        use_sde: bool = False,
        sde_sample_freq: int = -1,
        use_sde_at_warmup: bool = False,
        tensorboard_log: Optional[str] = None,
        create_eval_env: bool = False,
        policy_kwargs: Optional[Dict[str, Any]] = None,
        surgeon: Optional[Surgeon] = None,
        verbose: int = 0,
        seed: Optional[int] = None,
        device: Union[th.device, str] = "auto",
        _init_setup_model: bool = True,
    ):
        # Wrap the env
        env = Goalify(env)
        replay_buffer_kwargs = {} if replay_buffer_kwargs is None else replay_buffer_kwargs
        replay_buffer_kwargs.update(num_presampled_goals=num_presampled_goals)
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
            ent_coef,
            target_update_interval,
            target_entropy,
            use_sde,
            sde_sample_freq,
            use_sde_at_warmup,
            tensorboard_log,
            create_eval_env,
            policy_kwargs,
            surgeon,
            verbose,
            seed,
            device,
            _init_setup_model,
        )
        env.set_buffer(self.replay_buffer)
