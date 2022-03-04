import copy
from typing import Any, Dict, Optional, Tuple, Union

import gym
import numpy as np
import torch as th
import torch.distributions as D
from gym import spaces
from stable_baselines3.common.buffers import DictReplayBuffer
from torch import optim

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


class Goalify(gym.GoalEnv, gym.Wrapper):
    """
    Wrap an env into a GoalEnv.

    The goal is to reach a specific goal. The achived_goal and the observation are egals.
    If the goal is reached, reward is 0.0 and done is True. Otherwise, reward is -1.0 and done
    is equal to inner done.

    :param env: The environment
    :param nb_models: Number of mixed gaussian models used to model goal distribution, defaults to 100
    :param gradient_steps: How many MLE gradient steps during reset
    :param batch_size: Minibatch size for each gradient update
    :param learning_rate: Learning rate for Adam optimizer
    :param power:
    :param num_presampled_goals: Number of samples used when using Sample Importance Resampling, defaults to 10
    :param distance_threshold: Success if euclidian distance between desired and achived
        goal is less than distance threshold
    :param weights: Weights applied before the distance computation. If a dimension is not relevant,
        set its weight to 0.0.
    """

    def __init__(
        self,
        env: gym.Env,
        nb_models: int = 100,
        gradient_steps: int = 100,
        batch_size: int = 2048,
        learning_rate: float = 0.01,
        power: float = -1,
        num_presampled_goals: int = 500,
        distance_threshold: float = 0.5,
        weights: Optional[np.ndarray] = None,
    ) -> None:
        super().__init__(env)
        # Initial goal; will be re-sampled every time the env is reset
        obs_dim = self.observation_space.shape[0]
        self.goal = self.observation_space.sample()

        # Set a goalconditionned observation space
        self.observation_space = spaces.Dict(
            {
                "observation": copy.deepcopy(self.env.observation_space),
                "desired_goal": copy.deepcopy(self.env.observation_space),
                "achieved_goal": copy.deepcopy(self.env.observation_space),
            }
        )
        # The buffer is required to sample goals;
        # When the env is created, the buffer does notalready exists
        # It has to be set after the model creation
        self.buffer = None  # type: DictReplayBuffer
        # Parameters for distribution model
        self.gradient_steps = gradient_steps
        self.batch_size = batch_size
        self.probs = th.rand(nb_models, requires_grad=True)
        self.locs = th.rand((nb_models, obs_dim), requires_grad=True)
        self.scales = th.rand((nb_models, obs_dim), requires_grad=True)
        self.distribution_optimizer = optim.Adam([self.locs, self.probs, self.scales], lr=learning_rate)
        # Parameters for goal sampling
        self.power = power
        self.num_presampled_goals = num_presampled_goals
        # Parameters used for reward computation
        # Weights used to copute the reward: all dimension are not relevant
        self.distance_threshold = distance_threshold
        if weights is None:
            self.weights = np.ones(obs_dim)

    def set_buffer(self, buffer: DictReplayBuffer) -> None:
        """
        Set the buffer.

        :param buffer: The replay buffer of the model
        """
        self.buffer = buffer
        self.probs.to(buffer.device)
        self.locs.to(buffer.device)
        self.scales.to(buffer.device)

    def step(self, action: np.ndarray) -> Tuple[Dict[str, np.ndarray], float, bool, Dict[str, Any]]:
        obs, reward, done, info = self.env.step(action)
        reward = self.compute_reward(obs, self.goal, {})
        is_success = float(reward == 0.0)
        done = is_success or done
        info["is_success"] = is_success
        return self._get_dict_obs(obs), reward, done, info

    def reset(self) -> Dict[str, np.ndarray]:
        obs = self.env.reset()
        is_empty = self.buffer.pos == 0 and not self.buffer.full
        if not is_empty:
            self.fit_goal_distribution()
            goal_distribution = self.get_goal_distribution()  # estimate goal_distribution using MLE
            self.goal = self.sample_goal(goal_distribution, self.power, self.num_presampled_goals)
        else:
            self.goal = self.observation_space["desired_goal"].sample()
        return self._get_dict_obs(obs)

    def _get_dict_obs(self, obs: np.ndarray) -> Dict[str, np.ndarray]:
        achieved_goal = np.copy(obs)
        desired_goal = np.copy(self.goal)
        return {
            "observation": obs,
            "desired_goal": desired_goal,
            "achieved_goal": achieved_goal,
        }

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

    def fit_goal_distribution(self) -> None:
        """
        Fit the goal distribution using Maximum Likelyhood Estimation.
        """
        for _ in range(self.gradient_steps):
            # Sample goals from replay buffer
            samples = self.buffer.sample(self.batch_size)
            next_achieved_goal = samples.next_observations["achieved_goal"]
            # Create goal distribution
            distribution = GaussianMixtureModel(th.relu(self.probs) + EPSILON, self.locs, th.relu(self.scales) + EPSILON)
            neg_log_likelihood = -th.mean(distribution.log_prob(next_achieved_goal))
            # Optimize the distribution
            self.distribution_optimizer.zero_grad()
            neg_log_likelihood.backward()
            self.distribution_optimizer.step()

    def get_goal_distribution(self) -> GaussianMixtureModel:
        """
        Return the estimate goal distribution.
        """
        goal_distribution = GaussianMixtureModel(th.relu(self.probs) + EPSILON, self.locs, th.relu(self.scales) + EPSILON)
        return goal_distribution

    def sample_goal(self, goal_distribution: GaussianMixtureModel, power: float, num_presampled_goals: int) -> th.Tensor:
        """
        Sample a batch of goals, using Sample Importance Resampling.

        :param goal_distribution: The goal distribution
        :param power: Power applied on weights: 0.0 means no SkewFit, -1.0 means uniform sampling
        :param num_presampled_goals: Number of pre-sampled goals used for Sampling Importance Resampling
        :return: Tensor of goals
        """
        # Sampling
        samples = self.buffer.sample(batch_size=num_presampled_goals)
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
