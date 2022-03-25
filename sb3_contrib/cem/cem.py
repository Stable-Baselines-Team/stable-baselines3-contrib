import warnings
from typing import Any, Dict, Optional, Type, Union

import gym
import torch as th
import torch.nn.utils
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.policies import BasePolicy
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback
from torch.distributions.multivariate_normal import MultivariateNormal

from sb3_contrib.common.policies import ESPolicy
from sb3_contrib.common.population_based_algorithm import PopulationBasedAlgorithm
from sb3_contrib.common.vec_env.async_eval import AsyncEval


class CEM(PopulationBasedAlgorithm):
    """
    Noisy Cross Entropy Method: http://dx.doi.org/10.1162/neco.2006.18.12.2936
    "Learning Tetris Using the Noisy Cross-Entropy Method"

    CEM is part of the Evolution Strategies (ES), which does black-box optimization
    by sampling and evaluating a population of candidates:
    https://blog.otoro.net/2017/10/29/visual-evolution-strategies/

    John Schulman's implementation: https://github.com/joschu/modular_rl/blob/master/modular_rl/cem.py

    :param policy: The policy to train, can be an instance of ``ESPolicy``, or a string from ["LinearPolicy", "MlpPolicy"]
    :param env: The environment to train on, may be a string if registered with gym
    :param pop_size: Population size (number of individuals)
    :param n_top: How many of the top individuals to use in each update step. Default is pop_size
    :param initial_std: Initial standard deviation for the exploration noise,
        by default using Pytorch default variance at initialization.
    :param extra_noise_std: Initial standard deviation for the extra noise added to the covariance matrix
    :param noise_multiplier: Noise decay. We add noise to the standard deviation
        to avoid early collapse.
    :param zero_policy: Boolean determining if the passed policy should have it's weights zeroed before training.
    :param alive_bonus_offset: Constant added to the reward at each step, used to cancel out alive bonuses.
    :param n_eval_episodes: Number of episodes to evaluate each candidate.
    :param policy_kwargs: Keyword arguments to pass to the policy on creation
    :param policy_base: Base class to use for the policy
    :param tensorboard_log: String with the directory to put tensorboard logs:
    :param seed: Random seed for the training
    :param verbose: Verbosity level: 0 no output, 1 info, 2 debug
    :param device: Torch device to use for training, defaults to "cpu"
    :param _init_setup_model: Whether or not to build the network at the creation of the instance
    """

    def __init__(
        self,
        policy: Union[str, Type[ESPolicy]],
        env: Union[GymEnv, str],
        pop_size: int = 16,
        n_top: Optional[int] = None,
        initial_std: Optional[float] = None,
        extra_noise_std: float = 0.2,
        noise_multiplier: float = 0.999,
        zero_policy: bool = False,
        use_diagonal_covariance: bool = False,
        alive_bonus_offset: float = 0,
        n_eval_episodes: int = 1,
        policy_kwargs: Optional[Dict[str, Any]] = None,
        policy_base: Type[BasePolicy] = ESPolicy,
        tensorboard_log: Optional[str] = None,
        seed: Optional[int] = None,
        verbose: int = 0,
        device: Union[th.device, str] = "cpu",
        _init_setup_model: bool = True,
    ):

        super().__init__(
            policy,
            env,
            learning_rate=0.0,
            pop_size=pop_size,
            alive_bonus_offset=alive_bonus_offset,
            n_eval_episodes=n_eval_episodes,
            tensorboard_log=tensorboard_log,
            policy_base=policy_base,
            policy_kwargs=policy_kwargs,
            verbose=verbose,
            device=device,
            supported_action_spaces=(gym.spaces.Box, gym.spaces.Discrete),
            seed=seed,
        )

        self.initial_std = initial_std
        self.extra_noise_std = extra_noise_std
        self.noise_multiplier = noise_multiplier

        if n_top is None:
            n_top = self.pop_size

        # Make sure our hyper parameters are valid and auto correct them if they are not
        if n_top > self.pop_size:
            warnings.warn(f"n_top = {n_top} > pop_size = {self.pop_size}, setting n_top = pop_size")
            n_top = self.pop_size

        self.n_top = n_top
        self.zero_policy = zero_policy
        self.weights = None  # Need to call init model to initialize weights
        self.centroid_cov = None
        self.use_diagonal_covariance = use_diagonal_covariance
        self.extra_variance = None

        if _init_setup_model:
            self._setup_model()

    def _setup_model(self) -> None:
        self._setup_lr_schedule()
        self.set_random_seed(self.seed)

        self.policy = self.policy_class(self.observation_space, self.action_space, **self.policy_kwargs)
        self.policy = self.policy.to(self.device)
        self.weights = th.nn.utils.parameters_to_vector(self.policy.parameters()).detach()
        self.n_params = len(self.weights)

        if self.initial_std is None:
            # Use weights variance from Pytorch initialization by default
            initial_variance = self.weights.var().item()
        else:
            initial_variance = self.initial_std**2
        self.centroid_cov = th.ones_like(self.weights, requires_grad=False) * initial_variance
        if not self.use_diagonal_covariance:
            self.centroid_cov = th.diag(self.centroid_cov)
        # Initial extra noise vector (extra variance)
        self.extra_variance = th.ones_like(self.weights, requires_grad=False) * self.extra_noise_std**2

        if self.zero_policy:
            self.weights = th.zeros_like(self.weights, requires_grad=False)
            self.policy.load_from_vector(self.weights.cpu())

    def _do_one_update(self, callback: BaseCallback, async_eval: Optional[AsyncEval]) -> None:
        """
        Sample new candidates, evaluate them and then update current policy.

        :param callback: callback(s) called at every step with state of the algorithm.
        :param async_eval: The object for asynchronous evaluation of candidates.
        """

        if self.use_diagonal_covariance:
            # Sample using only the diagonal of the covariance matrix (+ extra noise)
            param_noise = th.normal(mean=0.0, std=1.0, size=(self.pop_size, self.n_params), device=self.device)
            policy_deltas = param_noise * th.sqrt(self.centroid_cov + self.extra_variance)
        else:
            # Sample using the full covariance matrix (+ extra noise)
            sample_cov = self.centroid_cov + th.diag(self.extra_variance)
            param_noise_distribution = MultivariateNormal(th.zeros_like(self.weights), covariance_matrix=sample_cov)
            policy_deltas = param_noise_distribution.sample((self.pop_size,))

        candidate_weights = self.weights + policy_deltas

        with th.no_grad():
            candidate_returns = self.evaluate_candidates(candidate_weights, callback, async_eval)

        # Keep only the top performing candidates for update
        top_idx = th.argsort(candidate_returns, descending=True)[: self.n_top]

        # Update centroid: barycenter of the best candidates
        self.weights = candidate_weights[top_idx].mean(dim=0)
        if self.use_diagonal_covariance:
            # Do not compute full cov matrix when use_diagonal_covariance=True
            self.centroid_cov = candidate_weights[top_idx].var(dim=0)
        else:
            # transpose to mimic rowvar=False in np.cov()
            self.centroid_cov = th.cov(candidate_weights[top_idx].transpose(-1, -2))

        # Update extra variance (prevents early converge)
        self.extra_noise_std = self.extra_noise_std * self.noise_multiplier
        self.extra_variance = th.ones_like(self.weights, requires_grad=False) * self.extra_noise_std**2

        # Current policy is the centroid of the best candidates
        self.policy.load_from_vector(self.weights.cpu())

        self.logger.record("rollout/return_std", candidate_returns.std().item())
        self.logger.record("train/iterations", self._n_updates, exclude="tensorboard")
        if self.use_diagonal_covariance:
            self.logger.record("train/diag_std", th.mean(th.sqrt(self.centroid_cov)).item())
        else:
            self.logger.record("train/diag_std", th.mean(th.sqrt(th.diagonal(self.centroid_cov))).item())
        self.logger.record("train/extra_noise_std", self.extra_noise_std)

        self._n_updates += 1

    def learn(
        self,
        total_timesteps: int,
        callback: MaybeCallback = None,
        log_interval: int = 1,
        tb_log_name: str = "CEM",
        eval_env: Optional[GymEnv] = None,
        eval_freq: int = -1,
        n_eval_episodes: int = 5,
        eval_log_path: Optional[str] = None,
        reset_num_timesteps: bool = True,
        async_eval: Optional[AsyncEval] = None,
    ) -> "CEM":
        """
        Return a trained model.

        :param total_timesteps: The total number of samples (env steps) to train on
        :param callback: callback(s) called at every step with state of the algorithm.
        :param log_interval: The number of timesteps before logging.
        :param tb_log_name: the name of the run for TensorBoard logging
        :param eval_env: Environment that will be used to evaluate the agent
        :param eval_freq: Evaluate the agent every ``eval_freq`` timesteps (this may vary a little)
        :param n_eval_episodes: Number of episode to evaluate the agent
        :param eval_log_path: Path to a folder where the evaluations will be saved
        :param reset_num_timesteps: whether or not to reset the current timestep number (used in logging)
        :param async_eval: The object for asynchronous evaluation of candidates.
        :return: the trained model
        """
        return super().learn(
            total_timesteps=total_timesteps,
            callback=callback,
            log_interval=log_interval,
            eval_env=eval_env,
            eval_freq=eval_freq,
            n_eval_episodes=n_eval_episodes,
            tb_log_name=tb_log_name,
            eval_log_path=eval_log_path,
            reset_num_timesteps=reset_num_timesteps,
            async_eval=async_eval,
        )
