import copy
import time
import warnings
from functools import partial
from typing import Any, Dict, Optional, Type, Union

import gym
import numpy as np
import torch as th
import torch.nn.utils
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.policies import BasePolicy
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback
from stable_baselines3.common.utils import safe_mean
from torch.distributions.multivariate_normal import MultivariateNormal

from sb3_contrib.ars.policies import ARSPolicy
from sb3_contrib.common.vec_env.async_eval import AsyncEval


class CEM(BaseAlgorithm):
    """
    Noisy Cross Entropy Method: http://dx.doi.org/10.1162/neco.2006.18.12.2936
    "Learning Tetris Using the Noisy Cross-Entropy Method"

    CEM is part of the Evolution Strategies (ES), which does black-box optimization
    by sampling and evaluating a population of candidates:
    https://blog.otoro.net/2017/10/29/visual-evolution-strategies/

    John Schulman's implementation: https://github.com/joschu/modular_rl/blob/master/modular_rl/cem.py

    :param policy: The policy to train, can be an instance of ``ARSPolicy``, or a string from ["LinearPolicy", "MlpPolicy"]
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
        policy: Union[str, Type[ARSPolicy]],
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
        policy_base: Type[BasePolicy] = ARSPolicy,
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
            tensorboard_log=tensorboard_log,
            policy_base=policy_base,
            policy_kwargs=policy_kwargs,
            verbose=verbose,
            device=device,
            supported_action_spaces=(gym.spaces.Box, gym.spaces.Discrete),
            support_multi_env=True,
            seed=seed,
        )

        self.pop_size = pop_size
        self.initial_std = initial_std
        self.extra_noise_std = extra_noise_std
        self.noise_multiplier = noise_multiplier
        self.n_eval_episodes = n_eval_episodes

        if n_top is None:
            n_top = self.pop_size

        # Make sure our hyper parameters are valid and auto correct them if they are not
        if n_top > self.pop_size:
            warnings.warn(f"n_top = {n_top} > pop_size = {self.pop_size}, setting n_top = pop_size")
            n_top = self.pop_size

        self.n_top = n_top

        self.alive_bonus_offset = alive_bonus_offset
        self.zero_policy = zero_policy
        self.weights = None  # Need to call init model to initialize weights
        self.centroid_cov = None
        self.use_diagonal_covariance = use_diagonal_covariance
        self.extra_variance = None
        self.processes = None
        # Keep track of how many steps where elapsed before a new rollout
        # Important for syncing observation normalization between workers
        self.old_count = 0

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

    def _mimic_monitor_wrapper(self, episode_rewards: np.ndarray, episode_lengths: np.ndarray) -> None:
        """
        Helper to mimic Monitor wrapper and report episode statistics (mean reward, mean episode length).

        :param episode_rewards: List containing per-episode rewards
        :param episode_lengths:  List containing per-episode lengths (in number of steps)
        """
        # Mimic Monitor Wrapper
        infos = [
            {"episode": {"r": episode_reward, "l": episode_length}}
            for episode_reward, episode_length in zip(episode_rewards, episode_lengths)
        ]

        self._update_info_buffer(infos)

    def _trigger_callback(
        self,
        _locals: Dict[str, Any],
        _globals: Dict[str, Any],
        callback: BaseCallback,
        n_envs: int,
    ) -> None:
        """
        Callback passed to the ``evaluate_policy()`` helper
        in order to increment the number of timesteps
        and trigger events in the single process version.

        :param _locals:
        :param _globals:
        :param callback: Callback that will be called at every step
        :param n_envs: Number of environments
        """
        self.num_timesteps += n_envs
        callback.on_step()

    def evaluate_candidates(
        self, candidate_weights: th.Tensor, callback: BaseCallback, async_eval: Optional[AsyncEval]
    ) -> th.Tensor:
        """
        Evaluate each candidate.

        :param candidate_weights: The candidate weights to be evaluated.
        :param callback: Callback that will be called at each step
            (or after evaluation in the multiprocess version)
        :param async_eval: The object for asynchronous evaluation of candidates.
        :return: The episodic return for each candidate.
        """

        batch_steps = 0
        # returns == sum of rewards
        candidate_returns = th.zeros(self.pop_size, device=self.device)
        train_policy = copy.deepcopy(self.policy)
        # Empty buffer to show only mean over one iteration (one set of candidates) in the logs
        self.ep_info_buffer = []
        callback.on_rollout_start()

        if async_eval is not None:
            # Multiprocess asynchronous version
            async_eval.send_jobs(candidate_weights, self.pop_size)
            results = async_eval.get_results()

            for weights_idx, (episode_rewards, episode_lengths) in results:

                # Update reward to cancel out alive bonus if needed
                candidate_returns[weights_idx] = sum(episode_rewards) + self.alive_bonus_offset * sum(episode_lengths)
                batch_steps += np.sum(episode_lengths)
                self._mimic_monitor_wrapper(episode_rewards, episode_lengths)

            # Combine the filter stats of each process for normalization
            for worker_obs_rms in async_eval.get_obs_rms():
                if self._vec_normalize_env is not None:
                    # worker_obs_rms.count -= self.old_count
                    self._vec_normalize_env.obs_rms.combine(worker_obs_rms)
                    # Hack: don't count timesteps twice (between the two are synced)
                    # otherwise it will lead to overflow,
                    # in practice we would need two RunningMeanStats
                    self._vec_normalize_env.obs_rms.count -= self.old_count

            # Synchronise VecNormalize if needed
            if self._vec_normalize_env is not None:
                async_eval.sync_obs_rms(self._vec_normalize_env.obs_rms.copy())
                self.old_count = self._vec_normalize_env.obs_rms.count

            # Hack to have Callback events
            for _ in range(batch_steps // len(async_eval.remotes)):
                self.num_timesteps += len(async_eval.remotes)
                callback.on_step()
        else:
            # Single process, synchronous version
            for weights_idx in range(self.pop_size):

                # Load current candidate weights
                train_policy.load_from_vector(candidate_weights[weights_idx].cpu())
                # Evaluate the candidate
                episode_rewards, episode_lengths = evaluate_policy(
                    train_policy,
                    self.env,
                    n_eval_episodes=self.n_eval_episodes,
                    return_episode_rewards=True,
                    # Increment num_timesteps too (slight mismatch with multi envs)
                    callback=partial(self._trigger_callback, callback=callback, n_envs=self.env.num_envs),
                    warn=False,
                )
                # Update reward to cancel out alive bonus if needed
                candidate_returns[weights_idx] = sum(episode_rewards) + self.alive_bonus_offset * sum(episode_lengths)
                batch_steps += sum(episode_lengths)
                self._mimic_monitor_wrapper(episode_rewards, episode_lengths)

            # Note: we increment the num_timesteps inside the evaluate_policy()
            # however when using multiple environments, there will be a slight
            # mismatch between the number of timesteps used and the number
            # of calls to the step() method (cf. implementation of evaluate_policy())
            # self.num_timesteps += batch_steps

        callback.on_rollout_end()

        return candidate_returns

    def _log_and_dump(self) -> None:
        """
        Dump information to the logger.
        """
        time_elapsed = time.time() - self.start_time
        fps = int((self.num_timesteps - self._num_timesteps_at_start) / (time_elapsed + 1e-8))
        if len(self.ep_info_buffer) > 0 and len(self.ep_info_buffer[0]) > 0:
            self.logger.record("rollout/ep_rew_mean", safe_mean([ep_info["r"] for ep_info in self.ep_info_buffer]))
            self.logger.record("rollout/ep_len_mean", safe_mean([ep_info["l"] for ep_info in self.ep_info_buffer]))
        self.logger.record("time/fps", fps)
        self.logger.record("time/time_elapsed", int(time_elapsed), exclude="tensorboard")
        self.logger.record("time/total_timesteps", self.num_timesteps, exclude="tensorboard")
        self.logger.dump(step=self.num_timesteps)

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

        total_steps, callback = self._setup_learn(
            total_timesteps, eval_env, callback, eval_freq, n_eval_episodes, eval_log_path, reset_num_timesteps, tb_log_name
        )

        callback.on_training_start(locals(), globals())

        while self.num_timesteps < total_steps:
            self._update_current_progress_remaining(self.num_timesteps, total_timesteps)
            self._do_one_update(callback, async_eval)
            if log_interval is not None and self._n_updates % log_interval == 0:
                self._log_and_dump()

        if async_eval is not None:
            async_eval.close()

        callback.on_training_end()

        return self
