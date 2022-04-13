import warnings
from typing import Any, Dict, Optional, Type, Union

import gym
import torch as th
import torch.nn.utils
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, Schedule
from stable_baselines3.common.utils import get_schedule_fn

from sb3_contrib.common.policies import ESPolicy
from sb3_contrib.common.population_based_algorithm import PopulationBasedAlgorithm
from sb3_contrib.common.vec_env.async_eval import AsyncEval


class ARS(PopulationBasedAlgorithm):
    """
    Augmented Random Search: https://arxiv.org/abs/1803.07055

    Original implementation: https://github.com/modestyachts/ARS
    C++/Cuda Implementation: https://github.com/google-research/tiny-differentiable-simulator/
    150 LOC Numpy Implementation: https://github.com/alexis-jacq/numpy_ARS/blob/master/asr.py

    :param policy: The policy to train, can be an instance of ``ARSPolicy``, or a string from ["LinearPolicy", "MlpPolicy"]
    :param env: The environment to train on, may be a string if registered with gym
    :param n_delta: How many random perturbations of the policy to try at each update step.
    :param n_top: How many of the top delta to use in each update step. Default is n_delta
    :param learning_rate: Float or schedule for the step size
    :param delta_std: Float or schedule for the exploration noise
    :param zero_policy: Boolean determining if the passed policy should have it's weights zeroed before training.
    :param alive_bonus_offset: Constant added to the reward at each step, used to cancel out alive bonuses.
    :param n_eval_episodes: Number of episodes to evaluate each candidate.
    :param policy_kwargs: Keyword arguments to pass to the policy on creation
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
        n_delta: int = 8,
        n_top: Optional[int] = None,
        learning_rate: Union[float, Schedule] = 0.02,
        delta_std: Union[float, Schedule] = 0.05,
        zero_policy: bool = True,
        alive_bonus_offset: float = 0,
        n_eval_episodes: int = 1,
        policy_kwargs: Optional[Dict[str, Any]] = None,
        tensorboard_log: Optional[str] = None,
        seed: Optional[int] = None,
        verbose: int = 0,
        device: Union[th.device, str] = "cpu",
        _init_setup_model: bool = True,
    ):

        super().__init__(
            policy,
            env,
            learning_rate=learning_rate,
            pop_size=2 * n_delta,
            alive_bonus_offset=alive_bonus_offset,
            n_eval_episodes=n_eval_episodes,
            tensorboard_log=tensorboard_log,
            policy_kwargs=policy_kwargs,
            verbose=verbose,
            device=device,
            supported_action_spaces=(gym.spaces.Box, gym.spaces.Discrete),
            seed=seed,
        )

        self.n_delta = n_delta
        self.delta_std_schedule = get_schedule_fn(delta_std)

        if n_top is None:
            n_top = n_delta

        # Make sure our hyper parameters are valid and auto correct them if they are not
        if n_top > n_delta:
            warnings.warn(f"n_top = {n_top} > n_delta = {n_top}, setting n_top = n_delta")
            n_top = n_delta

        self.n_top = n_top

        self.zero_policy = zero_policy
        self.weights = None  # Need to call init model to initialize weight

        if _init_setup_model:
            self._setup_model()

    def _setup_model(self) -> None:
        self._setup_lr_schedule()
        self.set_random_seed(self.seed)

        self.policy = self.policy_class(self.observation_space, self.action_space, **self.policy_kwargs)
        self.policy = self.policy.to(self.device)
        self.weights = th.nn.utils.parameters_to_vector(self.policy.parameters()).detach()
        self.n_params = len(self.weights)

        if self.zero_policy:
            self.weights = th.zeros_like(self.weights, requires_grad=False)
            self.policy.load_from_vector(self.weights.cpu())

    def _do_one_update(self, callback: BaseCallback, async_eval: Optional[AsyncEval]) -> None:
        """
        Sample new candidates, evaluate them and then update current policy.

        :param callback: callback(s) called at every step with state of the algorithm.
        :param async_eval: The object for asynchronous evaluation of candidates.
        """
        # Retrieve current parameter noise standard deviation
        # and current learning rate
        delta_std = self.delta_std_schedule(self._current_progress_remaining)
        learning_rate = self.lr_schedule(self._current_progress_remaining)
        # Sample the parameter noise, it will be scaled by delta_std
        deltas = th.normal(mean=0.0, std=1.0, size=(self.n_delta, self.n_params), device=self.device)
        policy_deltas = deltas * delta_std
        # Generate 2 * n_delta candidate policies by adding noise to the current weights
        candidate_weights = th.cat([self.weights + policy_deltas, self.weights - policy_deltas])

        with th.no_grad():
            candidate_returns = self.evaluate_candidates(candidate_weights, callback, async_eval)

        # Returns corresponding to weights + deltas
        plus_returns = candidate_returns[: self.n_delta]
        # Returns corresponding to weights - deltas
        minus_returns = candidate_returns[self.n_delta :]

        # Keep only the top performing candidates for update
        top_returns, _ = th.max(th.vstack((plus_returns, minus_returns)), dim=0)
        top_idx = th.argsort(top_returns, descending=True)[: self.n_top]
        plus_returns = plus_returns[top_idx]
        minus_returns = minus_returns[top_idx]
        deltas = deltas[top_idx]

        # Scale learning rate by the return standard deviation:
        # take smaller steps when there is a high variance in the returns
        return_std = th.cat([plus_returns, minus_returns]).std()
        step_size = learning_rate / (self.n_top * return_std + 1e-6)
        # Approximate gradient step
        self.weights = self.weights + step_size * ((plus_returns - minus_returns) @ deltas)
        self.policy.load_from_vector(self.weights.cpu())

        self.logger.record("train/iterations", self._n_updates, exclude="tensorboard")
        self.logger.record("train/delta_std", delta_std)
        self.logger.record("train/learning_rate", learning_rate)
        self.logger.record("train/step_size", step_size.item())
        self.logger.record("rollout/return_std", return_std.item())

        self._n_updates += 1

    def learn(
        self,
        total_timesteps: int,
        callback: MaybeCallback = None,
        log_interval: int = 1,
        tb_log_name: str = "ARS",
        eval_env: Optional[GymEnv] = None,
        eval_freq: int = -1,
        n_eval_episodes: int = 5,
        eval_log_path: Optional[str] = None,
        reset_num_timesteps: bool = True,
        async_eval: Optional[AsyncEval] = None,
    ) -> "ARS":
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
