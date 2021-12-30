import copy
import time
import warnings
from typing import Any, Dict, List, Optional, Tuple, Type, Union

import gym
import numpy as np
import torch as th
import torch.nn.utils
from mpire import WorkerPool
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.policies import BasePolicy
from stable_baselines3.common.running_mean_std import RunningMeanStd
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, Schedule
from stable_baselines3.common.utils import get_schedule_fn, safe_mean
from stable_baselines3.common.vec_env import VecEnv, unwrap_vec_normalize

from sb3_contrib.ars.policies import ARSPolicy


class ARS(BaseAlgorithm):
    """
    Augmented Random Search: https://arxiv.org/abs/1803.07055

    Original implementation: https://github.com/modestyachts/ARS
    C++/Cuda Implementation: https://github.com/google-research/tiny-differentiable-simulator/
    Numpy Implementation: https://github.com/alexis-jacq/numpy_ARS/blob/master/asr.py

    :param policy: The policy to train, can be an instance of ARSPolicy, or a string
    :param env: The environment to train on, may be a string if registred with gym
    :param n_delta: How many random pertubations of the policy to try at each update step.
    :param n_top: How many of the top delta to use in each update step. Default is n_delta
    :param learning_rate: Float or schedule for the step size
    :param delta_std: Float or schedule for the exploration noise
    :param zero_policy: Boolean determining if the passed policy should have it's weights zeroed before training, default True.
    :param alive_bonus_offset: Constant added to the reward at each step, a value of -1 is used in the original paper
    :param n_eval_episodes: Number of episodes to evaluate each candidate.
    :param policy_kwargs: Keyword arguments to pass to the policy on creation
    :param policy_base: Base class to use for the policy
    :param tensorboard_log: String with the directory to put tensorboard logs:
    :param seed: Random seed for the training
    :param verbose: Verbosity level: 0 no output, 1 info, 2 debug
    :param device: Torch device to use for training, defaults to "auto"
    :param _init_setup_model: Whether or not to build the network at the creation of the instance
    """

    def __init__(
        self,
        policy: Union[str, Type[ARSPolicy]],
        env: Union[GymEnv, str],
        n_delta: int = 64,
        n_top: Optional[int] = None,
        learning_rate: Union[float, Schedule] = 0.05,
        delta_std: Union[float, Schedule] = 0.05,
        zero_policy: bool = True,
        alive_bonus_offset: float = 0,
        n_eval_episodes: int = 1,
        policy_kwargs: Optional[Dict[str, Any]] = None,
        policy_base: Type[BasePolicy] = ARSPolicy,
        tensorboard_log: Optional[str] = None,
        seed: Optional[int] = None,
        verbose: int = 0,
        device: Union[th.device, str] = "auto",
        _init_setup_model: bool = True,
    ):

        super().__init__(
            policy,
            env,
            learning_rate=learning_rate,
            tensorboard_log=tensorboard_log,
            policy_base=policy_base,
            policy_kwargs=policy_kwargs,
            verbose=verbose,
            device=device,
            supported_action_spaces=(gym.spaces.Box, gym.spaces.Discrete),
            support_multi_env=True,
            seed=seed,
        )

        self.n_delta = n_delta
        self.pop_size = 2 * n_delta
        self.delta_std_schedule = get_schedule_fn(delta_std)
        self.n_eval_episodes = n_eval_episodes

        if n_top is None:
            n_top = n_delta
        self.n_top = n_top

        self.alive_bonus_offset = alive_bonus_offset
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
            self.policy.load_from_vector(self.weights)

    @staticmethod
    def _update_running_stats(rms_1: RunningMeanStd, rms_2: RunningMeanStd) -> None:
        # from https://github.com/modestyachts/ARS/blob/master/code/filter.py
        count_1 = rms_1.count
        count_2 = rms_2.count
        total_count = count_1 + count_2
        delta = rms_1.mean - rms_2.mean
        rms_1.count = total_count
        rms_1.mean = (count_1 * rms_1.mean + count_2 * rms_2.mean) / total_count
        rms_1.var = rms_1.var + rms_2.var + np.square(delta) * count_1 * count_2 / total_count

    @staticmethod
    def _sync_running_stats(rms_1: RunningMeanStd, rms_2: RunningMeanStd) -> None:
        rms_2.count = rms_1.count
        rms_2.mean = rms_1.mean.copy()
        rms_2.var = rms_1.var.copy()

    def _collect_rollouts(self, policy_deltas: th.Tensor, callback: BaseCallback, envs: List[VecEnv]) -> Tuple[th.Tensor, int]:
        batch_steps = 0

        # Generate 2*n_delta candidate policies by adding noise to the current weight
        candidate_weights = th.cat([self.weights + policy_deltas, self.weights - policy_deltas])
        candidate_returns = th.zeros(self.pop_size)  # returns == sum of rewards
        train_policy = copy.deepcopy(self.policy)
        self.ep_info_buffer = []

        # Multiprocess version
        if len(envs) > 1:

            def worker_init(worker_state: Dict[str, Any]) -> None:
                worker_state["n_envs"] = len(envs)
                worker_state["candidate_weights"] = candidate_weights
                worker_state["train_policy"] = copy.deepcopy(train_policy)

            def evaluate(worker_state: Dict[str, Any], weights_idx: int):
                worker_state["train_policy"].load_from_vector(worker_state["candidate_weights"][weights_idx])

                episode_rewards, episode_lengths = evaluate_policy(
                    worker_state["train_policy"],
                    envs[weights_idx % worker_state["n_envs"]],
                    n_eval_episodes=self.n_eval_episodes,
                    return_episode_rewards=True,
                    # Note: callback will be called by every process,
                    # there will be a race condition...
                    callback=lambda _locals, _globals: callback.on_step(),
                    warn=False,
                )
                return weights_idx, (episode_rewards, episode_lengths)

            callback.on_rollout_start()

            with WorkerPool(len(envs), use_worker_state=True, keep_alive=False) as pool:
                results = pool.map(evaluate, range(self.pop_size), worker_init=worker_init, chunk_size=1)

                for weights_idx, (episode_rewards, episode_lengths) in results:

                    # Check when using multiple episodes for evaluation
                    candidate_returns[weights_idx] = sum(episode_rewards) + self.alive_bonus_offset * sum(episode_lengths)
                    batch_steps += np.sum(episode_lengths)
                    # Mimic Monitor Wrapper
                    infos = [
                        {"episode": {"r": episode_reward, "l": episode_length}}
                        for episode_reward, episode_length in zip(episode_rewards, episode_lengths)
                    ]

                    self._update_info_buffer(infos)

            callback.on_rollout_end()
            # Sync VecNormalize if needed
            if self._vec_normalize_env is not None:
                vec_normalize_envs = [unwrap_vec_normalize(env) for env in envs]
                # Update normalization from the workers
                # Note: we are not updating the return rms
                for i in range(len(envs)):
                    self._update_running_stats(self._vec_normalize_env.obs_rms, vec_normalize_envs[i].obs_rms)

                # Sync normalization with workers
                for i in range(len(envs)):
                    self._sync_running_stats(self._vec_normalize_env.obs_rms, vec_normalize_envs[i].obs_rms)

        else:
            for weights_idx in range(self.pop_size):

                callback.on_rollout_start()
                train_policy.load_from_vector(candidate_weights[weights_idx])

                episode_rewards, episode_lengths = evaluate_policy(
                    train_policy,
                    self.env,
                    n_eval_episodes=self.n_eval_episodes,
                    return_episode_rewards=True,
                    callback=lambda _locals, _globals: callback.on_step(),
                    warn=False,
                )

                candidate_returns[weights_idx] = sum(episode_rewards) + self.alive_bonus_offset * sum(episode_lengths)
                batch_steps += sum(episode_lengths)

                # Mimic Monitor Wrapper
                infos = [
                    {"episode": {"r": episode_reward, "l": episode_length}}
                    for episode_reward, episode_length in zip(episode_rewards, episode_lengths)
                ]

                self._update_info_buffer(infos)

                callback.on_rollout_end()

        return candidate_returns, batch_steps

    # Make sure our hyper parameters are valid and auto correct them if they are not
    def _validate_hyper_params(self) -> None:
        if self.n_top > self.n_delta:
            warnings.warn(f"n_top = {self.n_top} > n_delta = {self.n_top}, setting n_top = n_delta")
            self.n_top = self.n_delta

    def _log_and_dump(self) -> None:
        fps = int(self.num_timesteps / (time.time() - self.start_time))
        self.logger.record("time/iterations", self._n_updates, exclude="tensorboard")
        if len(self.ep_info_buffer) > 0 and len(self.ep_info_buffer[0]) > 0:
            self.logger.record("rollout/ep_rew_mean", safe_mean([ep_info["r"] for ep_info in self.ep_info_buffer]))
            self.logger.record("rollout/ep_len_mean", safe_mean([ep_info["l"] for ep_info in self.ep_info_buffer]))
        self.logger.record("time/fps", fps)
        self.logger.record("time/time_elapsed", int(time.time() - self.start_time), exclude="tensorboard")
        self.logger.record("time/total_timesteps", self.num_timesteps, exclude="tensorboard")
        self.logger.dump(step=self.num_timesteps)

    def _do_one_update(self, callback: BaseCallback, envs: List[VecEnv]) -> None:
        delta_std = self.delta_std_schedule(self._current_progress_remaining)
        learning_rate = self.lr_schedule(self._current_progress_remaining)

        deltas = th.normal(mean=0.0, std=1.0, size=(self.n_delta, self.n_params))

        with th.no_grad():
            candidate_returns, batch_steps = self._collect_rollouts(deltas * delta_std, callback, envs)

        plus_returns = candidate_returns[: self.n_delta]
        minus_returns = candidate_returns[self.n_delta :]

        top_returns, _ = th.max(th.vstack((plus_returns, minus_returns)), dim=0)

        top_idx = th.argsort(top_returns, descending=True)[: self.n_top]
        plus_returns = plus_returns[top_idx]
        minus_returns = minus_returns[top_idx]
        deltas = deltas[top_idx]

        return_std = th.cat([plus_returns, minus_returns]).std()
        step_size = learning_rate / (self.n_top * return_std + 1e-6)
        self.weights = self.weights + step_size * ((plus_returns - minus_returns) @ deltas)
        self.policy.load_from_vector(self.weights)

        self._n_updates += 1
        self.num_timesteps += batch_steps

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
        envs: Optional[List[VecEnv]] = None,
    ) -> "ARS":

        total_steps = total_timesteps
        total_steps, callback = self._setup_learn(
            total_steps, eval_env, callback, eval_freq, n_eval_episodes, eval_log_path, reset_num_timesteps, tb_log_name
        )

        # Split envs so we can use parallel workers with different polcies
        self._validate_hyper_params()
        callback.on_training_start(locals(), globals())

        while self.num_timesteps < total_steps:
            self._update_current_progress_remaining(self.num_timesteps, total_timesteps)
            self._do_one_update(callback, envs or [self.env])
            if log_interval is not None and self._n_updates % log_interval == 0:
                self._log_and_dump()

        callback.on_training_end()

        return self
