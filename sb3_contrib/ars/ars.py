import copy
import io
import pathlib
import time
import warnings
from typing import Any, Dict, Optional, Type, Union

import gym
import torch as th
import torch.nn.utils
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.policies import BasePolicy
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, Schedule
from stable_baselines3.common.utils import get_schedule_fn, safe_mean
from stable_baselines3.common.evaluation import evaluate_policy

from sb3_contrib.ars.policies import ARSPolicy


class ARS(BaseAlgorithm):
    """
    Augmented Random Search: https://arxiv.org/abs/1803.07055

    :param policy: The policy to train, can be an instance of ARSPolicy, or a string
    :param env: The environment to train on, may be a string if registred with gym
    :param n_delta: How many random pertubations of the policy to try at each update step.
    :param n_top: How many of the top delta to use in each update step. Default is n_delta
    :param step_size: Float or schedule for the step size
    :param delta_std: Float or schedule for the exploration noise
    :param policy_kwargs: Keyword arguments to pass to the policy on creation
    :param policy_base: Base class to use for the policy
    :param tensorboard_log: String with the directory to put tensorboard logs:
    :param seed: Random seed for the training
    :param verbose: Verbosity level: 0 no output, 1 info, 2 debug
    :param device: Torch device to use for training, defaults to "cpu"
    :param _init_setup_model: Whether or not to build the network at the creation of the instance
    :param alive_bonus_offset: Constant added to the reward at each step, a value of -1 is used in the original paper
    """

    def __init__(
        self,
        policy: Union[str, Type[ARSPolicy]],
        env: Union[GymEnv, str],
        n_delta: int = 64,
        n_top: Optional[int] = None,
        step_size: Union[float, Schedule] = 0.05,
        delta_std: Union[float, Schedule] = 0.05,
        policy_kwargs: Optional[Dict[str, Any]] = None,
        policy_base: Type[BasePolicy] = ARSPolicy,
        tensorboard_log: Optional[str] = None,
        seed: Optional[int] = None,
        verbose: int = 0,
        device: Union[th.device, str] = "cpu",
        _init_setup_model: bool = True,
        alive_bonus_offset: float = 0,
        zero_policy: bool = True,
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
        )

        self.n_delta = n_delta
        self.step_size = get_schedule_fn(step_size)
        self.delta_std = get_schedule_fn(delta_std)

        if n_top is None:
            n_top = n_delta
        self.n_top = n_top

        self.n_workers = None  # We check this at training time, when the env is loaded

        if policy_kwargs is None:
            policy_kwargs = {}
        self.policy_kwargs = policy_kwargs

        self.seed = seed
        self.alive_bonus_offset = alive_bonus_offset

        self.zero_policy = zero_policy
        self.weight = None  # Need to call init model to initialize weight

        if (
            _init_setup_model
        ):  # TODO ... what do I do if this is false? am i guaranteed that someone will call this before training?
            self._setup_model()

    @classmethod
    def load(
        cls,
        path: Union[str, pathlib.Path, io.BufferedIOBase],
        env: Optional[GymEnv] = None,
        device: Union[th.device, str] = "cpu",
        custom_objects: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> "BaseAlgorithm":
        return super().load(path, env, device, custom_objects, **kwargs)

    def _setup_model(self) -> None:
        self.set_random_seed(self.seed)
        if self.seed is not None:
            th.manual_seed(self.seed)

        self.policy = self.policy_class(self.observation_space, self.action_space, **self.policy_kwargs)
        self.dtype = self.policy.parameters().__next__().dtype  # This seems sort of like a hack
        self.weight = th.nn.utils.parameters_to_vector(self.policy.parameters()).detach()

        if self.zero_policy:
            self.weight = th.zeros_like(self.weight, requires_grad=False, dtype=self.dtype)
            th.nn.utils.vector_to_parameters(self.weight, self.policy.parameters())

        self.n_params = len(self.weight)
        self.policy = self.policy.to(self.device)

    def _collect_rollouts(self, policy_deltas, callback):
        with th.no_grad():
            batch_steps = 0
            weight_idx = 0

            # Generate 2*n_delta candidate policies by adding noise to the current weight
            candidate_weights = th.cat([self.weight + policy_deltas, self.weight - policy_deltas])
            candidate_returns = th.zeros(candidate_weights.shape[0])  # returns == sum of rewards
            train_policy = copy.deepcopy(self.policy)
            self.ep_info_buffer = []

            for weight_idx in range(candidate_returns.shape[0]):

                callback.on_rollout_start()
                th.nn.utils.vector_to_parameters(candidate_weights[weight_idx], train_policy.parameters())

                def callback_hack(local, globals):
                    callback.on_step()

                episode_rewards, episode_lengths = evaluate_policy(
                    train_policy,
                    self.env,
                    n_eval_episodes=1,
                    return_episode_rewards=True,
                    callback=callback_hack,
                )

                candidate_returns[weight_idx] = sum(episode_rewards) + self.alive_bonus_offset * episode_lengths[0]
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
    def _validate_hyper_params(self):
        if self.n_top > self.n_delta:
            warnings.warn(f"n_top = {self.n_top} > n_delta = {self.n_top}, setting n_top = n_delta")
            self.n_top = self.n_delta

        if self.n_workers > 1:
            warnings.warn("n_workers > 1. For performance reasons we reccomend using a single environement.")

        # This makes sense if we switch from evaluate_policy to the old vectorized approach.
        # if self.n_delta % self.n_workers:
        #     new_n_delta = self.n_delta + (self.n_workers - self.n_delta % self.n_workers)
        #     warnings.warn(
        #         f"n_delta = {self.n_delta} should be a multiple of the number of workers = {self.n_workers}"
        #         f" automatically bumping n_delta to {new_n_delta}"
        #     )
        #    self.n_delta = new_n_delta

        return

    def _log_and_dump(self):
        fps = int(self.num_timesteps / (time.time() - self.start_time))
        self.logger.record("time/iterations", self._n_updates, exclude="tensorboard")
        if len(self.ep_info_buffer) > 0 and len(self.ep_info_buffer[0]) > 0:
            self.logger.record("rollout/ep_rew_mean", safe_mean([ep_info["r"] for ep_info in self.ep_info_buffer]))
            self.logger.record("rollout/ep_len_mean", safe_mean([ep_info["l"] for ep_info in self.ep_info_buffer]))
        self.logger.record("time/fps", fps)
        self.logger.record("time/time_elapsed", int(time.time() - self.start_time), exclude="tensorboard")
        self.logger.record("time/total_timesteps", self.num_timesteps, exclude="tensorboard")
        self.logger.dump(step=self.num_timesteps)

    def _do_one_update(self, callback):
        delta_std_this_step = self.delta_std(self._current_progress_remaining)
        step_size_this_step = self.step_size(self._current_progress_remaining)

        deltas = th.normal(mean=0.0, std=1.0, size=(self.n_delta, self.n_params))

        candidate_returns, batch_steps = self._collect_rollouts(deltas * delta_std_this_step, callback)

        plus_returns = candidate_returns[: self.n_delta]
        minus_returns = candidate_returns[self.n_delta :]

        top_returns = th.zeros_like(plus_returns)
        for i in range(len(top_returns)):
            top_returns[i] = max([plus_returns[i], minus_returns[i]])

        top_idx = th.argsort(top_returns, descending=True)[: self.n_top]
        plus_returns = plus_returns[top_idx]
        minus_returns = minus_returns[top_idx]
        deltas = deltas[top_idx]

        return_std = th.cat([plus_returns, minus_returns]).std()
        step_size = step_size_this_step / (self.n_top * return_std + 1e-6)
        self.weight = self.weight + step_size * ((plus_returns - minus_returns) @ deltas)
        th.nn.utils.vector_to_parameters(self.weight, self.policy.parameters())

        self._n_updates += 1
        self.num_timesteps += batch_steps

    def learn(
        self,
        total_timesteps: int,
        callback: MaybeCallback = None,
        log_interval: int = 100,
        tb_log_name: str = "ARS",
        eval_env: Optional[GymEnv] = None,
        eval_freq: int = -1,
        n_eval_episodes: int = 5,
        eval_log_path: Optional[str] = None,
        reset_num_timesteps: bool = True,
    ) -> "ARS":

        total_steps = total_timesteps
        total_steps, callback = self._setup_learn(
            total_steps, eval_env, callback, eval_freq, n_eval_episodes, eval_log_path, reset_num_timesteps, tb_log_name
        )

        self.n_workers = self.env.num_envs  # TODO can I do this somewhere earlier?? when is the env loaded ... ?
        self._validate_hyper_params()
        callback.on_training_start(locals(), globals())

        while self.num_timesteps < total_steps:
            self._update_current_progress_remaining(self.num_timesteps, total_timesteps)

            self._do_one_update(callback)

            if log_interval is not None and self._n_updates % log_interval == 0:
                self._log_and_dump()

        torch.nn.utils.vector_to_parameters(self.weight, self.policy.parameters())

        callback.on_training_end()

        return self
