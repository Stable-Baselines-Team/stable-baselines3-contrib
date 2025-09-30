import copy
import sys
import time
from collections import deque
from functools import partial
from typing import Any, ClassVar, Optional, TypeVar, Union

import numpy as np
import torch as th
from gymnasium import spaces
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.policies import BasePolicy
from stable_baselines3.common.running_mean_std import RunningMeanStd
from stable_baselines3.common.save_util import load_from_zip_file
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, Schedule, TensorDict
from stable_baselines3.common.utils import safe_mean

from sb3_contrib.common.policies import ESLinearPolicy, ESPolicy
from sb3_contrib.common.vec_env.async_eval import AsyncEval

SelfPopulationBased = TypeVar("SelfPopulationBased", bound="PopulationBasedAlgorithm")


class PopulationBasedAlgorithm(BaseAlgorithm):
    """
    Base class for population based algorithms like
    Evolution Strategies (ES), which does black-box optimization
    by sampling and evaluating a population of candidates:
    https://blog.otoro.net/2017/10/29/visual-evolution-strategies/

    :param policy: The policy to train, can be an instance of ``ESPolicy``, or a string from ["LinearPolicy", "MlpPolicy"]
    :param env: The environment to train on, may be a string if registered with gym
    :param pop_size: Population size (number of individuals)
    :param alive_bonus_offset: Constant added to the reward at each step, used to cancel out alive bonuses.
    :param n_eval_episodes: Number of episodes to evaluate each candidate.
    :param policy_kwargs: Keyword arguments to pass to the policy on creation
    :param stats_window_size: Window size for the rollout logging, specifying the number of episodes to average
        the reported success rate, mean episode length, and mean reward over
    :param tensorboard_log: String with the directory to put tensorboard logs:
    :param seed: Random seed for the training
    :param verbose: Verbosity level: 0 no output, 1 info, 2 debug
    :param device: Torch device to use for training, defaults to "cpu"
    :param _init_setup_model: Whether or not to build the network at the creation of the instance
    """

    policy_aliases: ClassVar[dict[str, type[BasePolicy]]] = {
        "MlpPolicy": ESPolicy,
        "LinearPolicy": ESLinearPolicy,
    }

    def __init__(
        self,
        policy: Union[str, type[ESPolicy]],
        env: Union[GymEnv, str],
        learning_rate: Union[float, Schedule],
        pop_size: int = 16,
        alive_bonus_offset: float = 0,
        n_eval_episodes: int = 1,
        policy_kwargs: Optional[dict[str, Any]] = None,
        stats_window_size: int = 100,
        tensorboard_log: Optional[str] = None,
        seed: Optional[int] = None,
        verbose: int = 0,
        device: Union[th.device, str] = "cpu",
        supported_action_spaces: Optional[tuple[type[spaces.Space], ...]] = None,
    ):

        super().__init__(
            policy,
            env,
            learning_rate=learning_rate,
            stats_window_size=stats_window_size,
            tensorboard_log=tensorboard_log,
            policy_kwargs=policy_kwargs,
            verbose=verbose,
            device=device,
            supported_action_spaces=supported_action_spaces,
            support_multi_env=True,
            seed=seed,
        )

        self.pop_size = pop_size
        self.n_eval_episodes = n_eval_episodes
        self.alive_bonus_offset = alive_bonus_offset

        # Keep track of how many steps where elapsed before a new rollout
        # Important for syncing observation normalization between workers
        self.old_count = 0

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
        _locals: dict[str, Any],
        _globals: dict[str, Any],
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
        self.ep_info_buffer = deque(maxlen=self._stats_window_size)
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
                    assert isinstance(
                        self._vec_normalize_env.obs_rms, RunningMeanStd
                    ), "ES Algorithms don't support dict obs with normalization yet"
                    # worker_obs_rms.count -= self.old_count
                    self._vec_normalize_env.obs_rms.combine(worker_obs_rms)
                    # Hack: don't count timesteps twice (between the two are synced)
                    # otherwise it will lead to overflow,
                    # in practice we would need two RunningMeanStats
                    self._vec_normalize_env.obs_rms.count -= self.old_count

            # Synchronise VecNormalize if needed
            if self._vec_normalize_env is not None:
                assert isinstance(
                    self._vec_normalize_env.obs_rms, RunningMeanStd
                ), "ES Algorithms don't support dict obs with normalization yet"
                async_eval.sync_obs_rms(self._vec_normalize_env.obs_rms.copy())
                self.old_count = self._vec_normalize_env.obs_rms.count

            # Hack to have Callback events
            for _ in range(batch_steps // len(async_eval.remotes)):
                self.num_timesteps += len(async_eval.remotes)
                callback.on_step()
        else:
            assert self.env is not None
            # Single process, synchronous version
            for weights_idx in range(self.pop_size):

                # Load current candidate weights
                train_policy.load_from_vector(candidate_weights[weights_idx].cpu().numpy())
                # Evaluate the candidate
                episode_rewards, episode_lengths = evaluate_policy(  # type: ignore[assignment]
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

    def dump_logs(self) -> None:
        """
        Dump information to the logger.
        """
        assert self.ep_info_buffer is not None
        assert self.ep_success_buffer is not None

        time_elapsed = max((time.time_ns() - self.start_time) / 1e9, sys.float_info.epsilon)
        fps = int((self.num_timesteps - self._num_timesteps_at_start) / (time_elapsed + 1e-8))
        if len(self.ep_info_buffer) > 0 and len(self.ep_info_buffer[0]) > 0:
            self.logger.record("rollout/ep_rew_mean", safe_mean([ep_info["r"] for ep_info in self.ep_info_buffer]))
            self.logger.record("rollout/ep_len_mean", safe_mean([ep_info["l"] for ep_info in self.ep_info_buffer]))
        self.logger.record("time/fps", fps)
        self.logger.record("time/time_elapsed", int(time_elapsed), exclude="tensorboard")
        self.logger.record("time/total_timesteps", self.num_timesteps, exclude="tensorboard")
        if len(self.ep_success_buffer) > 0:
            self.logger.record("rollout/success_rate", safe_mean(self.ep_success_buffer))
        self.logger.dump(step=self.num_timesteps)

    def _do_one_update(self, callback: BaseCallback, async_eval: Optional[AsyncEval]) -> None:
        """
        Sample new candidates, evaluate them and then update current policy.

        :param callback: callback(s) called at every step with state of the algorithm.
        :param async_eval: The object for asynchronous evaluation of candidates.
        """
        raise NotImplementedError()

    def learn(  # type: ignore[override]
        self: SelfPopulationBased,
        total_timesteps: int,
        callback: MaybeCallback = None,
        log_interval: int = 1,
        tb_log_name: str = "ES",
        reset_num_timesteps: bool = True,
        async_eval: Optional[AsyncEval] = None,
        progress_bar: bool = False,
    ) -> SelfPopulationBased:
        """
        Return a trained model.

        :param total_timesteps: The total number of samples (env steps) to train on
        :param callback: callback(s) called at every step with state of the algorithm.
        :param log_interval: The number of timesteps before logging.
        :param tb_log_name: the name of the run for TensorBoard logging
        :param reset_num_timesteps: whether or not to reset the current timestep number (used in logging)
        :param async_eval: The object for asynchronous evaluation of candidates.
        :return: the trained model
        """

        total_steps, callback = self._setup_learn(
            total_timesteps,
            callback,
            reset_num_timesteps,
            tb_log_name,
            progress_bar,
        )

        callback.on_training_start(locals(), globals())

        while self.num_timesteps < total_steps:
            self._update_current_progress_remaining(self.num_timesteps, total_timesteps)
            self._do_one_update(callback, async_eval)
            if log_interval is not None and self._n_updates % log_interval == 0:
                self.dump_logs()

        if async_eval is not None:
            async_eval.close()

        callback.on_training_end()

        return self

    def set_parameters(
        self,
        load_path_or_dict: Union[str, TensorDict],
        exact_match: bool = True,
        device: Union[th.device, str] = "auto",
    ) -> None:
        # Patched set_parameters() to handle ARS linear policy saved with sb3-contrib < 1.7.0
        params = None
        if isinstance(load_path_or_dict, dict):
            params = load_path_or_dict
        else:
            _, params, _ = load_from_zip_file(load_path_or_dict, device=device)

        # Patch to load LinearPolicy saved using sb3-contrib < 1.7.0
        # See https://github.com/Stable-Baselines-Team/stable-baselines3-contrib/pull/122#issuecomment-1331981230
        for name in {"weight", "bias"}:
            if f"action_net.{name}" in params.get("policy", {}):
                params["policy"][f"action_net.0.{name}"] = params["policy"][f"action_net.{name}"]  # type: ignore[index]
                del params["policy"][f"action_net.{name}"]  # type: ignore[attr-defined]

        super().set_parameters(params, exact_match=exact_match)
