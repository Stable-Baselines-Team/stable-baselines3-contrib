from typing import Any, Dict, List, Optional, Union

import gym
import numpy as np
import torch as th
from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.preprocessing import get_obs_shape
from stable_baselines3.common.type_aliases import ReplayBufferSamples
from stable_baselines3.common.vec_env import VecEnv, VecNormalize
from stable_baselines3.her.her_replay_buffer import get_time_limit


class QLambdaReplayBuffer(ReplayBuffer):
    """
    Peng's Q(lambda) replay buffer
    Paper: https://arxiv.org/abs/2103.00107

    .. warning::

      For performance reasons, the maximum number of steps per episodes must be specified.
      In most cases, it will be inferred if you specify ``max_episode_steps`` when registering the environment
      or if you use a ``gym.wrappers.TimeLimit`` (and ``env.spec`` is not None).
      Otherwise, you can directly pass ``max_episode_length`` to the replay buffer constructor.

    :param buffer_size: The size of the buffer measured in transitions.
    :param max_episode_length: The maximum length of an episode. If not specified,
        it will be automatically inferred if the environment uses a ``gym.wrappers.TimeLimit`` wrapper.
    :param device: PyTorch device
    :param handle_timeout_termination: Handle timeout termination (due to timelimit)
        separately and treat the task as infinite horizon task.
        https://github.com/DLR-RM/stable-baselines3/issues/284
    """

    def __init__(
        self,
        buffer_size: int,
        observation_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        device: Union[th.device, str] = "cpu",
        n_envs: int = 1,
        optimize_memory_usage: bool = False,
        gamma: float = 0.99,
        peng_lambda: float = 0.7,
        n_steps: int = 5,
        max_episode_length: Optional[int] = None,
        handle_timeout_termination: bool = True,
    ):

        super(QLambdaReplayBuffer, self).__init__(buffer_size, observation_space, action_space, device, n_envs)
        self.peng_lambda = peng_lambda
        self.n_steps = n_steps
        self.gamma = gamma

        # Handle timeouts termination properly if needed
        # see https://github.com/DLR-RM/stable-baselines3/issues/284
        self.handle_timeout_termination = handle_timeout_termination
        self.max_episode_length = max_episode_length

    def setup_buffer(self, env: VecEnv, actor: th.nn.Module, critic_target: th.nn.Module, gamma: float):
        self.gamma = gamma
        # maximum steps in episode
        self.max_episode_length = get_time_limit(env, self.max_episode_length)

        # buffer with episodes
        # number of episodes which can be stored until buffer size is reached
        self.max_episode_stored = self.buffer_size // self.max_episode_length
        self.current_idx = 0
        # Counter to prevent overflow
        self.episode_steps = 0

        # Get shape of observation and goal (usually the same)
        self.obs_shape = get_obs_shape(self.observation_space)

        # input dimensions for buffer initialization
        input_shape = {
            "observation": (self.n_envs,) + self.obs_shape,
            "action": (self.action_dim,),
            "reward": (1,),
            "next_obs": (self.n_envs,) + self.obs_shape,
            "done": (1,),
            # "timeout": (1,),
        }
        self._buffer = {
            key: np.zeros((self.max_episode_stored, self.max_episode_length, *dim), dtype=np.float32)
            for key, dim in input_shape.items()
        }
        # episode length storage, needed for episodes which has less steps than the maximum length
        self.episode_lengths = np.zeros(self.max_episode_stored, dtype=np.int64)

        self.actor = actor
        self.critic_target = critic_target

    def _get_samples(self, batch_inds: np.ndarray, env: Optional[VecNormalize] = None) -> ReplayBufferSamples:
        """
        Abstract method from base class.
        """
        raise NotImplementedError()

    def sample(
        self,
        batch_size: int,
        env: Optional[VecNormalize],
    ) -> ReplayBufferSamples:
        """
        Sample function that replaces the "regular" replay buffer ``sample()``
        method in the ``train()`` function.

        :param batch_size: Number of element to sample
        :param env: Associated gym VecEnv
            to normalize the observations/rewards when sampling
        :return: Samples.
        """
        maybe_vec_env = env
        assert maybe_vec_env is None, "Normalization not supported yet"
        # Do not sample the episode with index `self.pos` as the episode is invalid
        if self.full:
            episode_indices = (np.random.randint(1, self.n_episodes_stored, batch_size) + self.pos) % self.n_episodes_stored
        else:
            episode_indices = np.random.randint(0, self.n_episodes_stored, batch_size)

        ep_lengths = self.episode_lengths[episode_indices]
        # transitions_indices = np.random.randint(self.episode_lengths[her_episode_indices])
        # Select which transitions to use
        transitions_indices = np.random.randint(ep_lengths)

        # get selected transitions
        transitions = {key: self._buffer[key][episode_indices, transitions_indices].copy() for key in self._buffer.keys()}

        # TODO: SAC add entropy term
        peng_targets = np.zeros((batch_size, self.n_steps), dtype=np.float32)

        with th.no_grad():
            # TODO: add ent coeff
            # TODO: better handle last step
            # n_step=1 does not do the proper thing
            valid_indices = np.where(transitions_indices + self.n_steps <= ep_lengths)
            next_obs = self._buffer["next_obs"][
                episode_indices[valid_indices], transitions_indices[valid_indices] + self.n_steps - 1
            ]
            next_obs = self.to_torch(next_obs)
            next_q_values = th.cat(self.critic_target(next_obs, self.actor(next_obs)), dim=1)
            next_q_values, _ = th.min(next_q_values, dim=1, keepdim=True)
            peng_targets[valid_indices, -1] = next_q_values.cpu().numpy().flatten()

            for t in reversed(range(self.n_steps - 1)):
                valid_indices = np.where(transitions_indices + t < ep_lengths)

                rewards = self._buffer["reward"][episode_indices[valid_indices], transitions_indices[valid_indices] + t]
                next_obs = self._buffer["next_obs"][episode_indices[valid_indices], transitions_indices[valid_indices] + t]
                dones = self._buffer["done"][episode_indices[valid_indices], transitions_indices[valid_indices] + t]
                next_obs = self.to_torch(next_obs)

                next_q_values = th.cat(self.critic_target(next_obs, self.actor(next_obs)), dim=1)
                next_q_values, _ = th.min(next_q_values, dim=1, keepdim=True)
                next_q_values = next_q_values.cpu().numpy().flatten()

                peng_targets[valid_indices, t] = (
                    rewards.flatten()
                    + self.gamma * next_q_values * (1 - dones).flatten()
                    + self.gamma
                    * self.peng_lambda
                    * (1 - dones).flatten()
                    * (peng_targets[valid_indices, t + 1] - next_q_values)
                )

            # last_td_lam = 0.0
            # for t in reversed(range(self.n_steps)):
            #     valid_indices = np.where(transitions_indices + t <= ep_lengths)
            #     if t == self.n_steps - 1:
            #         next_non_terminal = (1.0 - dones).flatten()
            #         # peng_targets[valid_indices, -1] = next_q_values.cpu().numpy().flatten()
            #     else:
            #         dones = self._buffer["done"][episode_indices[valid_indices], transitions_indices[valid_indices] + t]
            #         episode_starts = dones.flatten()
            #         next_non_terminal = 1.0 - episode_starts[step + 1]
            #         rewards = self._buffer["reward"][episode_indices[valid_indices], transitions_indices[valid_indices] + t]
            #
            #     next_obs = self._buffer["next_obs"][episode_indices[valid_indices], transitions_indices[valid_indices] + t]
            #     next_obs = self.to_torch(next_obs)
            #     next_q_values = th.cat(self.critic_target(next_obs, self.actor(next_obs)), dim=1)
            #     next_q_values, _ = th.min(next_q_values, dim=1, keepdim=True)
            #     next_q_values = next_q_values.cpu().numpy().flatten()
            #
            #     peng_targets[valid_indices, t] = (
            #         rewards.flatten()
            #         + self.gamma * next_q_values * (1 - dones).flatten()
            #         + self.gamma
            #         * self.peng_lambda
            #         * (1 - dones).flatten()
            #         * (peng_targets[valid_indices, t + 1] - next_q_values)
            #     )

        return ReplayBufferSamples(
            observations=self.to_torch(transitions["observation"]),
            actions=self.to_torch(transitions["action"]),
            next_observations=self.to_torch(transitions["next_obs"]),
            dones=self.to_torch(transitions["done"]),
            rewards=self.to_torch(peng_targets[:, 0].reshape(-1, 1)),
        )

    def add(
        self,
        obs: Dict[str, np.ndarray],
        next_obs: Dict[str, np.ndarray],
        action: np.ndarray,
        reward: np.ndarray,
        done: np.ndarray,
        infos: List[Dict[str, Any]],
    ) -> None:

        # Remove termination signals due to timeout
        if self.handle_timeout_termination:
            done_ = done * (1 - np.array([info.get("TimeLimit.truncated", False) for info in infos]))
        else:
            done_ = done

        self._buffer["observation"][self.pos][self.current_idx] = obs
        self._buffer["action"][self.pos][self.current_idx] = action
        self._buffer["done"][self.pos][self.current_idx] = done_
        self._buffer["reward"][self.pos][self.current_idx] = reward
        self._buffer["next_obs"][self.pos][self.current_idx] = next_obs

        # update current pointer
        self.current_idx += 1
        self.episode_steps += 1

        if done or self.episode_steps >= self.max_episode_length:
            self.store_episode()
            self.episode_steps = 0

    def store_episode(self) -> None:
        """
        Increment episode counter
        and reset transition pointer.
        """
        # add episode length to length storage
        self.episode_lengths[self.pos] = self.current_idx

        self.pos += 1
        if self.pos == self.max_episode_stored:
            self.full = True
            self.pos = 0
        # reset transition pointer
        self.current_idx = 0

    @property
    def n_episodes_stored(self) -> int:
        if self.full:
            return self.max_episode_stored
        return self.pos

    def size(self) -> int:
        """
        :return: The current number of transitions in the buffer.
        """
        return int(np.sum(self.episode_lengths))

    def reset(self) -> None:
        """
        Reset the buffer.
        """
        self.pos = 0
        self.current_idx = 0
        self.full = False
        self.episode_lengths = np.zeros(self.max_episode_stored, dtype=np.int64)

    #
    # def truncate_last_trajectory(self) -> None:
    #     """
    #     Only for online sampling, called when loading the replay buffer.
    #     If called, we assume that the last trajectory in the replay buffer was finished
    #     (and truncate it).
    #     If not called, we assume that we continue the same trajectory (same episode).
    #     """
    #     # If we are at the start of an episode, no need to truncate
    #     current_idx = self.current_idx
    #
    #     # truncate interrupted episode
    #     if current_idx > 0:
    #         warnings.warn(
    #             "The last trajectory in the replay buffer will be truncated.\n"
    #             "If you are in the same episode as when the replay buffer was saved,\n"
    #             "you should use `truncate_last_trajectory=False` to avoid that issue."
    #         )
    #         # get current episode and transition index
    #         pos = self.pos
    #         # set episode length for current episode
    #         self.episode_lengths[pos] = current_idx
    #         # set done = True for current episode
    #         # current_idx was already incremented
    #         self._buffer["done"][pos][current_idx - 1] = np.array([True], dtype=np.float32)
    #         # reset current transition index
    #         self.current_idx = 0
    #         # increment episode counter
    #         self.pos = (self.pos + 1) % self.max_episode_stored
    #         # update "full" indicator
    #         self.full = self.full or self.pos == 0
