from typing import Generator, NamedTuple, Optional, Tuple, Union

import numpy as np
import torch as th
from gym import spaces
from stable_baselines3.common.buffers import DictRolloutBuffer, RolloutBuffer
from stable_baselines3.common.type_aliases import TensorDict
from stable_baselines3.common.vec_env import VecNormalize


class RecurrentRolloutBufferSamples(NamedTuple):
    observations: th.Tensor
    actions: th.Tensor
    old_values: th.Tensor
    old_log_prob: th.Tensor
    advantages: th.Tensor
    returns: th.Tensor
    lstm_states: Tuple[th.Tensor, th.Tensor]
    episode_starts: th.Tensor


class RecurrentDictRolloutBufferSamples(RecurrentRolloutBufferSamples):
    observations: TensorDict
    actions: th.Tensor
    old_values: th.Tensor
    old_log_prob: th.Tensor
    advantages: th.Tensor
    returns: th.Tensor
    lstm_states: Tuple[th.Tensor, th.Tensor]
    episode_starts: th.Tensor


class RecurrentRolloutBuffer(RolloutBuffer):
    """
    Rollout buffer that also stores the invalid action masks associated with each observation.

    :param buffer_size: Max number of element in the buffer
    :param observation_space: Observation space
    :param action_space: Action space
    :param device:
    :param gae_lambda: Factor for trade-off of bias vs variance for Generalized Advantage Estimator
        Equivalent to classic advantage when set to 1.
    :param gamma: Discount factor
    :param n_envs: Number of parallel environments
    """

    def __init__(
        self,
        buffer_size: int,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        # lstm_states: Tuple[np.ndarray, np.ndarray],
        device: Union[th.device, str] = "cpu",
        gae_lambda: float = 1,
        gamma: float = 0.99,
        n_envs: int = 1,
    ):
        # self.lstm_states = lstm_states
        # self.dones = None
        self.initial_lstm_states = None
        super().__init__(buffer_size, observation_space, action_space, device, gae_lambda, gamma, n_envs)

    def reset(self):
        super().reset()
        # self.hidden_states = np.zeros_like(self.lstm_states[0])
        # self.cell_states = np.zeros_like(self.lstm_states[1])
        # self.dones = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)

    # def add(self, *args, lstm_states: Tuple[np.ndarray, np.ndarray], **kwargs) -> None:
    #     """
    #     :param hidden_states: LSTM cell and hidden state
    #     """
    #     self.hidden_states[self.pos] = np.array(lstm_states[0])
    #     self.cell_states[self.pos] = np.array(lstm_states[1])
    #     self.dones[self.pos] = np.array(dones)
    #
    #     super().add(*args, **kwargs)

    def get(self, batch_size: Optional[int] = None) -> Generator[RecurrentRolloutBufferSamples, None, None]:
        assert self.full, ""

        # Prepare the data
        if not self.generator_ready:
            # hidden_state_shape = (self.n_steps, lstm.num_layers, self.n_envs, lstm.hidden_size)
            # swap first to (self.n_steps, self.n_envs, lstm.num_layers, lstm.hidden_size)
            # self.hidden_states = self.hidden_states.swapaxes(1, 2)
            # self.cell_states = self.cell_states.swapaxes(1, 2)

            for tensor in [
                "observations",
                "actions",
                "values",
                "log_probs",
                "advantages",
                "returns",
                # "hidden_states",
                # "cell_states",
                "episode_starts",
            ]:
                self.__dict__[tensor] = self.swap_and_flatten(self.__dict__[tensor])
            self.generator_ready = True

        # Return everything, don't create minibatches
        if batch_size is None:
            batch_size = self.buffer_size * self.n_envs

        # indices = np.arange(self.buffer_size * self.n_envs)
        # start_idx = 0
        # while start_idx < self.buffer_size * self.n_envs:
        #     yield self._get_samples(indices[start_idx : start_idx + batch_size])
        #     start_idx += batch_size

        # Do not shuffle the sequence, only the env indices
        # n_minibatches = (self.buffer_size * self.n_envs) // batch_size
        n_minibatches = 1
        # assert (
        #     self.n_envs % n_minibatches == 0
        # ), f"{self.n_envs} not a multiple of {n_minibatches} = {self.buffer_size * self.n_envs} // {batch_size}"
        n_envs_per_batch = self.n_envs // n_minibatches
        # n_envs_per_batch = batch_size // self.buffer_size

        env_indices = np.random.permutation(self.n_envs)
        flat_indices = np.arange(self.buffer_size * self.n_envs).reshape(self.n_envs, self.buffer_size)

        for start_env_idx in range(0, self.n_envs, n_envs_per_batch):
            end_env_idx = start_env_idx + n_envs_per_batch
            mini_batch_env_indices = env_indices[start_env_idx:end_env_idx]
            batch_inds = flat_indices[mini_batch_env_indices].ravel()
            # lstm_states = (
            #     self.hidden_states[:, :, mini_batch_env_indices, :][0],
            #     self.cell_states[:, :, mini_batch_env_indices, :][0],
            # )
            lstm_states_pi = (
                self.initial_lstm_states[0][0][:, mini_batch_env_indices].clone(),
                self.initial_lstm_states[0][1][:, mini_batch_env_indices].clone(),
            )
            # lstm_states_vf = (
            #     self.initial_lstm_states[1][0][:, mini_batch_env_indices].clone(),
            #     self.initial_lstm_states[1][1][:, mini_batch_env_indices].clone(),
            # )
            lstm_states_vf = None

            yield RecurrentRolloutBufferSamples(
                observations=self.to_torch(self.observations[batch_inds]),
                actions=self.to_torch(self.actions[batch_inds]),
                old_values=self.to_torch(self.values[batch_inds].flatten()),
                old_log_prob=self.to_torch(self.log_probs[batch_inds].flatten()),
                advantages=self.to_torch(self.advantages[batch_inds].flatten()),
                returns=self.to_torch(self.returns[batch_inds].flatten()),
                # lstm_states=(self.to_torch(lstm_states[0]), self.to_torch(lstm_states[1])),
                lstm_states=(lstm_states_pi, lstm_states_vf),
                # dones=self.to_torch(self.dones[batch_inds].flatten()),
                episode_starts=self.to_torch(self.episode_starts[batch_inds].flatten()),
            )


class RecurrentDictRolloutBuffer(DictRolloutBuffer):
    """
    Dict Rollout buffer used in on-policy algorithms like A2C/PPO.
    Extends the RolloutBuffer to use dictionary observations

    It corresponds to ``buffer_size`` transitions collected
    using the current policy.
    This experience will be discarded after the policy update.
    In order to use PPO objective, we also store the current value of each state
    and the log probability of each taken action.

    The term rollout here refers to the model-free notion and should not
    be used with the concept of rollout used in model-based RL or planning.
    Hence, it is only involved in policy and value function training but not action selection.

    :param buffer_size: Max number of element in the buffer
    :param observation_space: Observation space
    :param action_space: Action space
    :param device:
    :param gae_lambda: Factor for trade-off of bias vs variance for Generalized Advantage Estimator
        Equivalent to classic advantage when set to 1.
    :param gamma: Discount factor
    :param n_envs: Number of parallel environments
    """

    def __init__(
        self,
        buffer_size: int,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        lstm_states: Tuple[np.ndarray, np.ndarray],
        device: Union[th.device, str] = "cpu",
        gae_lambda: float = 1,
        gamma: float = 0.99,
        n_envs: int = 1,
    ):
        super(RecurrentDictRolloutBuffer, self).__init__(
            buffer_size, observation_space, action_space, device, gae_lambda, gamma, n_envs=n_envs
        )
        self.lstm_states = lstm_states
        self.dones = None

    def reset(self):
        self.hidden_states = np.zeros_like(self.lstm_states[0])
        self.cell_states = np.zeros_like(self.lstm_states[1])
        self.dones = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        super().reset()

    def add(self, *args, lstm_states: Tuple[np.ndarray, np.ndarray], dones: np.ndarray, **kwargs) -> None:
        """
        :param hidden_states: LSTM cell and hidden state
        """
        self.hidden_states[self.pos] = np.array(lstm_states[0])
        self.cell_states[self.pos] = np.array(lstm_states[1])
        self.dones[self.pos] = np.array(dones)

        super().add(*args, **kwargs)

    def get(self, batch_size: Optional[int] = None) -> Generator[RecurrentDictRolloutBufferSamples, None, None]:
        assert self.full, ""
        # indices = np.random.permutation(self.buffer_size * self.n_envs)
        # Do not shuffle the data
        indices = np.arange(self.buffer_size * self.n_envs)
        # Prepare the data
        if not self.generator_ready:
            # hidden_state_shape = (self.n_steps, lstm.num_layers, self.n_envs, lstm.hidden_size)
            # swap first to (self.n_steps, self.n_envs, lstm.num_layers, lstm.hidden_size)
            self.hidden_states = self.hidden_states.swapaxes(1, 2)
            self.cell_states = self.cell_states.swapaxes(1, 2)

            for key, obs in self.observations.items():
                self.observations[key] = self.swap_and_flatten(obs)

            _tensor_names = [
                "actions",
                "values",
                "log_probs",
                "advantages",
                "returns",
                "hidden_states",
                "cell_states",
                "dones",
            ]

            for tensor in _tensor_names:
                self.__dict__[tensor] = self.swap_and_flatten(self.__dict__[tensor])
            self.generator_ready = True

        # Return everything, don't create minibatches
        if batch_size is None:
            batch_size = self.buffer_size * self.n_envs

        start_idx = 0
        while start_idx < self.buffer_size * self.n_envs:
            yield self._get_samples(indices[start_idx : start_idx + batch_size])
            start_idx += batch_size

    def _get_samples(self, batch_inds: np.ndarray, env: Optional[VecNormalize] = None) -> RecurrentDictRolloutBufferSamples:

        return RecurrentDictRolloutBufferSamples(
            observations={key: self.to_torch(obs[batch_inds]) for (key, obs) in self.observations.items()},
            actions=self.to_torch(self.actions[batch_inds]),
            old_values=self.to_torch(self.values[batch_inds].flatten()),
            old_log_prob=self.to_torch(self.log_probs[batch_inds].flatten()),
            advantages=self.to_torch(self.advantages[batch_inds].flatten()),
            returns=self.to_torch(self.returns[batch_inds].flatten()),
            lstm_states=(self.to_torch(self.hidden_states[batch_inds]), self.to_torch(self.cell_states[batch_inds])),
            dones=self.to_torch(self.dones[batch_inds]),
        )
