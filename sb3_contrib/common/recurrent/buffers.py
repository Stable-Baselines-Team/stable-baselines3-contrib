from typing import Generator, Optional, Tuple, Union

import numpy as np
import torch as th
from gym import spaces
from stable_baselines3.common.buffers import DictRolloutBuffer, RolloutBuffer
from stable_baselines3.common.vec_env import VecNormalize

from sb3_contrib.common.recurrent.type_aliases import (
    RecurrentDictRolloutBufferSamples,
    RecurrentRolloutBufferSamples,
    RNNStates,
)


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
        lstm_states: Tuple[np.ndarray, np.ndarray],
        device: Union[th.device, str] = "cpu",
        gae_lambda: float = 1,
        gamma: float = 0.99,
        n_envs: int = 1,
        sampling_strategy: str = "default",  # "default" or "per_env"
    ):
        self.lstm_states = lstm_states
        self.initial_lstm_states = None
        self.sampling_strategy = sampling_strategy
        self.starts, self.ends = None, None
        super().__init__(buffer_size, observation_space, action_space, device, gae_lambda, gamma, n_envs)

    def reset(self):
        super().reset()
        self.hidden_states_pi = np.zeros_like(self.lstm_states[0])
        self.cell_states_pi = np.zeros_like(self.lstm_states[1])
        self.hidden_states_vf = np.zeros_like(self.lstm_states[0])
        self.cell_states_vf = np.zeros_like(self.lstm_states[1])

    def add(self, *args, lstm_states: RNNStates, **kwargs) -> None:
        """
        :param hidden_states: LSTM cell and hidden state
        """
        self.hidden_states_pi[self.pos] = np.array(lstm_states.pi[0].cpu().numpy())
        self.cell_states_pi[self.pos] = np.array(lstm_states.pi[1].cpu().numpy())
        self.hidden_states_vf[self.pos] = np.array(lstm_states.vf[0].cpu().numpy())
        self.cell_states_vf[self.pos] = np.array(lstm_states.vf[1].cpu().numpy())

        super().add(*args, **kwargs)

    def get(self, batch_size: Optional[int] = None) -> Generator[RecurrentRolloutBufferSamples, None, None]:
        assert self.full, ""

        # Prepare the data
        if not self.generator_ready:
            # hidden_state_shape = (self.n_steps, lstm.num_layers, self.n_envs, lstm.hidden_size)
            # swap first to (self.n_steps, self.n_envs, lstm.num_layers, lstm.hidden_size)
            for tensor in ["hidden_states_pi", "cell_states_pi", "hidden_states_vf", "cell_states_vf"]:
                self.__dict__[tensor] = self.__dict__[tensor].swapaxes(1, 2)

            for tensor in [
                "observations",
                "actions",
                "values",
                "log_probs",
                "advantages",
                "returns",
                "hidden_states_pi",
                "cell_states_pi",
                "hidden_states_vf",
                "cell_states_vf",
                "episode_starts",
            ]:
                self.__dict__[tensor] = self.swap_and_flatten(self.__dict__[tensor])
            self.generator_ready = True

        # Return everything, don't create minibatches
        if batch_size is None:
            batch_size = self.buffer_size * self.n_envs

        # Sampling strategy that allows any mini batch size but requires
        # more complexity and use of padding
        if self.sampling_strategy == "default":
            # No shuffling
            # indices = np.arange(self.buffer_size * self.n_envs)
            # Trick to shuffle a bit: keep the sequence order
            # but split the indices in two
            split_index = np.random.randint(self.buffer_size * self.n_envs)
            indices = np.arange(self.buffer_size * self.n_envs)
            indices = np.concatenate((indices[split_index:], indices[:split_index]))

            env_change = np.zeros(self.buffer_size * self.n_envs).reshape(self.buffer_size, self.n_envs)
            # Flag first timestep as change of environment
            env_change[0, :] = 1.0
            env_change = self.swap_and_flatten(env_change)

            start_idx = 0
            while start_idx < self.buffer_size * self.n_envs:
                batch_inds = indices[start_idx : start_idx + batch_size]
                yield self._get_samples(batch_inds, env_change)
                start_idx += batch_size
            return

        # ==== OpenAI Baselines way of sampling, constraint in the batch size and number of environments ====
        n_minibatches = (self.buffer_size * self.n_envs) // batch_size

        assert (
            self.n_envs % n_minibatches == 0
        ), f"{self.n_envs} not a multiple of {n_minibatches} = {self.buffer_size * self.n_envs} // {batch_size}"
        n_envs_per_batch = self.n_envs // n_minibatches

        # Do not shuffle the sequence, only the env indices
        env_indices = np.random.permutation(self.n_envs)
        flat_indices = np.arange(self.buffer_size * self.n_envs).reshape(self.n_envs, self.buffer_size)

        for start_env_idx in range(0, self.n_envs, n_envs_per_batch):
            end_env_idx = start_env_idx + n_envs_per_batch
            mini_batch_env_indices = env_indices[start_env_idx:end_env_idx]
            batch_inds = flat_indices[mini_batch_env_indices].ravel()
            lstm_states_pi = (
                self.initial_lstm_states.pi[0][:, mini_batch_env_indices].clone(),
                self.initial_lstm_states.pi[1][:, mini_batch_env_indices].clone(),
            )
            lstm_states_vf = (
                self.initial_lstm_states.vf[0][:, mini_batch_env_indices].clone(),
                self.initial_lstm_states.vf[1][:, mini_batch_env_indices].clone(),
            )

            yield RecurrentRolloutBufferSamples(
                observations=self.to_torch(self.observations[batch_inds]),
                actions=self.to_torch(self.actions[batch_inds]),
                old_values=self.to_torch(self.values[batch_inds].flatten()),
                old_log_prob=self.to_torch(self.log_probs[batch_inds].flatten()),
                advantages=self.to_torch(self.advantages[batch_inds].flatten()),
                returns=self.to_torch(self.returns[batch_inds].flatten()),
                lstm_states=RNNStates(lstm_states_pi, lstm_states_vf),
                episode_starts=self.to_torch(self.episode_starts[batch_inds].flatten()),
            )

    def pad(self, tensor: np.ndarray) -> th.Tensor:
        seq = [self.to_torch(tensor[start : end + 1]) for start, end in zip(self.starts, self.ends)]
        return th.nn.utils.rnn.pad_sequence(seq)

    def _get_samples(
        self,
        batch_inds: np.ndarray,
        env_change: np.ndarray,
        env: Optional[VecNormalize] = None,
    ) -> RecurrentRolloutBufferSamples:
        # Create sequence if env change too
        seq_start = np.logical_or(self.episode_starts[batch_inds], env_change[batch_inds]).flatten()
        # First index is always the beginning of a sequence
        seq_start[0] = True
        self.starts = np.where(seq_start == True)[0]  # noqa: E712
        self.ends = np.concatenate([(self.starts - 1)[1:], np.array([len(batch_inds)])])

        n_layers = self.hidden_states_pi.shape[1]
        n_seq = len(self.starts)
        max_length = self.pad(self.actions[batch_inds]).shape[0]
        # TODO: output mask to not backpropagate everywhere
        padded_batch_size = n_seq * max_length
        lstm_states_pi = (
            # (n_steps, n_layers, n_envs, dim) -> (n_layers, n_seq, dim)
            self.hidden_states_pi[batch_inds][seq_start == True].reshape(n_layers, n_seq, -1),  # noqa: E712
            self.cell_states_pi[batch_inds][seq_start == True].reshape(n_layers, n_seq, -1),  # noqa: E712
        )
        lstm_states_vf = (
            # (n_steps, n_layers, n_envs, dim) -> (n_layers, n_seq, dim)
            self.hidden_states_vf[batch_inds][seq_start == True].reshape(n_layers, n_seq, -1),  # noqa: E712
            self.cell_states_vf[batch_inds][seq_start == True].reshape(n_layers, n_seq, -1),  # noqa: E712
        )
        lstm_states_pi = (self.to_torch(lstm_states_pi[0]), self.to_torch(lstm_states_pi[1]))
        lstm_states_vf = (self.to_torch(lstm_states_vf[0]), self.to_torch(lstm_states_vf[1]))

        return RecurrentRolloutBufferSamples(
            observations=self.pad(self.observations[batch_inds]).swapaxes(0, 1).reshape((padded_batch_size,) + self.obs_shape),
            actions=self.pad(self.actions[batch_inds]).swapaxes(0, 1).reshape((padded_batch_size,) + self.actions.shape[1:]),
            old_values=self.pad(self.values[batch_inds]).swapaxes(0, 1).flatten(),
            old_log_prob=self.pad(self.log_probs[batch_inds]).swapaxes(0, 1).flatten(),
            advantages=self.pad(self.advantages[batch_inds]).swapaxes(0, 1).flatten(),
            returns=self.pad(self.returns[batch_inds]).swapaxes(0, 1).flatten(),
            lstm_states=RNNStates(lstm_states_pi, lstm_states_vf),
            episode_starts=self.pad(self.episode_starts[batch_inds]).swapaxes(0, 1).flatten(),
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
        sampling_strategy: str = "default",  # "default" or "per_env"
    ):
        self.lstm_states = lstm_states
        self.initial_lstm_states = None
        self.sampling_strategy = sampling_strategy
        assert sampling_strategy == "default", "'per_env' strategy not supported with dict obs"
        super().__init__(buffer_size, observation_space, action_space, device, gae_lambda, gamma, n_envs=n_envs)

    def reset(self):
        super().reset()
        self.hidden_states_pi = np.zeros_like(self.lstm_states[0])
        self.cell_states_pi = np.zeros_like(self.lstm_states[1])
        self.hidden_states_vf = np.zeros_like(self.lstm_states[0])
        self.cell_states_vf = np.zeros_like(self.lstm_states[1])

    def add(self, *args, lstm_states: RNNStates, **kwargs) -> None:
        """
        :param hidden_states: LSTM cell and hidden state
        """
        self.hidden_states_pi[self.pos] = np.array(lstm_states.pi[0].cpu().numpy())
        self.cell_states_pi[self.pos] = np.array(lstm_states.pi[1].cpu().numpy())
        self.hidden_states_vf[self.pos] = np.array(lstm_states.vf[0].cpu().numpy())
        self.cell_states_vf[self.pos] = np.array(lstm_states.vf[1].cpu().numpy())

        super().add(*args, **kwargs)

    def get(self, batch_size: Optional[int] = None) -> Generator[RecurrentDictRolloutBufferSamples, None, None]:
        assert self.full, ""

        # Prepare the data
        if not self.generator_ready:
            # hidden_state_shape = (self.n_steps, lstm.num_layers, self.n_envs, lstm.hidden_size)
            # swap first to (self.n_steps, self.n_envs, lstm.num_layers, lstm.hidden_size)
            for tensor in ["hidden_states_pi", "cell_states_pi", "hidden_states_vf", "cell_states_vf"]:
                self.__dict__[tensor] = self.__dict__[tensor].swapaxes(1, 2)

            for key, obs in self.observations.items():
                self.observations[key] = self.swap_and_flatten(obs)

            for tensor in [
                "actions",
                "values",
                "log_probs",
                "advantages",
                "returns",
                "hidden_states_pi",
                "cell_states_pi",
                "hidden_states_vf",
                "cell_states_vf",
                "episode_starts",
            ]:
                self.__dict__[tensor] = self.swap_and_flatten(self.__dict__[tensor])
            self.generator_ready = True

        # Return everything, don't create minibatches
        if batch_size is None:
            batch_size = self.buffer_size * self.n_envs

        # No shuffling:
        # indices = np.arange(self.buffer_size * self.n_envs)
        # Trick to shuffle a bit: keep the sequence order
        # but split the indices in two
        split_index = np.random.randint(self.buffer_size * self.n_envs)
        indices = np.arange(self.buffer_size * self.n_envs)
        indices = np.concatenate((indices[split_index:], indices[:split_index]))

        env_change = np.zeros(self.buffer_size * self.n_envs).reshape(self.buffer_size, self.n_envs)
        # Flag first timestep as change of environment
        env_change[0, :] = 1.0
        env_change = self.swap_and_flatten(env_change)

        start_idx = 0
        while start_idx < self.buffer_size * self.n_envs:
            batch_inds = indices[start_idx : start_idx + batch_size]
            yield self._get_samples(batch_inds, env_change)
            start_idx += batch_size

    def pad(self, tensor: np.ndarray) -> th.Tensor:
        seq = [self.to_torch(tensor[start : end + 1]) for start, end in zip(self.starts, self.ends)]
        return th.nn.utils.rnn.pad_sequence(seq)

    def _get_samples(
        self,
        batch_inds: np.ndarray,
        env_change: np.ndarray,
        env: Optional[VecNormalize] = None,
    ) -> RecurrentDictRolloutBufferSamples:
        # Create sequence if env change too
        seq_start = np.logical_or(self.episode_starts[batch_inds], env_change[batch_inds]).flatten()
        # First index is always the beginning of a sequence
        seq_start[0] = True
        self.starts = np.where(seq_start == True)[0]  # noqa: E712
        self.ends = np.concatenate([(self.starts - 1)[1:], np.array([len(batch_inds)])])

        n_layers = self.hidden_states_pi.shape[1]
        n_seq = len(self.starts)
        max_length = self.pad(self.actions[batch_inds]).shape[0]
        # TODO: output mask to not backpropagate everywhere
        padded_batch_size = n_seq * max_length
        lstm_states_pi = (
            # (n_steps, n_layers, n_envs, dim) -> (n_layers, n_seq, dim)
            self.hidden_states_pi[batch_inds][seq_start == True].reshape(n_layers, n_seq, -1),  # noqa: E712
            self.cell_states_pi[batch_inds][seq_start == True].reshape(n_layers, n_seq, -1),  # noqa: E712
        )
        lstm_states_vf = (
            # (n_steps, n_layers, n_envs, dim) -> (n_layers, n_seq, dim)
            self.hidden_states_vf[batch_inds][seq_start == True].reshape(n_layers, n_seq, -1),  # noqa: E712
            self.cell_states_vf[batch_inds][seq_start == True].reshape(n_layers, n_seq, -1),  # noqa: E712
        )
        lstm_states_pi = (self.to_torch(lstm_states_pi[0]), self.to_torch(lstm_states_pi[1]))
        lstm_states_vf = (self.to_torch(lstm_states_vf[0]), self.to_torch(lstm_states_vf[1]))

        observations = {key: self.pad(obs[batch_inds]) for (key, obs) in self.observations.items()}
        observations = {
            key: obs.swapaxes(0, 1).reshape((padded_batch_size,) + self.obs_shape[key]) for (key, obs) in observations.items()
        }

        return RecurrentDictRolloutBufferSamples(
            observations=observations,
            actions=self.pad(self.actions[batch_inds]).swapaxes(0, 1).reshape((padded_batch_size,) + self.actions.shape[1:]),
            old_values=self.pad(self.values[batch_inds]).swapaxes(0, 1).flatten(),
            old_log_prob=self.pad(self.log_probs[batch_inds]).swapaxes(0, 1).flatten(),
            advantages=self.pad(self.advantages[batch_inds]).swapaxes(0, 1).flatten(),
            returns=self.pad(self.returns[batch_inds]).swapaxes(0, 1).flatten(),
            lstm_states=RNNStates(lstm_states_pi, lstm_states_vf),
            episode_starts=self.pad(self.episode_starts[batch_inds]).swapaxes(0, 1).flatten(),
        )
