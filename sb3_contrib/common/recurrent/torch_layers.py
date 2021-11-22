from copy import deepcopy
from typing import Optional, Tuple

import gym
import torch as th
from stable_baselines3.common.preprocessing import get_flattened_obs_dim
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from torch import nn


class LSTMExtractor(BaseFeaturesExtractor):
    """
    Feature extract that pass the data through an LSTM after flattening it.
    Used as a placeholder when feature extraction is not needed.

    :param observation_space:
    """

    def __init__(self, observation_space: gym.Space, hidden_size: int = 64, num_layers: int = 1):
        super().__init__(observation_space, hidden_size)
        self.flatten = nn.Flatten()
        self.lstm = nn.LSTM(get_flattened_obs_dim(observation_space), hidden_size, num_layers=num_layers)
        # One forward pass to initial hidden state
        # dummy_cell_state, dummy_hidden = self.lstm()
        # Cell and hidden state
        n_envs = 1
        self.initial_hidden_state = (th.zeros(num_layers, n_envs, hidden_size), th.zeros(num_layers, n_envs, hidden_size))
        self._lstm_states = deepcopy(self.initial_hidden_state)
        self.dones = None

    def reset_state(self) -> None:
        self._lstm_states = deepcopy(self.initial_hidden_state)
        self.dones = None

    def process_sequence(
        self,
        observations: th.Tensor,
        lstm_states: Tuple[th.Tensor, th.Tensor],
        dones: th.Tensor,
    ) -> Tuple[th.Tensor, th.Tensor]:
        features = self.flatten(observations)

        # LSTM logic
        # (sequence length, batch size, features dim) (batch size = n envs)
        batch_size = lstm_states[0].shape[1]
        features_sequence = features.reshape((-1, batch_size, self.lstm.input_size))
        dones = dones.reshape((-1, batch_size))
        lstm_output = []
        # Iterate over the sequence
        for features, done in zip(features_sequence, dones):
            hidden, lstm_states = self.lstm(
                features.unsqueeze(0),
                (
                    (1.0 - done).view(1, -1, 1) * lstm_states[0],
                    (1.0 - done).view(1, -1, 1) * lstm_states[1],
                ),
            )
            lstm_output += [hidden]
        lstm_output = th.flatten(th.cat(lstm_output), start_dim=0, end_dim=1)
        return lstm_output, lstm_states

    def set_lstm_states(self, lstm_states: Optional[Tuple[th.Tensor]] = None) -> None:
        if lstm_states is None:
            self.reset_state()
        else:
            self._lstm_states = deepcopy(lstm_states)

    def set_dones(self, dones: th.Tensor) -> None:
        self.dones = dones

    @property
    def lstm_states(self) -> Tuple[th.Tensor, th.Tensor]:
        return self._lstm_states

    def forward(self, observations: th.Tensor) -> th.Tensor:
        if self.dones is None:
            self.dones = th.zeros(len(observations)).float().to(observations.device)
        features, self._lstm_states = self.process_sequence(observations, self._lstm_states, self.dones)
        return features
