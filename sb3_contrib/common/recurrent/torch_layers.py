from copy import deepcopy
from typing import Optional, Tuple

import gym
import torch as th
from stable_baselines3.common.preprocessing import get_flattened_obs_dim
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.utils import zip_strict
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
        self._lstm_states = None
        self.debug = False

    def process_sequence(
        self,
        observations: th.Tensor,
        lstm_states: Tuple[th.Tensor, th.Tensor],
        episode_starts: th.Tensor,
    ) -> Tuple[th.Tensor, th.Tensor]:
        features = self.flatten(observations)

        # LSTM logic
        # (sequence length, n_envs, features dim) (batch size = n envs)
        n_envs = lstm_states[0].shape[1]
        # Note: order matters and should be consistent with the one from the buffer
        # above is when envs are interleaved
        # Batch to sequence
        features_sequence = features.reshape((n_envs, -1, self.lstm.input_size)).swapaxes(0, 1)
        episode_starts = episode_starts.reshape((n_envs, -1)).swapaxes(0, 1)
        # if self.debug:
        #     import ipdb; ipdb.set_trace()
        lstm_output = []
        # Iterate over the sequence
        for features, episode_start in zip_strict(features_sequence, episode_starts):
            hidden, lstm_states = self.lstm(
                features.unsqueeze(dim=0),
                (
                    (1.0 - episode_start).view(1, -1, 1) * lstm_states[0],
                    (1.0 - episode_start).view(1, -1, 1) * lstm_states[1],
                ),
            )
            # if self.debug:
            #     import ipdb; ipdb.set_trace()
            lstm_output += [hidden]
        # Sequence to batch
        lstm_output = th.flatten(th.cat(lstm_output).transpose(0, 1), start_dim=0, end_dim=1)
        return lstm_output, lstm_states

    @property
    def lstm_states(self) -> Tuple[th.Tensor, th.Tensor]:
        return self._lstm_states

    def forward(
        self,
        observations: th.Tensor,
        lstm_states: Tuple[th.Tensor, th.Tensor],
        episode_starts: th.Tensor,
    ) -> th.Tensor:
        features, self._lstm_states = self.process_sequence(observations, lstm_states, episode_starts)
        return features
