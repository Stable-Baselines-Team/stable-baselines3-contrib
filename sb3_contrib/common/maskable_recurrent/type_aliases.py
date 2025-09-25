from typing import NamedTuple

import torch as th
from stable_baselines3.common.type_aliases import TensorDict
from sb3_contrib.common.recurrent.type_aliases import (
    RNNStates,
)

class MaskableRecurrentRolloutBufferSamples(NamedTuple):
    observations: th.Tensor
    actions: th.Tensor
    old_values: th.Tensor
    old_log_prob: th.Tensor
    advantages: th.Tensor
    returns: th.Tensor
    lstm_states: RNNStates
    episode_starts: th.Tensor
    mask: th.Tensor
    action_masks: th.Tensor


class MaskableRecurrentDictRolloutBufferSamples(NamedTuple):
    observations: TensorDict
    actions: th.Tensor
    old_values: th.Tensor
    old_log_prob: th.Tensor
    advantages: th.Tensor
    returns: th.Tensor
    lstm_states: RNNStates
    episode_starts: th.Tensor
    mask: th.Tensor
    action_masks: th.Tensor
