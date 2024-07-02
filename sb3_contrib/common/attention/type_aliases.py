from typing import NamedTuple, Tuple

import torch as th
from stable_baselines3.common.type_aliases import TensorDict


class AttnMemory(NamedTuple):
    pi: th.Tensor #Tuple[th.Tensor, ...]
    # vf: Tuple[th.Tensor, ...]


class AttentionRolloutBufferSamples(NamedTuple):
    observations: th.Tensor
    actions: th.Tensor
    old_values: th.Tensor
    old_log_prob: th.Tensor
    advantages: th.Tensor
    returns: th.Tensor
    attn_memory: AttnMemory
    episode_starts: th.Tensor
    mask: th.Tensor


class AttentionDictRolloutBufferSamples(NamedTuple):
    observations: TensorDict
    actions: th.Tensor
    old_values: th.Tensor
    old_log_prob: th.Tensor
    advantages: th.Tensor
    returns: th.Tensor
    attn_memory: AttnMemory
    episode_starts: th.Tensor
    mask: th.Tensor
