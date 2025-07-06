from typing import Any, Optional, Union
from stable_baselines3.common.policies import BasePolicy
from gymnasium import spaces
from stable_baselines3.common.type_aliases import PyTorchObs, Schedule
import torch as th
from torch import nn
from stable_baselines3.common.torch_layers import (
    BaseFeaturesExtractor,
    CombinedExtractor,
    FlattenExtractor,
    MlpExtractor,
    NatureCNN,
)


class HybridActorCriticPolicy(BasePolicy):
    pass


# TODO: check superclass
class HybridActorCriticCnnPolicy(HybridActorCriticPolicy):
    pass


# TODO: check superclass
class HybridMultiInputActorCriticPolicy(HybridActorCriticPolicy):
    pass