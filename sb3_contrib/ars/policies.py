from typing import Any, Dict, List, Optional, Union

import gym
import torch as th
from stable_baselines3.common.policies import BaseModel, BasePolicy, register_policy
from stable_baselines3.common.preprocessing import get_action_dim
from stable_baselines3.common.torch_layers import create_mlp
from torch import nn


class ARSPolicy(BasePolicy):
    """
    Policy network for ARS

    :param observation_space: The observation space of the environment
    :param action_space: The action space of the environment
    :param net_arch: Network architecture, defaults to a linear policy with bias
    """

    def __init__(
        self,
        observation_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        net_arch: Optional[List[int]] = None,
    ):

        super().__init__(
            observation_space,
            action_space,
            squash_output=isinstance(action_space, gym.spaces.Box),
        )

        if net_arch is None:
            net_arch = [64, 64]

        self.net_arch = net_arch
        self.features_extractor = self.make_features_extractor()
        self.features_dim = self.features_extractor.features_dim

        if isinstance(action_space, gym.spaces.Box):
            action_dim = get_action_dim(action_space)
            actor_net = create_mlp(self.features_dim, action_dim, net_arch, squash_output=False)
        else:
            raise NotImplementedError("Error: ARS policy not implemented for action space" f"of type {type(action_space)}.")

        self.action_net = nn.Sequential(*actor_net)

    @classmethod  # override to set default device to cpu
    def load(cls, path: str, device: Union[th.device, str] = "cpu") -> "BaseModel":
        return super().load(path, device)

    def _get_constructor_parameters(self) -> Dict[str, Any]:
        # data = super()._get_constructor_parameters() this adds normalize_images, which we don't support...
        data = dict(observation_space=self.observation_space, action_space=self.action_space, net_arch=self.net_arch)
        return data

    def forward(self, obs: th.Tensor) -> th.Tensor:
        features = self.extract_features(obs)

        if isinstance(self.action_space, gym.spaces.Box):
            return self.action_net(features)
        raise NotImplementedError()

    def _predict(self, observation: th.Tensor, deterministic: bool = True) -> th.Tensor:
        # Non deterministic action does not really make sense for ARS, we ignore this parameter for now..
        return self.forward(observation)


class ARSLinearPolicy(ARSPolicy):
    def __init__(
        self,
        observation_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
    ):

        net_arch = []
        super().__init__(observation_space, action_space, net_arch)


MlpPolicy = ARSPolicy
LinearPolicy = ARSLinearPolicy


register_policy("LinearPolicy", LinearPolicy)
register_policy("MlpPolicy", MlpPolicy)
