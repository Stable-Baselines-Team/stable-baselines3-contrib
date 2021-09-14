from typing import Any, Dict, List, Optional, Type
import gym
import torch as th
from torch import nn

from stable_baselines3.common.policies import BasePolicy, register_policy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor, FlattenExtractor, create_mlp
from stable_baselines3.common.preprocessing import get_action_dim

class ARSPolicy(BasePolicy):
    """
    Policy network for ARS

    :param observation_space: The observation space of the environment
    :param action_space: The action space of the environment
    :param net_arch: Network architecture, defaults to a linear policy with bias
    :param features_extractor_class: Feature extractor to use, defaults to a FlattenExtractor
    :param features_extractor_kwargs: Keyword arguments for the feature extractor
    """

    def __init__(self,
                 observation_space : gym.spaces.Space,
                 action_space : gym.spaces.Space,
                 net_arch: Optional[List[int]] = None,
                 features_extractor_class: Type[BaseFeaturesExtractor] = FlattenExtractor,
                 features_extractor_kwargs: Optional[Dict[str, Any]] = None,
                 ):

        super().__init__(
            observation_space,
            action_space,
            squash_output=isinstance(action_space, gym.spaces.Box),
        )

        if net_arch is None:
            net_arch = []

        self.net_arch = net_arch
        self.features_extractor = self.make_features_extractor()
        self.features_dim = self.features_extractor.features_dim

        if isinstance(action_space, gym.spaces.Box):
            action_dim = get_action_dim(action_space)
            actor_net = create_mlp(self.features_dim, action_dim, net_arch, squash_output=True)
        else:
            raise NotImplementedError("Error: ARS policy not implemented for action space" f"of type {type(action_space)}.")

        self.action_net = nn.Sequential(*actor_net)

    def _get_constructor_parameters(self) -> Dict[str, Any]:
        data = super()._get_constructor_parameters()
        data.update(dict(net_arch=self.net_arch))
        return data

    def forward(self, obs: th.Tensor) -> th.Tensor:
        features = self.extract_features(obs)

        if isinstance(self.action_space, gym.spaces.Box):
            return self.action_net(features)
        raise NotImplementedError()

    def _predict(self, observation: th.Tensor, deterministic: bool = True) -> th.Tensor:
        # Non deterministic action does not really make sense for ARS, we ignore this parameter for now..
        return self.forward(observation)


LinearPolicy = ARSPolicy

register_policy("LinearPolicy", LinearPolicy)

# Smoke test
# if __name__ == "__main__":
#     import pybullet_envs
#     env = gym.make("HalfCheetahBulletEnv-v0")
#     pol = ARSPolicy(env.observation_space, env.action_space)
#
#     print(pol.predict(env.reset()))