from typing import Any, Dict, List, Optional, Type

import gym
import torch as th
from stable_baselines3.common.policies import BasePolicy
from stable_baselines3.common.preprocessing import get_action_dim
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor, create_mlp
from torch import nn


class ARSPolicy(BasePolicy):
    """
    Policy network for ARS.

    :param observation_space: The observation space of the environment
    :param action_space: The action space of the environment
    :param net_arch: Network architecture, defaults to a 2 layers MLP with 64 hidden nodes.
    :param activation_fn: Activation function
    :param with_bias: If set to False, the layers will not learn an additive bias
    :param squash_output: For continuous actions, whether the output is squashed
        or not using a ``tanh()`` function. If not squashed with tanh the output will instead be clipped.
    """

    def __init__(
        self,
        observation_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        net_arch: Optional[List[int]] = None,
        activation_fn: Type[nn.Module] = nn.ReLU,
        with_bias: bool = True,
        squash_output: bool = True,
    ):

        super().__init__(
            observation_space,
            action_space,
            squash_output=isinstance(action_space, gym.spaces.Box) and squash_output,
        )

        if net_arch is None:
            net_arch = [64, 64]

        self.net_arch = net_arch
        self.features_extractor = self.make_features_extractor()
        self.features_dim = self.features_extractor.features_dim
        self.activation_fn = activation_fn

        if isinstance(action_space, gym.spaces.Box):
            action_dim = get_action_dim(action_space)
            actor_net = create_mlp(
                self.features_dim, action_dim, net_arch, activation_fn, with_bias=with_bias, squash_output=squash_output
            )
        elif isinstance(action_space, gym.spaces.Discrete):
            actor_net = create_mlp(self.features_dim, action_space.n, net_arch, activation_fn, with_bias=with_bias)
        else:
            raise NotImplementedError(f"Error: ARS policy not implemented for action space of type {type(action_space)}.")

        self.action_net = nn.Sequential(*actor_net)

    def _get_constructor_parameters(self) -> Dict[str, Any]:
        # data = super()._get_constructor_parameters() this adds normalize_images, which we don't support...
        data = dict(
            observation_space=self.observation_space,
            action_space=self.action_space,
            net_arch=self.net_arch,
            activation_fn=self.activation_fn,
        )
        return data

    def forward(self, obs: th.Tensor) -> th.Tensor:
        # Make mypy happy:
        assert isinstance(self.features_extractor, BaseFeaturesExtractor)

        features = self.extract_features(obs, self.features_extractor)
        if isinstance(self.action_space, gym.spaces.Box):
            return self.action_net(features)
        elif isinstance(self.action_space, gym.spaces.Discrete):
            logits = self.action_net(features)
            return th.argmax(logits, dim=1)
        else:
            raise NotImplementedError()

    def _predict(self, observation: th.Tensor, deterministic: bool = True) -> th.Tensor:
        # Non deterministic action does not really make sense for ARS, we ignore this parameter for now..
        return self(observation)


class ARSLinearPolicy(ARSPolicy):
    """
    Linear policy network for ARS.

    :param observation_space: The observation space of the environment
    :param action_space: The action space of the environment
    :param with_bias: With or without bias on the output
    :param squash_output: For continuous actions, whether the output is squashed
        or not using a ``tanh()`` function. If not squashed with tanh the output will instead be clipped.
    """

    def __init__(
        self,
        observation_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        with_bias: bool = False,
        squash_output: bool = False,
    ):

        super().__init__(observation_space, action_space, net_arch=[], with_bias=with_bias, squash_output=squash_output)


MlpPolicy = ARSPolicy
LinearPolicy = ARSLinearPolicy
