from typing import Any, Dict, List, Optional, Type

import gym
import torch as th
from stable_baselines3.common.policies import BasePolicy, register_policy
from stable_baselines3.common.torch_layers import (
    BaseFeaturesExtractor,
    CombinedExtractor,
    FlattenExtractor,
    NatureCNN,
    create_mlp,
)
from stable_baselines3.common.type_aliases import Schedule
from torch import nn


class QuantileNetwork(BasePolicy):
    """
    Quantile network for QR-DQN

    :param observation_space: Observation space
    :param action_space: Action space
    :param n_quantiles: Number of quantiles
    :param net_arch: The specification of the network architecture.
    :param activation_fn: Activation function
    :param normalize_images: Whether to normalize images or not,
         dividing by 255.0 (True by default)
    """

    def __init__(
        self,
        observation_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        features_extractor: nn.Module,
        features_dim: int,
        n_quantiles: int = 200,
        net_arch: Optional[List[int]] = None,
        activation_fn: Type[nn.Module] = nn.ReLU,
        normalize_images: bool = True,
    ):
        super(QuantileNetwork, self).__init__(
            observation_space,
            action_space,
            features_extractor=features_extractor,
            normalize_images=normalize_images,
        )

        if net_arch is None:
            net_arch = [64, 64]

        self.net_arch = net_arch
        self.activation_fn = activation_fn
        self.features_extractor = features_extractor
        self.features_dim = features_dim
        self.n_quantiles = n_quantiles
        self.normalize_images = normalize_images
        action_dim = self.action_space.n  # number of actions
        quantile_net = create_mlp(self.features_dim, action_dim * self.n_quantiles, self.net_arch, self.activation_fn)
        self.quantile_net = nn.Sequential(*quantile_net)

    def forward(self, obs: th.Tensor) -> th.Tensor:
        """
        Predict the quantiles.

        :param obs: Observation
        :return: The estimated quantiles for each action.
        """
        quantiles = self.quantile_net(self.extract_features(obs))
        return quantiles.view(-1, self.n_quantiles, self.action_space.n)

    def _predict(self, observation: th.Tensor, deterministic: bool = True) -> th.Tensor:
        q_values = self.forward(observation).mean(dim=1)
        # Greedy action
        action = q_values.argmax(dim=1).reshape(-1)
        return action

    def _get_constructor_parameters(self) -> Dict[str, Any]:
        data = super()._get_constructor_parameters()

        data.update(
            dict(
                net_arch=self.net_arch,
                features_dim=self.features_dim,
                n_quantiles=self.n_quantiles,
                activation_fn=self.activation_fn,
                features_extractor=self.features_extractor,
            )
        )
        return data


class QRDQNPolicy(BasePolicy):
    """
    Policy class with quantile and target networks for QR-DQN.

    :param observation_space: Observation space
    :param action_space: Action space
    :param lr_schedule: Learning rate schedule (could be constant)
    :param n_quantiles: Number of quantiles
    :param net_arch: The specification of the network architecture.
    :param activation_fn: Activation function
    :param features_extractor_class: Features extractor to use.
    :param features_extractor_kwargs: Keyword arguments
        to pass to the features extractor.
    :param normalize_images: Whether to normalize images or not,
         dividing by 255.0 (True by default)
    :param optimizer_class: The optimizer to use,
        ``th.optim.Adam`` by default
    :param optimizer_kwargs: Additional keyword arguments,
        excluding the learning rate, to pass to the optimizer
    """

    def __init__(
        self,
        observation_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        lr_schedule: Schedule,
        n_quantiles: int = 200,
        net_arch: Optional[List[int]] = None,
        activation_fn: Type[nn.Module] = nn.ReLU,
        features_extractor_class: Type[BaseFeaturesExtractor] = FlattenExtractor,
        features_extractor_kwargs: Optional[Dict[str, Any]] = None,
        normalize_images: bool = True,
        optimizer_class: Type[th.optim.Optimizer] = th.optim.Adam,
        optimizer_kwargs: Optional[Dict[str, Any]] = None,
    ):

        super(QRDQNPolicy, self).__init__(
            observation_space,
            action_space,
            features_extractor_class,
            features_extractor_kwargs,
            optimizer_class=optimizer_class,
            optimizer_kwargs=optimizer_kwargs,
        )

        if net_arch is None:
            if features_extractor_class == NatureCNN:
                net_arch = []
            else:
                net_arch = [64, 64]

        self.n_quantiles = n_quantiles
        self.net_arch = net_arch
        self.activation_fn = activation_fn
        self.normalize_images = normalize_images

        self.net_args = {
            "observation_space": self.observation_space,
            "action_space": self.action_space,
            "n_quantiles": self.n_quantiles,
            "net_arch": self.net_arch,
            "activation_fn": self.activation_fn,
            "normalize_images": normalize_images,
        }

        self.quantile_net, self.quantile_net_target = None, None
        self._build(lr_schedule)

    def _build(self, lr_schedule: Schedule) -> None:
        """
        Create the network and the optimizer.

        :param lr_schedule: Learning rate schedule
            lr_schedule(1) is the initial learning rate
        """
        self.quantile_net = self.make_quantile_net()
        self.quantile_net_target = self.make_quantile_net()
        self.quantile_net_target.load_state_dict(self.quantile_net.state_dict())
        self.quantile_net_target.set_training_mode(False)

        # Setup optimizer with initial learning rate
        self.optimizer = self.optimizer_class(self.parameters(), lr=lr_schedule(1), **self.optimizer_kwargs)

    def make_quantile_net(self) -> QuantileNetwork:
        # Make sure we always have separate networks for features extractors etc
        net_args = self._update_features_extractor(self.net_args, features_extractor=None)
        return QuantileNetwork(**net_args).to(self.device)

    def forward(self, obs: th.Tensor, deterministic: bool = True) -> th.Tensor:
        return self._predict(obs, deterministic=deterministic)

    def _predict(self, obs: th.Tensor, deterministic: bool = True) -> th.Tensor:
        return self.quantile_net._predict(obs, deterministic=deterministic)

    def _get_constructor_parameters(self) -> Dict[str, Any]:
        data = super()._get_constructor_parameters()

        data.update(
            dict(
                n_quantiles=self.net_args["n_quantiles"],
                net_arch=self.net_args["net_arch"],
                activation_fn=self.net_args["activation_fn"],
                lr_schedule=self._dummy_schedule,  # dummy lr schedule, not needed for loading policy alone
                optimizer_class=self.optimizer_class,
                optimizer_kwargs=self.optimizer_kwargs,
                features_extractor_class=self.features_extractor_class,
                features_extractor_kwargs=self.features_extractor_kwargs,
            )
        )
        return data

    def set_training_mode(self, mode: bool) -> None:
        """
        Put the policy in either training or evaluation mode.
        This affects certain modules, such as batch normalisation and dropout.
        :param mode: if true, set to training mode, else set to evaluation mode
        """
        self.quantile_net.set_training_mode(mode)
        self.training = mode


MlpPolicy = QRDQNPolicy


class CnnPolicy(QRDQNPolicy):
    """
    Policy class for QR-DQN when using images as input.

    :param observation_space: Observation space
    :param action_space: Action space
    :param lr_schedule: Learning rate schedule (could be constant)
    :param n_quantiles: Number of quantiles
    :param net_arch: The specification of the network architecture.
    :param activation_fn: Activation function
    :param features_extractor_class: Features extractor to use.
    :param normalize_images: Whether to normalize images or not,
         dividing by 255.0 (True by default)
    :param optimizer_class: The optimizer to use,
        ``th.optim.Adam`` by default
    :param optimizer_kwargs: Additional keyword arguments,
        excluding the learning rate, to pass to the optimizer
    """

    def __init__(
        self,
        observation_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        lr_schedule: Schedule,
        n_quantiles: int = 200,
        net_arch: Optional[List[int]] = None,
        activation_fn: Type[nn.Module] = nn.ReLU,
        features_extractor_class: Type[BaseFeaturesExtractor] = NatureCNN,
        features_extractor_kwargs: Optional[Dict[str, Any]] = None,
        normalize_images: bool = True,
        optimizer_class: Type[th.optim.Optimizer] = th.optim.Adam,
        optimizer_kwargs: Optional[Dict[str, Any]] = None,
    ):
        super(CnnPolicy, self).__init__(
            observation_space,
            action_space,
            lr_schedule,
            n_quantiles,
            net_arch,
            activation_fn,
            features_extractor_class,
            features_extractor_kwargs,
            normalize_images,
            optimizer_class,
            optimizer_kwargs,
        )


class MultiInputPolicy(QRDQNPolicy):
    """
    Policy class for QR-DQN when using dict observations as input.

    :param observation_space: Observation space
    :param action_space: Action space
    :param lr_schedule: Learning rate schedule (could be constant)
    :param n_quantiles: Number of quantiles
    :param net_arch: The specification of the network architecture.
    :param activation_fn: Activation function
    :param features_extractor_class: Features extractor to use.
    :param normalize_images: Whether to normalize images or not,
         dividing by 255.0 (True by default)
    :param optimizer_class: The optimizer to use,
        ``th.optim.Adam`` by default
    :param optimizer_kwargs: Additional keyword arguments,
        excluding the learning rate, to pass to the optimizer
    """

    def __init__(
        self,
        observation_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        lr_schedule: Schedule,
        n_quantiles: int = 200,
        net_arch: Optional[List[int]] = None,
        activation_fn: Type[nn.Module] = nn.ReLU,
        features_extractor_class: Type[BaseFeaturesExtractor] = CombinedExtractor,
        features_extractor_kwargs: Optional[Dict[str, Any]] = None,
        normalize_images: bool = True,
        optimizer_class: Type[th.optim.Optimizer] = th.optim.Adam,
        optimizer_kwargs: Optional[Dict[str, Any]] = None,
    ):
        super(MultiInputPolicy, self).__init__(
            observation_space,
            action_space,
            lr_schedule,
            n_quantiles,
            net_arch,
            activation_fn,
            features_extractor_class,
            features_extractor_kwargs,
            normalize_images,
            optimizer_class,
            optimizer_kwargs,
        )


register_policy("MlpPolicy", MlpPolicy)
register_policy("CnnPolicy", CnnPolicy)
register_policy("MultiInputPolicy", MultiInputPolicy)
