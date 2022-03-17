from itertools import chain
from typing import Iterator, List, Optional, Type, Union

import torch as th
import torch.nn.functional as F
from stable_baselines3.common.surgeon import Surgeon
from stable_baselines3.common.type_aliases import ReplayBufferSamples
from stable_baselines3.common.utils import get_device
from torch import nn
from torch.nn.parameter import Parameter


class InverseModel(nn.Module):
    """
    Inverse model, used to predict action based on obs and next_obs.

    :param feature_dim: The feature dimension
    :param net_arch: The specification of the network
    :param action_dim: The action dimension
    :param activation_fn: The activation function to use for the networks
    :param device:
    """

    def __init__(
        self,
        feature_dim: int,
        net_arch: List[int],
        action_dim: int,
        activation_fn: Type[nn.Module],
        device: Union[th.device, str] = "auto",
    ) -> None:
        super().__init__()
        device = get_device(device)
        layers = []
        previous_layer_dim = feature_dim + feature_dim
        # Iterate through the shared layers and build the shared parts of the network
        for layer_dim in net_arch:
            layers.append(nn.Linear(previous_layer_dim, layer_dim))  # add linear of size layer
            layers.append(activation_fn())
            previous_layer_dim = layer_dim
        layers.append(nn.Linear(previous_layer_dim, action_dim))

        # Create network
        self.net = nn.Sequential(*layers).to(device)

    def forward(self, obs_feature: th.Tensor, next_obs_feature: th.Tensor) -> th.Tensor:
        x = th.concat((obs_feature, next_obs_feature), dim=-1)
        action = self.net(x)
        return action


class ForwardModel(nn.Module):
    """
    The forward model takes as inputs feature and action and predicts the next feature representation.

    :param feature_dim: The feature dimension
    :param action_dim: The action dimension
    :param net_arch: The specification of the network
    :param activation_fn: The activation function to use for the networks
    :param device:
    """

    def __init__(
        self,
        feature_dim: int,
        action_dim: int,
        net_arch: List[int],
        activation_fn: Type[nn.Module],
        device: Union[th.device, str] = "auto",
    ) -> None:
        super().__init__()
        device = get_device(device)
        layers = []
        previous_layer_dim = feature_dim + action_dim
        # Iterate through the shared layers and build the shared parts of the network
        for layer_dim in net_arch:
            layers.append(nn.Linear(previous_layer_dim, layer_dim))  # add linear of size layer
            layers.append(activation_fn())
            previous_layer_dim = layer_dim
        layers.append(nn.Linear(previous_layer_dim, feature_dim))

        # Create network
        self.net = nn.Sequential(*layers).to(device)

    def forward(self, action: th.Tensor, obs_feature: th.Tensor) -> th.Tensor:
        x = th.concat((action, obs_feature), dim=-1)
        next_obs_feature = self.net(x)
        return next_obs_feature


class FeatureExtractor(nn.Module):
    """
    The feature extractor takes as input an observation and returns its feature representation.

    :param obs_dim: Observation dimension
    :param net_arch: The specification of the network
    :param feature_dim: The feature dimension
    :param activation_fn: The activation function to use for the networks
    :param device:
    """

    def __init__(
        self,
        obs_dim: int,
        net_arch: List[int],
        feature_dim: int,
        activation_fn: Type[nn.Module],
        device: Union[th.device, str] = "auto",
    ) -> None:
        super().__init__()
        device = get_device(device)
        layers = []
        previous_layer_dim = obs_dim
        # Iterate through the shared layers and build the shared parts of the network
        for layer_dim in net_arch:
            layers.append(nn.Linear(previous_layer_dim, layer_dim))  # add linear of size layer
            layers.append(activation_fn())
            previous_layer_dim = layer_dim
        layers.append(nn.Linear(previous_layer_dim, feature_dim))

        # Create network
        self.net = nn.Sequential(*layers).to(device)

    def forward(self, obs: th.Tensor) -> th.Tensor:
        obs_feature = self.net(obs)
        return obs_feature


class VIME(Surgeon):
    def __init__(
        self,
        scaling_factor: float,
        actor_loss_coef: float,
        inverse_loss_coef: float,
        forward_loss_coef: float,
        obs_dim: int,
        action_dim: int,
        feature_dim: int = 16,
        net_arch: Optional[List[int]] = None,
        activation_fn: Type[nn.Module] = nn.ReLU,
        device: Union[th.device, str] = "auto",
    ):
        """
        VIME.

        :param scaling_factor: Scalar weights the intrinsic motivation
        :param actor_loss_coef: Coef for the actor loss in the loss computation
        :param inverse_loss_coef: Coef for the inverse loss in the loss computation
        :param forward_loss_coef: Coef for the forward loss in the loss computation
        :param obs_dim: Observation dimension
        :param action_dim: Action dimension
        :param feature_dim: The feature dimension, defaults to 16
        :param net_arch: The specification of the network, default to [64, 64]
        :param activation_fn: The activation function to use for the networks, default to ReLU
        :param device:
        """
        self.scaling_factor = scaling_factor
        self.actor_loss_coef = actor_loss_coef
        self.inverse_loss_coef = inverse_loss_coef
        self.forward_loss_coef = forward_loss_coef

        if net_arch is None:
            net_arch = [64, 64]

        self.forward_model = ForwardModel(feature_dim, action_dim, net_arch, activation_fn, device)
        self.inverse_model = InverseModel(feature_dim, net_arch, action_dim, activation_fn, device)
        self.feature_extractor = FeatureExtractor(obs_dim, net_arch, feature_dim, activation_fn, device)

    def parameters(self) -> Iterator[Parameter]:
        return chain(self.forward_model.parameters(), self.inverse_model.parameters(), self.feature_extractor.parameters())

    def modify_actor_loss(self, actor_loss: th.Tensor, replay_data: ReplayBufferSamples) -> th.Tensor:
        obs = replay_data.observations.to(th.float32)
        next_obs = replay_data.next_observations.to(th.float32)
        obs_feature = self.feature_extractor(obs)
        next_obs_feature = self.feature_extractor(next_obs)
        pred_action = self.inverse_model(obs_feature, next_obs_feature)
        pred_next_obs_feature = self.forward_model(replay_data.actions, obs_feature)
        # equation (5) of the original paper
        # 1/2*||φˆ(st+1)−φ(st+1)||^2
        forward_loss = F.mse_loss(pred_next_obs_feature, next_obs_feature)
        inverse_loss = F.mse_loss(pred_action, replay_data.actions)
        # equation (7) of the original paper
        # − λEπ(st;θP )[Σtrt] + (1 − β)LI + βLF
        new_actor_loss = (
            self.actor_loss_coef * actor_loss + self.inverse_loss_coef * inverse_loss + self.forward_loss_coef * forward_loss
        )
        self.logger.record("VIME/loss_forward", forward_loss.item())
        self.logger.record("VIME/loss_inverse", inverse_loss.item())
        return new_actor_loss

    def modify_reward(self, replay_data: ReplayBufferSamples) -> ReplayBufferSamples:
        obs = replay_data.observations.to(th.float32)
        next_obs = replay_data.next_observations.to(th.float32)
        obs_feature = self.feature_extractor(obs)
        next_obs_feature = self.feature_extractor(next_obs)
        # Equation (6) of the original paper
        # r^i = η/2*||φˆ(st+1)−φ(st+1)||
        pred_error = th.linalg.norm(obs_feature - next_obs_feature, dim=1)
        intrinsic_reward = self.scaling_factor * pred_error.unsqueeze(1).detach()  # unsqueeze to match reward shape
        new_rewards = replay_data.rewards + intrinsic_reward
        new_replay_data = ReplayBufferSamples(
            replay_data.observations, replay_data.actions, replay_data.next_observations, replay_data.dones, new_rewards
        )
        self.logger.record("VIME/rew_intr_mean", intrinsic_reward.mean().item())
        self.logger.record("VIME/rew_extr_mean", replay_data.rewards.mean().item())
        return new_replay_data
