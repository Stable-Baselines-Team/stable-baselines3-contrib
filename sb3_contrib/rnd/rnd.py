from typing import List, Optional, Type, Union

import torch as th
import torch.nn.functional as F
from stable_baselines3.common.surgeon import Surgeon
from stable_baselines3.common.type_aliases import ReplayBufferSamples
from stable_baselines3.common.utils import get_device
from torch import nn


class Network(nn.Module):
    """
    Network. Used for predictor and target.

    :param obs_dim: Observation dimension
    :param net_arch: The specification of the network
    :param out_dim: Output dimension
    :param activation_fn: The activation function to use for the networks
    :param device:
    """

    def __init__(
        self,
        obs_dim: int,
        net_arch: List[int],
        out_dim: int,
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
        layers.append(nn.Linear(previous_layer_dim, out_dim))

        # Create network
        self.net = nn.Sequential(*layers).to(device)

    def forward(self, obs: th.Tensor) -> th.Tensor:
        out = self.net(obs)
        return out


class RND(Surgeon):
    """
    Random Distillation Network.

    :param scaling_factor: scaling factor for the intrinsic reward
    :param obs_dim: observation dimension
    :param net_arch: The specification of the network, default to [64, 64]
    :param out_dim: Output dimension
    :param activation_fn: The activation function to use for the networks, default to ReLU
    :param gradient_steps: How many gradient steps to do after each rollout (see ``train_freq``)
    :param device:
    """

    def __init__(
        self,
        scaling_factor: float,
        obs_dim: int,
        net_arch: Optional[List[int]] = None,
        out_dim: int = 64,
        activation_fn: Type[nn.Module] = nn.ReLU,
        train_freq: int = 1,
        gradient_steps: int = 1,
        weight_decay: float = 0,
        lr: float = 1e-3,
        batch_size: int = 256,
        device: Union[th.device, str] = "auto",
    ) -> None:
        self.scaling_factor = scaling_factor  # we use here a tuned scaling factor instead of normalization
        if net_arch is None:
            net_arch = [64, 64]
        self.target = Network(obs_dim, net_arch, out_dim, activation_fn, device)
        self.predictor = Network(obs_dim, net_arch, out_dim, activation_fn, device)

        self.train_freq = train_freq
        self.gradient_steps = gradient_steps
        self.batch_size = batch_size
        self.optimizer = th.optim.Adam(self.predictor.parameters(), lr=lr, weight_decay=weight_decay)
        self.criterion = th.nn.MSELoss()
        self.last_time_trigger = 0

    def modify_reward(self, replay_data: ReplayBufferSamples) -> ReplayBufferSamples:
        obs = replay_data.next_observations.to(th.float32)
        target = self.target(obs)
        pred = self.predictor(obs)
        pred_error = F.mse_loss(pred, target, reduction="none").mean(1).unsqueeze(1)  # unsqueeze to match reward shape
        intrinsic_reward = self.scaling_factor * pred_error
        new_rewards = replay_data.rewards + self.scaling_factor * intrinsic_reward.detach()
        new_replay_data = ReplayBufferSamples(
            replay_data.observations, replay_data.actions, replay_data.next_observations, replay_data.dones, new_rewards
        )
        self.logger.record("RND/rew_intr_mean", intrinsic_reward.mean().item())
        self.logger.record("RND/rew_extr_mean", replay_data.rewards.mean().item())
        return new_replay_data

    def on_step(self) -> bool:
        if (self.model.num_timesteps - self.last_time_trigger) >= self.train_freq:
            self.last_time_trigger = self.model.num_timesteps
            self.train_once()
        return True

    def train_once(self) -> None:
        for _ in range(self.gradient_steps):
            try:
                batch = self.model.replay_buffer.sample(self.batch_size)
            except ValueError:
                return
            obs = batch.observations.to(th.float32)
            pred = self.predictor(obs)
            target = self.target(obs)
            loss = self.criterion(pred, target.detach())
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        self.logger.record("RND/loss", loss.item())
