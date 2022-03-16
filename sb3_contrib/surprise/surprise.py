from typing import List, Optional, Type, Union

import torch as th
from stable_baselines3.common.off_policy_algorithm import OffPolicyAlgorithm
from stable_baselines3.common.surgeon import Surgeon
from stable_baselines3.common.type_aliases import ReplayBufferSamples
from stable_baselines3.common.utils import get_device
from torch import nn, optim
from torch.distributions import Normal

LOG_SIG_MAX = 2
LOG_SIG_MIN = -20


def sum_independent_dims(tensor: th.Tensor) -> th.Tensor:
    """
    Continuous actions are usually considered to be independent,
    so we can sum components of the ``log_prob`` or the entropy.
    :param tensor: shape: (n_batch, n_actions) or (n_batch,)
    :return: shape: (n_batch,)
    """
    if len(tensor.shape) > 1:
        tensor = tensor.sum(dim=1)
    else:
        tensor = tensor.sum()
    return tensor


class TransitionModel(nn.Module):
    """
    Transition model. Compute the probability of a next_obs given the obs and the action.

    :param obs_dim: Observation dimension
    :param action_dim: Observation dimension
    :param net_arch: The specification of the network
    :param feature_dim: Feature dimension
    :param activation_fn: The activation function to use for the networks
    :param device:
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        net_arch: List[int],
        feature_dim: int,
        activation_fn: Type[nn.Module],
        device: Union[th.device, str] = "auto",
    ) -> None:
        super().__init__()
        device = get_device(device)
        layers = []
        previous_layer_dim = obs_dim + action_dim
        # Iterate through the shared layers and build the shared parts of the network
        for layer_dim in net_arch:
            layers.append(nn.Linear(previous_layer_dim, layer_dim))  # add linear of size layer
            layers.append(activation_fn())
            previous_layer_dim = layer_dim
        layers.append(nn.Linear(previous_layer_dim, feature_dim))
        self.mean_net = nn.Linear(feature_dim, obs_dim).to(device)
        self.log_std_net = nn.Linear(feature_dim, obs_dim).to(device)

        # Create network
        self.net = nn.Sequential(*layers).to(device)

    def forward(self, obs: th.Tensor, action: th.Tensor, next_obs: th.Tensor) -> th.Tensor:
        obs_action = th.concat((obs, action), dim=-1)
        x = self.net(obs_action)
        mean = self.mean_net(x)
        log_std = self.log_std_net(x)
        log_std = th.clamp(log_std, min=LOG_SIG_MIN, max=LOG_SIG_MAX)
        std = th.ones_like(mean) * log_std.exp()
        distribution = Normal(mean, std)
        log_prob = distribution.log_prob(next_obs)
        log_prob = sum_independent_dims(log_prob)
        return log_prob


class Surprise(Surgeon):
    """
    Implementation of surprise-based intrinsic motivation.
    https://arxiv.org/abs/1703.01732

    :param obs_dim: Observation dimension
    :param action_dim: Action dimension
    :param net_arch: :param net_arch: The specification of the network, default to [64, 64]
    :param feature_dim: The feature dimension, defaults to 16
    :param eta_0: Desired average bonus, defaults to 1
    :param activation_fn: _description_, defaults to nn.ReLU
    :param device: _description_, defaults to "auto"

    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        net_arch: Optional[List[int]] = None,
        feature_dim: int = 16,
        eta_0: float = 1,
        train_freq: int = 1,
        weight_decay: float = 0.0,
        lr: float = 0.001,
        batch_size: int = 256,
        activation_fn: Type[nn.Module] = nn.ReLU,
        device: Union[th.device, str] = "auto",
    ) -> None:
        self.eta_0 = eta_0
        if net_arch is None:
            net_arch = [64, 64]
        self.transition_model = TransitionModel(obs_dim, action_dim, net_arch, feature_dim, activation_fn, device)
        self.train_freq = train_freq
        self.batch_size = batch_size
        self.optimizer = optim.Adam(self.transition_model.parameters(), lr=lr, weight_decay=weight_decay)
        self.last_time_trigger = 0
        self.model = None  # type: OffPolicyAlgorithm

    def modify_reward(self, replay_data: ReplayBufferSamples) -> ReplayBufferSamples:
        log_prob = self.transition_model(replay_data.observations, replay_data.actions, replay_data.next_observations)
        # normalize η by the average intrinsic reward of the current batch of data
        eta = self.eta_0 / max(1, th.mean(-log_prob))
        # compute the reshaped rewards
        intrinsic_rewards = -eta * log_prob
        new_rewards = replay_data.rewards + intrinsic_rewards.unsqueeze(1)
        new_replay_data = ReplayBufferSamples(
            replay_data.observations, replay_data.actions, replay_data.next_observations, replay_data.dones, new_rewards
        )
        self.logger.record("Surprise/rew_intr_mean", intrinsic_rewards.mean().item())
        self.logger.record("Surprise/rew_extr_mean", replay_data.rewards.mean().item())

        return new_replay_data

    def on_step(self):
        if (self.model.num_timesteps - self.last_time_trigger) >= self.train_freq:
            self.last_time_trigger = self.model.num_timesteps
            self.train_model_once()

    def train_model_once(self) -> None:
        try:
            batch = self.model.replay_buffer.sample(self.batch_size)  # (s,a,s')∈D
        except ValueError:
            return
        log_prob = self.transition_model(batch.observations, batch.actions, batch.next_observations)
        loss = -th.mean(log_prob)  # −1/|D| sum_{(s,a,s')∈D} logPφ(s′|s,a) + α∥φ∥^2
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.logger.record("Surprise/model_loss", loss.item())
