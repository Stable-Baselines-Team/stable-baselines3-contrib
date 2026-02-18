"""ActorQ policy network for REPPO in Stable-Baselines3.

This module implements the Actor-Q architecture which uses:
- A stochastic policy (actor) for action selection
- A distributional Q-function (critic) for value estimation using HL-Gauss encoding
- Separate observation processing for actor and critic
"""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import torch as th
import torch.nn as nn
from stable_baselines3.common.policies import BasePolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor, FlattenExtractor
from stable_baselines3.common.type_aliases import Schedule
from torch.distributions import Normal

from sb3_contrib.common.hl_gauss import HLGaussLayer


class TanhNormal:
    """Normal distribution followed by tanh squashing.

    Applies tanh to bound actions to [-1, 1] and corrects log probabilities
    for the change of variables.
    """

    def __init__(self, normal: Normal, eps: float = 1e-6):
        self.normal = normal
        self.eps = eps

    @property
    def mean(self) -> th.Tensor:
        """Deterministic action (tanh of the Gaussian mean)."""
        return th.tanh(self.normal.mean)

    def sample(self, sample_shape: tuple = ()) -> th.Tensor:
        """Sample with reparameterization (rsample) for pathwise gradients.

        Args:
            sample_shape: Shape prefix for multi-sample (e.g. (16,) for KL estimation).
        """
        return th.tanh(self.normal.rsample(sample_shape))

    def log_prob(self, actions: th.Tensor) -> th.Tensor:
        """Compute log probability of tanh-squashed actions.

        Inverts the tanh to recover the pre-squash value, computes the
        Gaussian log prob, then applies the Jacobian correction.
        """
        # Clip to valid atanh domain
        actions_clipped = actions.clamp(-1.0 + self.eps, 1.0 - self.eps)
        # Inverse tanh
        pre_squash = th.atanh(actions_clipped)
        # Gaussian log prob
        lp = self.normal.log_prob(pre_squash)
        # Jacobian correction: log |d tanh / dx| = log(1 - tanh(x)^2)
        lp = lp - th.log(1.0 - actions_clipped.pow(2) + self.eps)
        return lp


class ActorQNetwork(nn.Module):
    """Actor-Q network combining a stochastic actor with a distributional critic.
    
    Args:
        observation_space: Observation space
        action_space: Action space  
        net_arch: Network architecture for actor and critic
        activation_fn: Activation function
        num_critic_bins: Number of bins for distributional critic
        vmin: Minimum value for critic support
        vmax: Maximum value for critic support
        state_dependent_std: Whether to use state-dependent std for actor
        log_std_init: Initial value for log std
    """
    
    def __init__(
        self,
        observation_dim: int,
        action_dim: int,
        net_arch: dict[str, list[int]],
        activation_fn: type[nn.Module] = nn.SiLU,
        num_critic_bins: int = 151,
        vmin: float = -10.0,
        vmax: float = 10.0,
        state_dependent_std: bool = True,
        log_std_init: float = 0.0,
    ):
        super().__init__()
        
        self.observation_dim = observation_dim
        self.action_dim = action_dim
        self.num_critic_bins = num_critic_bins
        self.vmin = vmin
        self.vmax = vmax
        self.state_dependent_std = state_dependent_std
        self.activation_fn = activation_fn
        
        # Build actor network
        actor_layers = []
        last_dim = observation_dim
        for hidden_dim in net_arch.get("pi", [256, 256, 256]):
            actor_layers.append(nn.Linear(last_dim, hidden_dim))
            actor_layers.append(activation_fn())
            last_dim = hidden_dim
        
        if state_dependent_std:
            # Output both mean and log_std
            actor_layers.append(nn.Linear(last_dim, 2 * action_dim))
            self.actor = nn.Sequential(*actor_layers)
        else:
            # Output only mean, log_std is a parameter
            actor_layers.append(nn.Linear(last_dim, action_dim))
            self.actor = nn.Sequential(*actor_layers)
            self.log_std = nn.Parameter(th.ones(action_dim) * log_std_init, requires_grad=True)
        
        # Build critic network (takes obs + action as input)
        critic_input_dim = observation_dim + action_dim
        critic_layers = []
        last_dim = critic_input_dim
        critic_arch = net_arch.get("qf", [256, 256, 256])
        for hidden_dim in critic_arch[:-1]:
            critic_layers.append(nn.Linear(last_dim, hidden_dim))
            critic_layers.append(activation_fn())
            last_dim = hidden_dim
        
        # Last layer before HL-Gauss
        self.critic_features = nn.Sequential(*critic_layers)
        self.critic_final = nn.Linear(last_dim, critic_arch[-1])
        self.critic_norm = nn.LayerNorm(critic_arch[-1])
        
        # Distributional critic output using HL-Gauss
        self.critic_embedding = HLGaussLayer(
            in_features=critic_arch[-1],
            min_value=vmin,
            max_value=vmax,
            num_bins=num_critic_bins,
            sigma=0.75,
            offset_mult=40.0,
        )
        
        # Forward prediction head for auxiliary self-prediction loss
        # Paper: ℒ_aux = ||f_p(ψ_t) - sg(ψ_{t+1})||²
        # where ψ = encoder output (critic features before HL-Gauss)
        encoder_dim = critic_arch[-1]
        self.predictor = nn.Sequential(
            nn.Linear(encoder_dim, encoder_dim),
            activation_fn(),
            nn.Linear(encoder_dim, encoder_dim),
        )

        # Temperature parameters (trainable)
        self.log_alpha_temp = nn.Parameter(th.log(th.tensor(0.1)), requires_grad=True)
        self.log_alpha_kl = nn.Parameter(th.log(th.tensor(0.1)), requires_grad=True)

        # Distribution
        self.action_dist: TanhNormal | None = None
        
    @property
    def alpha_temp(self) -> th.Tensor:
        """Temperature parameter for entropy regularization."""
        return self.log_alpha_temp.exp()
    
    @property
    def alpha_kl(self) -> th.Tensor:
        """Temperature parameter for KL regularization."""
        return self.log_alpha_kl.exp()
    
    def get_action_dist(self, obs: th.Tensor) -> TanhNormal:
        """Get tanh-squashed action distribution from observations.

        Args:
            obs: Observations

        Returns:
            TanhNormal distribution over actions (bounded to [-1, 1])
        """
        if self.state_dependent_std:
            mean_and_logstd = self.actor(obs)
            mean = mean_and_logstd[..., :self.action_dim]
            log_std = mean_and_logstd[..., self.action_dim:]
            log_std = th.clamp(log_std, min=-5.0, max=2.0)
            std = th.exp(log_std)
        else:
            mean = self.actor(obs)
            std = self.log_std.exp().expand_as(mean)

        self.action_dist = TanhNormal(Normal(mean, std))
        return self.action_dist
    
    def forward(self, obs: th.Tensor, deterministic: bool = False) -> th.Tensor:
        """Get actions from observations.
        
        Args:
            obs: Observations
            deterministic: Whether to return mean action
            
        Returns:
            Actions
        """
        dist = self.get_action_dist(obs)
        if deterministic:
            return dist.mean
        return dist.sample()
    
    def _critic_encoder(self, obs: th.Tensor, actions: th.Tensor) -> th.Tensor:
        """Shared critic encoder: Linear → LayerNorm → Activation."""
        critic_input = th.cat([obs, actions], dim=-1)
        features = self.critic_features(critic_input)
        features = self.critic_final(features)
        features = self.critic_norm(features)
        features = self.activation_fn()(features)
        return features

    def evaluate_actions(
        self,
        obs: th.Tensor,
        actions: th.Tensor,
    ) -> th.Tensor:
        """Evaluate Q-values for given observations and actions.

        Args:
            obs: Observations
            actions: Actions to evaluate

        Returns:
            Q-values (scalar values)
        """
        features = self._critic_encoder(obs, actions)
        return self.critic_embedding(features, return_logits=False)
    
    def evaluate_actions_with_logits(
        self,
        obs: th.Tensor,
        actions: th.Tensor,
    ) -> tuple[th.Tensor, th.Tensor]:
        """Evaluate Q-values and get distributional logits.

        Args:
            obs: Observations
            actions: Actions to evaluate

        Returns:
            Tuple of (Q-values, distributional logits)
        """
        features = self._critic_encoder(obs, actions)
        return self.critic_embedding(features, return_logits=True)

    def critic_forward_full(
        self,
        obs: th.Tensor,
        actions: th.Tensor,
    ) -> tuple[th.Tensor, th.Tensor, th.Tensor]:
        """Full critic forward returning Q-values, logits, and predictor output.

        Used during critic training to compute both distributional loss
        and auxiliary embedding loss in a single forward pass.

        Returns:
            Tuple of (Q-values, distributional logits, predictor_output)
        """
        features = self._critic_encoder(obs, actions)
        q_value, logits = self.critic_embedding(features, return_logits=True)
        pred = self.predictor(features)
        return q_value, logits, pred

    def get_encoder_features(self, obs: th.Tensor, actions: th.Tensor) -> th.Tensor:
        """Get encoder representation ψ = f_φ(x, a) for auxiliary loss.

        Returns the activated critic features before the HL-Gauss head.
        """
        return self._critic_encoder(obs, actions)

    def get_distribution(self, obs: th.Tensor) -> TanhNormal:
        """Get action distribution (alias for compatibility)."""
        return self.get_action_dist(obs)


class ActorQPolicy(BasePolicy):
    """ActorQ policy class for REPPO.
    
    This policy wraps the ActorQNetwork and provides the interface
    required by Stable-Baselines3.
    
    Args:
        observation_space: Observation space
        action_space: Action space
        lr_schedule: Learning rate schedule
        net_arch: Network architecture
        activation_fn: Activation function
        num_critic_bins: Number of bins for distributional critic
        vmin: Minimum value for critic support
        vmax: Maximum value for critic support
        state_dependent_std: Whether to use state-dependent std
    """
    
    def __init__(
        self,
        observation_space: gym.Space,
        action_space: gym.Space,
        lr_schedule: Schedule,
        net_arch: list[int] | dict[str, list[int]] | None = None,
        activation_fn: type[nn.Module] = nn.SiLU,
        num_critic_bins: int = 151,
        vmin: float = -10.0,
        vmax: float = 10.0,
        state_dependent_std: bool = True,
        features_extractor_class: type[BaseFeaturesExtractor] = FlattenExtractor,
        features_extractor_kwargs: dict[str, Any] | None = None,
        normalize_images: bool = True,
        optimizer_class: type[th.optim.Optimizer] = th.optim.AdamW,
        optimizer_kwargs: dict[str, Any] | None = None,
        use_sde: bool = False,  # Added for compatibility with SB3
        **kwargs,  # Catch any other SB3-specific arguments
    ):
        if net_arch is None:
            net_arch = {"pi": [256, 256, 256], "qf": [256, 256, 256]}
        elif isinstance(net_arch, list):
            # SB3 convention: list[int] means shared architecture for both networks
            net_arch = {"pi": net_arch, "qf": net_arch}
        
        if optimizer_kwargs is None:
            optimizer_kwargs = {"weight_decay": 1e-3, "betas": (0.9, 0.95)}
        
        super().__init__(
            observation_space,
            action_space,
            features_extractor_class,
            features_extractor_kwargs,
            optimizer_class=optimizer_class,
            optimizer_kwargs=optimizer_kwargs,
            normalize_images=normalize_images,
        )
        
        self.net_arch = net_arch
        self.activation_fn = activation_fn
        self.num_critic_bins = num_critic_bins
        self.vmin = vmin
        self.vmax = vmax
        self.state_dependent_std = state_dependent_std
        
        self._build(lr_schedule)
    
    def _build(self, lr_schedule: Schedule) -> None:
        """Build the networks."""
        # Get observation and action dimensions
        obs_dim = self.observation_space.shape[0]
        action_dim = self.action_space.shape[0]
        
        # Create the network
        self.q_net = ActorQNetwork(
            observation_dim=obs_dim,
            action_dim=action_dim,
            net_arch=self.net_arch,
            activation_fn=self.activation_fn,
            num_critic_bins=self.num_critic_bins,
            vmin=self.vmin,
            vmax=self.vmax,
            state_dependent_std=self.state_dependent_std,
        )
        
        # Setup optimizer
        self.optimizer = self.optimizer_class(
            self.parameters(), 
            lr=lr_schedule(1),
            **self.optimizer_kwargs
        )
    
    def _predict(self, observation: th.Tensor, deterministic: bool = False) -> th.Tensor:
        """Get the action from an observation (returns only actions for BasePolicy.predict compatibility)."""
        actions, _, _ = self(observation, deterministic=deterministic)
        return actions

    def forward(self, obs: th.Tensor, deterministic: bool = False) -> tuple[th.Tensor, th.Tensor, th.Tensor]:
        """
        Full policy forward compatible with SB3 `ActorCriticPolicy` interface.

        Returns (actions, values, log_prob) where:
        - `actions` has shape (batch, *action_shape)
        - `values` has shape (batch, 1)
        - `log_prob` has shape (batch,) (or (batch, 1) for some spaces)
        """
        # Use the raw observation tensor (already a torch tensor)
        dist = self.q_net.get_action_dist(obs)
        if deterministic:
            actions = dist.mean
        else:
            actions = dist.sample()

        # Ensure actions shape matches action space
        actions = actions.reshape((-1, *self.action_space.shape))  # type: ignore[misc]

        # Evaluate critic for these actions
        values = self.q_net.evaluate_actions(obs, actions).unsqueeze(-1)

        # Log probability of actions
        log_prob = dist.log_prob(actions)
        if log_prob.dim() > 1:
            log_prob = log_prob.sum(-1)

        return actions, values, log_prob

    def predict_values(self, obs: th.Tensor) -> th.Tensor:
        """Return value estimates for given observations (shape: batch x 1)."""
        # Use deterministic mean action for value prediction
        with th.no_grad():
            dist = self.q_net.get_action_dist(obs)
            mean_action = dist.mean
            values = self.q_net.evaluate_actions(obs, mean_action).unsqueeze(-1)
        return values
    
    def get_distribution(self, obs: th.Tensor) -> TanhNormal:
        """Get the action distribution."""
        return self.q_net.get_distribution(obs)
    
    def get_critic_parameters(self):
        """Get all critic parameters for freezing/unfreezing."""
        return [
            self.q_net.critic_features.parameters(),
            self.q_net.critic_final.parameters(),
            self.q_net.critic_norm.parameters(),
            self.q_net.critic_embedding.parameters(),
        ]
    
    def evaluate_actions(
        self, 
        obs: th.Tensor, 
        actions: th.Tensor,
    ) -> th.Tensor:
        """Evaluate Q-values for given observations and actions."""
        return self.q_net.evaluate_actions(obs, actions)


# Aliases for compatibility with SB3 conventions
MlpPolicy = ActorQPolicy
CnnPolicy = ActorQPolicy  # REPPO doesn't currently support CNN
MultiInputPolicy = ActorQPolicy  # REPPO doesn't currently support dict observations
