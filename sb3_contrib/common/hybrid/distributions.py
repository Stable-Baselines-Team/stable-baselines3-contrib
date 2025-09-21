import numpy as np
import torch as th
from torch import nn
from typing import Any, Optional, TypeVar, Union
from stable_baselines3.common.distributions import Distribution
from torch.distributions import Categorical, Normal
from gymnasium import spaces


SelfHybridDistribution = TypeVar("SelfHybridDistribution", bound="HybridDistribution")


class HybridDistributionNet(nn.Module):
    """
    Base class for hybrid distributions that handle both discrete and continuous actions.
    This class should be extended to implement specific hybrid distributions.
    """

    def __init__(self, latent_dim: int, categorical_dimensions: np.ndarray, n_continuous: int):
        super().__init__()
        # For discrete action space
        self.categorical_nets = nn.ModuleList([nn.Linear(latent_dim, out_dim) for out_dim in categorical_dimensions])
        # For continuous action space
        self.gaussian_net = nn.Linear(latent_dim, n_continuous)
        
    def forward(self, latent: th.Tensor) -> tuple[list[th.Tensor], th.Tensor]:
        """
        Forward pass through all categorical nets and the gaussian net.

        :param latent: Latent tensor input
        :return: Tuple (list of categorical outputs, gaussian output)
        """
        categorical_outputs = [net(latent) for net in self.categorical_nets]
        gaussian_output = self.gaussian_net(latent)
        return categorical_outputs, gaussian_output


class Hybrid(th.distributions.Distribution):
    """
    A hybrid distribution that combines multiple categorical distributions for discrete actions
    and a Gaussian distribution for continuous actions.
    """

    def __init__(self,
        probs: Optional[tuple[list[th.Tensor], th.Tensor]] = None,
        logits: Optional[tuple[list[th.Tensor], th.Tensor]] = None,
        validate_args: Optional[bool] = None,
    ):
        super().__init__()
        categorical_logits: list[th.Tensor] = logits[0]
        gaussian_means: th.Tensor = logits[1]
        self.categorical_dists = [Categorical(logits=logit) for logit in categorical_logits]
        self.gaussian_dist = Normal(loc=gaussian_means, scale=th.ones_like(gaussian_means))
    
    def sample(self) -> tuple[th.Tensor, th.Tensor]:
        categorical_samples = [dist.sample() for dist in self.categorical_dists]
        gaussian_samples = self.gaussian_dist.sample()
        return th.stack(categorical_samples, dim=-1), gaussian_samples
        
    def log_prob(self) -> tuple[th.Tensor, th.Tensor]:
        """
        Returns the log probability of the given actions, both discrete and continuous.
        """
        gaussian_log_prob = self.gaussian_dist.log_prob(self.gaussian_dist.sample())
        categorical_log_probs = [dist.log_prob(dist.sample()) for dist in self.categorical_dists]

        # TODO: check dimensions
        return th.sum(th.stack(categorical_log_probs, dim=-1), dim=-1), th.sum(gaussian_log_prob, dim=-1)

    def entropy(self) -> tuple[th.Tensor, th.Tensor]:
        """
        Returns the entropy of the hybrid distribution, which is the sum of the entropies
        of the categorical and gaussian components.

        :return: Tuple of (categorical entropy, gaussian entropy)
        """
        categorical_entropies = [dist.entropy() for dist in self.categorical_dists]
        # Sum entropies for all categorical distributions
        categorical_entropy = th.sum(th.stack(categorical_entropies, dim=-1), dim=-1)
        gaussian_entropy = self.gaussian_dist.entropy().sum(dim=-1)
        return categorical_entropy, gaussian_entropy


class HybridDistribution(Distribution):
    def __init__(self, categorical_dimensions: np.ndarray, n_continuous: int):
        """
        Initialize the hybrid distribution with categorical and continuous components.
        
        :param categorical_dimensions: An array specifying the dimensions of the categorical actions.
        :param n_continuous: The number of continuous actions.
        """
        super().__init__()
        self.categorical_dimensions = categorical_dimensions
        self.n_continuous = n_continuous
        self.categorical_dists = None
        self.gaussian_dist = None
    
    def proba_distribution_net(self, latent_dim: int) -> Union[nn.Module, tuple[nn.Module, nn.Parameter]]:
        """Create the layers and parameters that represent the distribution.

        Subclasses must define this, but the arguments and return type vary between
        concrete classes."""
        action_net = HybridDistributionNet(latent_dim, self.categorical_dimensions)
        return action_net

    def proba_distribution(self: SelfHybridDistribution, action_logits: tuple[list[th.Tensor], th.Tensor]) -> SelfHybridDistribution:
        """Set parameters of the distribution.

        :return: self
        """
        self.distribution = Hybrid(logits=action_logits)
        return self

    def log_prob(self, discrete_actions: th.Tensor, continuous_actions: th.Tensor) -> tuple[th.Tensor, th.Tensor]:
        """
        Returns the log likelihood

        :param x: the taken action
        :return: The log likelihood of the distribution for discrete and continuous distributions
        """
        assert self.distribution is not None, "Must set distribution parameters"
        return self.distribution.log_prob(continuous_actions, discrete_actions)

    def entropy(self) -> tuple[th.Tensor, th.Tensor]:
        """
        Returns Shannon's entropy of the probability

        :return: the entropy of discrete and continuous distributions
        """
        assert self.distribution is not None, "Must set distribution parameters"
        return self.distribution.entropy()

    def sample(self) -> tuple[th.Tensor, th.Tensor]:
        """
        Returns a sample from the probability distribution

        :return: the stochastic action
        """
        assert self.distribution is not None, "Must set distribution parameters"
        return self.distribution.sample()

    def mode(self) -> th.Tensor:
        """
        Returns the most likely action (deterministic output)
        from the probability distribution

        :return: the stochastic action
        """

    def get_actions(self, deterministic: bool = False) -> tuple[th.Tensor, th.Tensor]:
        """
        Return actions according to the probability distribution.

        :param deterministic:
        :return:
        """
        if deterministic:
            return self.mode()
        return self.sample()

    def actions_from_params(self, *args, **kwargs) -> th.Tensor:
        """
        Returns samples from the probability distribution
        given its parameters.

        :return: actions
        """

    def log_prob_from_params(self, *args, **kwargs) -> tuple[th.Tensor, th.Tensor]:
        """
        Returns samples and the associated log probabilities
        from the probability distribution given its parameters.

        :return: actions and log prob
        """


def make_hybrid_proba_distribution(action_space: spaces.Tuple[spaces.MultiDiscrete, spaces.Box]) -> HybridDistribution:
    """
    Create a hybrid probability distribution for the given action space.

    :param action_space: Tuple Action space containing a MultiDiscrete action space and a Box action space.
    :return: A HybridDistribution object that handles the hybrid action space.
    """
    assert len(action_space[1].shape) == 1, "Continuous action space must have a monodimensional shape (e.g., (n,))"
    return HybridDistribution(
        categorical_dimensions=len(action_space[0].nvec),
        n_continuous=action_space[1].shape[0]
    )

