import numpy as np
import torch as th
from torch import nn
from typing import Any, Optional, TypeVar, Union
from stable_baselines3.common.distributions import Distribution
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

    def proba_distribution(self: SelfHybridDistribution, *args, **kwargs) -> SelfHybridDistribution:
        """Set parameters of the distribution.

        :return: self
        """

    def log_prob(self, x: th.Tensor) -> th.Tensor:
        """
        Returns the log likelihood

        :param x: the taken action
        :return: The log likelihood of the distribution
        """

    def entropy(self) -> Optional[th.Tensor]:
        """
        Returns Shannon's entropy of the probability

        :return: the entropy, or None if no analytical form is known
        """

    def sample(self) -> th.Tensor:
        """
        Returns a sample from the probability distribution

        :return: the stochastic action
        """

    def mode(self) -> th.Tensor:
        """
        Returns the most likely action (deterministic output)
        from the probability distribution

        :return: the stochastic action
        """

    # TODO: this is not abstract in superclass, you can also not re-implement it --> check
    def get_actions(self, deterministic: bool = False) -> th.Tensor:
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

