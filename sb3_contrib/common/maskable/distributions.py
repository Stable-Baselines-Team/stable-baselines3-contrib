from abc import ABC, abstractmethod
from typing import TypeVar, Union

import numpy as np
import torch as th
from gymnasium import spaces
from stable_baselines3.common.distributions import Distribution
from torch import nn
from torch.distributions import Categorical
from torch.distributions.utils import probs_to_logits

SelfMaskableCategoricalDistribution = TypeVar("SelfMaskableCategoricalDistribution", bound="MaskableCategoricalDistribution")
SelfMaskableMultiCategoricalDistribution = TypeVar(
    "SelfMaskableMultiCategoricalDistribution", bound="MaskableMultiCategoricalDistribution"
)
MaybeMasks = Union[th.Tensor, np.ndarray, None]  # noqa: UP007


def _mask_logits(logits: th.Tensor, mask: MaybeMasks, neg_inf: float) -> th.Tensor:
    """
    Eliminate chosen categorical outcomes by setting their logits to `neg_inf`.

    :param logits: A tensor of unnormalized log probabilities (logits) for the categorical distribution.
        The shape should be compatible with the mask.

    :param mask: An optional boolean ndarray of compatible shape with the distribution.
        If True, the corresponding choice's logit value is preserved. If False, it is set
        to a large negative value, resulting in near 0 probability. If mask is None, any
        previously applied masking is removed, and the original logits are restored.

    :param neg_inf: The value to use for masked logits, typically negative infinity
        to ensure the masked actions have zero (or near-zero) probability when passed
        through a softmax or categorical distribution.
    """

    if mask is None:
        return logits
    mask_t = th.as_tensor(mask, dtype=th.bool, device=logits.device).reshape(logits.shape)
    return th.where(mask_t, logits, th.tensor(neg_inf, dtype=logits.dtype, device=logits.device))


class MaskableCategorical(Categorical):
    """
    Modified PyTorch Categorical distribution with support for invalid action masking.

    To instantiate, must provide either probs or logits, but not both.

    :param probs: Tensor containing finite non-negative values, which will be renormalized
        to sum to 1 along the last dimension.
    :param logits: Tensor of unnormalized log probabilities.
    :param validate_args: Whether or not to validate that arguments to methods like lob_prob()
        and icdf() match the distribution's shape, support, etc.
    :param masks: An optional boolean ndarray of compatible shape with the distribution.
        If True, the corresponding choice's logit value is preserved. If False, it is set to a
        large negative value, resulting in near 0 probability.
    """

    def __init__(
        self,
        probs: th.Tensor | None = None,
        logits: th.Tensor | None = None,
        validate_args: bool | None = None,
        masks: MaybeMasks = None,
    ):
        # Validate that exactly one of probs or logits is provided
        if (probs is None) == (logits is None):
            raise ValueError("Specify exactly one of probs or logits but not both.")

        # If probs provided, convert it to logits
        if logits is None:
            logits = probs_to_logits(probs)

        # Save pristine logits for later masking
        self._original_logits = logits.detach().clone()
        self._neg_inf = float("-inf")
        self.masks = None if masks is None else th.as_tensor(masks, dtype=th.bool, device=logits.device).reshape(logits.shape)
        masked_logits = _mask_logits(logits, self.masks, self._neg_inf)
        super().__init__(logits=masked_logits, validate_args=validate_args)

    def apply_masking(self, masks: MaybeMasks) -> None:
        if masks is None:
            self.masks = None
            logits = self._original_logits
        else:
            self.masks = th.as_tensor(masks, dtype=th.bool, device=self._original_logits.device).reshape(
                self._original_logits.shape
            )
            logits = _mask_logits(self._original_logits, self.masks, self._neg_inf)
        # Reinitialize with updated logits
        super().__init__(logits=logits)

    def entropy(self) -> th.Tensor:
        if self.masks is None:
            return super().entropy()

        # Prevent numerical issues with masked logits
        min_real = th.finfo(self.logits.dtype).min
        logits = self.logits.clone()
        mask = (~self.masks) | (~logits.isfinite())
        logits = logits.masked_fill(mask, min_real)
        logits = logits - logits.logsumexp(-1, keepdim=True)
        probs = logits.exp()
        return -(logits * probs).sum(-1)


class MaskableDistribution(Distribution, ABC):
    @abstractmethod
    def apply_masking(self, masks: MaybeMasks) -> None:
        """
        Eliminate ("mask out") chosen distribution outcomes by setting their probability to 0.

        :param masks: An optional boolean ndarray of compatible shape with the distribution.
            If True, the corresponding choice's logit value is preserved. If False, it is set
            to a large negative value, resulting in near 0 probability. If masks is None, any
            previously applied masking is removed, and the original logits are restored.
        """

    @abstractmethod
    def proba_distribution_net(self, *args, **kwargs) -> nn.Module:
        """Create the layers and parameters that represent the distribution.

        Subclasses must define this, but the arguments and return type vary between
        concrete classes."""


class MaskableCategoricalDistribution(MaskableDistribution):
    """
    Categorical distribution for discrete actions. Supports invalid action masking.

    :param action_dim: Number of discrete actions
    """

    distribution: MaskableCategorical

    def __init__(self, action_dim: int):
        super().__init__()
        self.action_dim = action_dim

    def proba_distribution_net(self, latent_dim: int) -> nn.Module:
        """
        Create the layer that represents the distribution:
        it will be the logits of the Categorical distribution.
        You can then get probabilities using a softmax.

        :param latent_dim: Dimension of the last layer
            of the policy network (before the action layer)
        :return:
        """
        action_logits = nn.Linear(latent_dim, self.action_dim)
        return action_logits

    def proba_distribution(
        self: SelfMaskableCategoricalDistribution, action_logits: th.Tensor
    ) -> SelfMaskableCategoricalDistribution:
        # Restructure shape to align with logits
        reshaped_logits = action_logits.view(-1, self.action_dim)
        self.distribution = MaskableCategorical(logits=reshaped_logits)
        return self

    def log_prob(self, actions: th.Tensor) -> th.Tensor:
        return self.distribution.log_prob(actions)

    def entropy(self) -> th.Tensor:
        return self.distribution.entropy()

    def sample(self) -> th.Tensor:
        return self.distribution.sample()

    def mode(self) -> th.Tensor:
        return th.argmax(self.distribution.probs, dim=1)

    def actions_from_params(self, action_logits: th.Tensor, deterministic: bool = False) -> th.Tensor:
        # Update the proba distribution
        self.proba_distribution(action_logits)
        return self.get_actions(deterministic=deterministic)

    def log_prob_from_params(self, action_logits: th.Tensor) -> tuple[th.Tensor, th.Tensor]:
        actions = self.actions_from_params(action_logits)
        log_prob = self.log_prob(actions)
        return actions, log_prob

    def apply_masking(self, masks: MaybeMasks) -> None:
        self.distribution.apply_masking(masks)


class MaskableMultiCategoricalDistribution(MaskableDistribution):
    """
    MultiCategorical distribution for multi discrete actions. Supports invalid action masking.

    :param action_dims: List of sizes of discrete action spaces
    """

    def __init__(self, action_dims: list[int]):
        super().__init__()
        self.distributions: list[MaskableCategorical] = []
        self.action_dims = action_dims

    def proba_distribution_net(self, latent_dim: int) -> nn.Module:
        """
        Create the layer that represents the distribution:
        it will be the logits (flattened) of the MultiCategorical distribution.
        You can then get probabilities using a softmax on each sub-space.

        :param latent_dim: Dimension of the last layer
            of the policy network (before the action layer)
        :return:
        """

        action_logits = nn.Linear(latent_dim, sum(self.action_dims))
        return action_logits

    def proba_distribution(
        self: SelfMaskableMultiCategoricalDistribution, action_logits: th.Tensor
    ) -> SelfMaskableMultiCategoricalDistribution:
        # Restructure shape to align with logits
        reshaped_logits = action_logits.view(-1, sum(self.action_dims))

        self.distributions = [
            MaskableCategorical(logits=split) for split in th.split(reshaped_logits, list(self.action_dims), dim=1)
        ]
        return self

    def log_prob(self, actions: th.Tensor) -> th.Tensor:
        assert len(self.distributions) > 0, "Must set distribution parameters"

        # Restructure shape to align with each categorical
        actions = actions.view(-1, len(self.action_dims))

        # Extract each discrete action and compute log prob for their respective distributions
        return th.stack(
            [dist.log_prob(action) for dist, action in zip(self.distributions, th.unbind(actions, dim=1), strict=True)], dim=1
        ).sum(dim=1)

    def entropy(self) -> th.Tensor:
        assert len(self.distributions) > 0, "Must set distribution parameters"
        return th.stack([dist.entropy() for dist in self.distributions], dim=1).sum(dim=1)

    def sample(self) -> th.Tensor:
        assert len(self.distributions) > 0, "Must set distribution parameters"
        return th.stack([dist.sample() for dist in self.distributions], dim=1)

    def mode(self) -> th.Tensor:
        assert len(self.distributions) > 0, "Must set distribution parameters"
        return th.stack([th.argmax(dist.probs, dim=1) for dist in self.distributions], dim=1)

    def actions_from_params(self, action_logits: th.Tensor, deterministic: bool = False) -> th.Tensor:
        # Update the proba distribution
        self.proba_distribution(action_logits)
        return self.get_actions(deterministic=deterministic)

    def log_prob_from_params(self, action_logits: th.Tensor) -> tuple[th.Tensor, th.Tensor]:
        actions = self.actions_from_params(action_logits)
        log_prob = self.log_prob(actions)
        return actions, log_prob

    def apply_masking(self, masks: MaybeMasks) -> None:
        assert len(self.distributions) > 0, "Must set distribution parameters"

        split_masks = [None] * len(self.distributions)
        if masks is not None:
            masks_tensor = th.as_tensor(masks)
            # Restructure shape to align with logits
            masks_tensor = masks_tensor.view(-1, sum(self.action_dims))
            # Then split columnwise for each discrete action
            split_masks = th.split(masks_tensor, list(self.action_dims), dim=1)  # type: ignore[assignment]

        for distribution, mask in zip(self.distributions, split_masks, strict=True):
            distribution.apply_masking(mask)


class MaskableBernoulliDistribution(MaskableMultiCategoricalDistribution):
    """
    Bernoulli distribution for multibinary actions. Supports invalid action masking.

    :param action_dim: Number of binary actions
    """

    def __init__(self, action_dim: int):
        # Two states per binary action
        action_dims = [2] * action_dim
        super().__init__(action_dims)


def make_masked_proba_distribution(action_space: spaces.Space) -> MaskableDistribution:
    """
    Return an instance of Distribution for the correct type of action space

    :param action_space: the input action space
    :return: the appropriate Distribution object
    """

    if isinstance(action_space, spaces.Discrete):
        return MaskableCategoricalDistribution(int(action_space.n))
    elif isinstance(action_space, spaces.MultiDiscrete):
        return MaskableMultiCategoricalDistribution(list(action_space.nvec))
    elif isinstance(action_space, spaces.MultiBinary):
        assert isinstance(
            action_space.n, int
        ), f"Multi-dimensional MultiBinary({action_space.n}) action space is not supported. You can flatten it instead."
        return MaskableBernoulliDistribution(action_space.n)
    else:
        raise NotImplementedError(
            "Error: probability distribution, not implemented for action space"
            f"of type {type(action_space)}."
            " Must be of type Gym Spaces: Discrete, MultiDiscrete."
        )
