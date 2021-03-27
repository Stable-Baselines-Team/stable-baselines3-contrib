from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import torch as th
from stable_baselines3.common.distributions import Distribution
from torch import nn
from torch.distributions import Categorical


class MaskableCategorical(Categorical):
    # Adapted from https://github.com/vwxyzjn/invalid-action-masking

    def __init__(
        self,
        probs=None,
        logits=None,
        validate_args=None,
        masks: Optional[np.ndarray] = None,
    ):
        self.masks: Optional[th.Tensor] = None
        super().__init__(probs, logits, validate_args)
        self.apply_masking(masks)

    def apply_masking(self, masks: Optional[np.ndarray]) -> None:
        if masks is not None:
            device = self.logits.device
            self.masks = th.as_tensor(masks, dtype=th.bool, device=device).reshape(
                self.logits.shape
            )
            HUGE_NEG = th.tensor(-1e8, dtype=self.logits.dtype, device=device)

            logits = th.where(self.masks, self.logits, HUGE_NEG)
        else:
            self.masks = None
            logits = self.logits

        # Reinitialize updated logits
        super().__init__(logits=logits)

    def entropy(self):
        if self.masks is None:
            return super().entropy()

        # Highly negative logits don't result in 0 probs, so we must replace
        # with 0s to ensure 0 contribution to the distribution's entropy, since
        # masked actions possess no uncertainty.
        device = self.logits.device
        p_log_p = self.logits * self.probs
        p_log_p = th.where(self.masks, p_log_p, th.tensor(0.0, device=device))
        return -p_log_p.sum(-1)


class CategoricalDistribution(Distribution):
    """
    Categorical distribution for discrete actions.

    :param action_dim: Number of discrete actions
    """

    def __init__(self, action_dim: int):
        super(CategoricalDistribution, self).__init__()
        self.distribution = MaskableCategorical()
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

    def proba_distribution(self, action_logits: th.Tensor) -> CategoricalDistribution:
        self.distribution = MaskableCategorical(logits=action_logits)
        return self

    def log_prob(self, actions: th.Tensor) -> th.Tensor:
        assert self.distribution is not None, "Must set distribution parameters"
        return self.distribution.log_prob(actions)

    def entropy(self) -> th.Tensor:
        assert self.distribution is not None, "Must set distribution parameters"
        return self.distribution.entropy()

    def sample(self) -> th.Tensor:
        assert self.distribution is not None, "Must set distribution parameters"
        return self.distribution.sample()

    def mode(self) -> th.Tensor:
        assert self.distribution is not None, "Must set distribution parameters"
        return th.argmax(self.distribution.probs, dim=1)

    def actions_from_params(
        self, action_logits: th.Tensor, deterministic: bool = False
    ) -> th.Tensor:
        # Update the proba distribution
        self.proba_distribution(action_logits)
        return self.get_actions(deterministic=deterministic)

    def log_prob_from_params(
        self, action_logits: th.Tensor
    ) -> Tuple[th.Tensor, th.Tensor]:
        actions = self.actions_from_params(action_logits)
        log_prob = self.log_prob(actions)
        return actions, log_prob
