from typing import Generator, Optional

import numpy as np
import torch as th
from gym import spaces
from stable_baselines3.common.buffers import RolloutBuffer
from stable_baselines3.common.type_aliases import RolloutBufferSamples
from stable_baselines3.common.vec_env import VecNormalize


class MaskableRolloutBufferSamples(RolloutBufferSamples):
    action_masks: th.Tensor


class MaskableRolloutBuffer(RolloutBuffer):
    def __init__(self, *args, **kwargs):
        self.action_masks = None
        super().__init__(*args, **kwargs)

    def reset(self) -> None:
        # TODO: need to handle different action_space attrs based on shape
        action_masks = np.array([], dtype=np.float32)
        if isinstance(self.action_space, spaces.Discrete):
            action_masks = np.ones(
                (self.buffer_size, self.n_envs, self.action_space.n), dtype=np.float32
            )
        elif isinstance(self.action_space, spaces.MultiDiscrete):
            action_masks = np.ones(
                (self.buffer_size, self.n_envs, sum(self.action_space.nvec)),
                dtype=np.float32,
            )
        self.action_masks = action_masks

        super().reset()

    def add(self, *args, action_masks: np.ndarray = None, **kwargs) -> None:
        """
        :param action_masks: Masks applied to constrain the choice of possible actions.
        """
        if action_masks is not None:
            self.action_masks[self.pos] = np.array(action_masks).copy()

        super().add(*args, **kwargs)

    def get(
        self, batch_size: Optional[int] = None
    ) -> Generator[MaskableRolloutBufferSamples, None, None]:
        assert self.full, ""
        indices = np.random.permutation(self.buffer_size * self.n_envs)
        # Prepare the data
        if not self.generator_ready:
            for tensor in [
                "observations",
                "actions",
                "action_masks",
                "values",
                "log_probs",
                "advantages",
                "returns",
            ]:
                self.__dict__[tensor] = self.swap_and_flatten(self.__dict__[tensor])
            self.generator_ready = True

        # Return everything, don't create minibatches
        if batch_size is None:
            batch_size = self.buffer_size * self.n_envs

        start_idx = 0
        while start_idx < self.buffer_size * self.n_envs:
            yield self._get_samples(indices[start_idx : start_idx + batch_size])
            start_idx += batch_size

    def _get_samples(
        self, batch_inds: np.ndarray, env: Optional[VecNormalize] = None
    ) -> MaskableRolloutBufferSamples:
        data = (
            self.observations[batch_inds],
            self.actions[batch_inds],
            self.action_masks[batch_inds].flatten(),
            self.values[batch_inds].flatten(),
            self.log_probs[batch_inds].flatten(),
            self.advantages[batch_inds].flatten(),
            self.returns[batch_inds].flatten(),
        )
        return MaskableRolloutBufferSamples(*tuple(map(self.to_torch, data)))
