from typing import Callable, Union

import gym
import numpy as np


class ActionMasker(gym.Wrapper):
    """
    Env wrapper providing the method required to support masking.

    Exposes a method called action_masks(), which returns masks for the wrapped env.
    This wrapper is not needed if the env exposes the expected method itself.

    :param env: the Gym environment to wrap
    :param action_mask_fn: A function that takes a Gym environment and returns an action mask,
        or the name of such a method provided by the environment.
    """

    def __init__(self, env: gym.Env, action_mask_fn: Union[str, Callable[[gym.Env], np.ndarray]]):
        super().__init__(env)

        if isinstance(action_mask_fn, str):
            found_method = getattr(self.env, action_mask_fn)
            if not callable(found_method):
                raise ValueError(f"Environment attribute {action_mask_fn} is not a method")

            self._action_mask_fn = found_method
        else:
            self._action_mask_fn = action_mask_fn

    def action_masks(self) -> np.ndarray:
        return self._action_mask_fn(self.env)
