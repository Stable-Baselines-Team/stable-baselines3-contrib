from typing import Callable, Union

import gym
import numpy as np


class ActionMasker(gym.Wrapper):
    """
    Env wrapper to provide attributes/methods required to support masking.

    Summary of the exposed attributes/methods:

    action_mask_fn: attribute exposing a function that takes an env and returns the action mask
    action_masks(): method that calls action_mask_fn with the wrapped env

    This wrapper is not needed if the env exposes the expected attributes/methods itself.

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

            self.action_mask_fn = found_method
        else:
            self.action_mask_fn = action_mask_fn

    def action_masks(self) -> np.ndarray:
        return self.action_mask_fn(self.env)
