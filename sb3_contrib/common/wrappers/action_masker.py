from typing import Callable, Union

import gym
import numpy as np


class ActionMasker(gym.Wrapper):
    """
    Vectorized environment base class

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

            self.action_action_mask_fn = found_method
        else:
            self.action_mask_fn = action_mask_fn

    def valid_actions(self) -> np.ndarray:
        return self.action_mask_fn(self.env)
