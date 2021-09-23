from typing import List

import pytest
from gym.spaces import Discrete
from stable_baselines3.common.envs import IdentityEnv

from sb3_contrib.common.wrappers import ActionMasker


class IdentityEnvDiscrete(IdentityEnv):
    def __init__(self, dim: int = 1, ep_length: int = 100):
        """
        Identity environment for testing purposes

        :param dim: the size of the dimensions you want to learn
        :param ep_length: the length of each episode in timesteps
        """
        space = Discrete(dim)
        self.useless_property = 1
        super().__init__(ep_length=ep_length, space=space)

    def _action_masks(self) -> List[int]:
        return [i == self.state for i in range(self.action_space.n)]


def action_mask_fn(env: IdentityEnvDiscrete) -> List[int]:
    return [i == env.state for i in range(env.action_space.n)]


def test_wrapper_accepts_function():
    """
    ActionMasker accepts a function
    """

    env = IdentityEnvDiscrete()

    assert not hasattr(env, "action_masks")
    env = ActionMasker(env, action_mask_fn)
    assert hasattr(env, "action_masks")


# Wrapper accepts as string name of a method on the underlying env
def test_wrapper_accepts_attr_name():
    """
    ActionMasker accepts a string name of a method on the underlying env
    """

    env = IdentityEnvDiscrete()

    assert not hasattr(env, "action_masks")
    env = ActionMasker(env, "_action_masks")
    assert hasattr(env, "action_masks")


def test_attr_must_be_callable():
    """
    Passing ActionMasker the string name of a non-callable is an error
    """

    env = IdentityEnvDiscrete()

    with pytest.raises(ValueError):
        env = ActionMasker(env, "useless_property")


# Wrapper method returns expected results
def test_action_masks_returns_expected_result():
    """
    ActionMasker-provided action_masks() method returns expected results
    """

    env = IdentityEnvDiscrete()
    env = ActionMasker(env, action_mask_fn)

    # Only one valid action expected
    masks = env.action_masks()
    masks[env.state] = not masks[env.state]  # Bit-flip the one expected valid action
    assert all([not mask for mask in masks])
