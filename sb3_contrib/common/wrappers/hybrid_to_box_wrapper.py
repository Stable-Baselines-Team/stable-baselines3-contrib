from typing import Any, Dict, Tuple
import numpy as np
from gymnasium import spaces, Wrapper


class HybridToBoxWrapper(Wrapper):
    """
    Wrapper that converts a hybrid action space (Tuple of MultiDiscrete and Box)
    into a single Box action space, enabling compatibility with standard algorithms.
    The wrapper handles the conversion between the flattened Box action space and
    the hybrid action space internally.
    """

    def __init__(self, env):
        """
        Initialize the wrapper.

        :param env: The environment to wrap
        """
        super().__init__(env)
        assert isinstance(env.action_space, spaces.Tuple), "Environment must have a Tuple action space"
        assert len(env.action_space.spaces) == 2, "Action space must contain exactly 2 subspaces"
        assert isinstance(env.action_space.spaces[0], spaces.MultiDiscrete), "First subspace must be MultiDiscrete"
        assert isinstance(env.action_space.spaces[1], spaces.Box), "Second subspace must be Box"

        # Store original action space
        self.hybrid_action_space = env.action_space

        # Calculate total dimensions needed for Box space
        self.discrete_dims = sum(env.action_space.spaces[0].nvec)  # One-hot encoding for each discrete action
        self.continuous_dims = env.action_space.spaces[1].shape[0]

        # Create new Box action space
        # First part: one-hot encoding for discrete actions
        # Second part: continuous actions
        total_dims = self.discrete_dims + self.continuous_dims
        self.action_space = spaces.Box(
            low=np.concatenate([np.zeros(self.discrete_dims), env.action_space.spaces[1].low]),
            high=np.concatenate([np.ones(self.discrete_dims), env.action_space.spaces[1].high]),
            dtype=np.float32
        )

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        """
        Convert the Box action back to hybrid format and step the environment.

        :param action: Action from the Box action space
        :return: Next observation, reward, done flag, and info dictionary
        """
        # Split action into discrete and continuous parts
        discrete_part = action[:self.discrete_dims]
        continuous_part = action[self.discrete_dims:]

        # Convert one-hot encodings back to MultiDiscrete format
        discrete_actions = []
        start_idx = 0
        for n in self.hybrid_action_space.spaces[0].nvec:
            one_hot = discrete_part[start_idx:start_idx + n]
            discrete_actions.append(np.argmax(one_hot))
            start_idx += n

        # Create hybrid action tuple
        hybrid_action = (
            np.array(discrete_actions, dtype=np.int64),
            continuous_part
        )

        return self.env.step(hybrid_action)