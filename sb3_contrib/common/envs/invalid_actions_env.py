from typing import List, Optional

import numpy as np
from gym import spaces
from stable_baselines3.common.envs import IdentityEnv


class InvalidActionEnvDiscrete(IdentityEnv):
    """
    Identity env with a discrete action space. Supports action masking.
    """

    def __init__(
        self,
        dim: Optional[int] = None,
        ep_length: int = 100,
        n_invalid_actions: int = 0,
    ):
        if dim is None:
            dim = 1
        assert n_invalid_actions < dim, f"Too many invalid actions: {n_invalid_actions} < {dim}"

        space = spaces.Discrete(dim)
        self.n_invalid_actions = n_invalid_actions
        self.possible_actions = np.arange(space.n)
        self.invalid_actions: List[int] = []
        super().__init__(space=space, ep_length=ep_length)

    def _choose_next_state(self) -> None:
        self.state = self.action_space.sample()
        # Randomly choose invalid actions that are not the current state
        potential_invalid_actions = [i for i in self.possible_actions if i != self.state]
        self.invalid_actions = np.random.choice(potential_invalid_actions, self.n_invalid_actions, replace=False)

    def action_masks(self) -> List[bool]:
        return [action not in self.invalid_actions for action in self.possible_actions]


class InvalidActionEnvMultiDiscrete(IdentityEnv):
    """
    Identity env with a multidiscrete action space. Supports action masking.
    """

    def __init__(
        self,
        dims: Optional[List[int]] = None,
        ep_length: int = 100,
        n_invalid_actions: int = 0,
    ):
        if dims is None:
            dims = [1, 1]

        if n_invalid_actions > sum(dims) - len(dims):
            raise ValueError(f"Cannot find a valid action for each dim. Set n_invalid_actions <= {sum(dims) - len(dims)}")

        space = spaces.MultiDiscrete(dims)
        self.n_invalid_actions = n_invalid_actions
        self.possible_actions = np.arange(sum(dims))
        self.invalid_actions: List[int] = []
        super().__init__(space=space, ep_length=ep_length)

    def _choose_next_state(self) -> None:
        self.state = self.action_space.sample()

        converted_state: List[int] = []
        running_total = 0
        for i in range(len(self.action_space.nvec)):
            converted_state.append(running_total + self.state[i])
            running_total += self.action_space.nvec[i]

        # Randomly choose invalid actions that are not the current state
        potential_invalid_actions = [i for i in self.possible_actions if i not in converted_state]
        self.invalid_actions = np.random.choice(potential_invalid_actions, self.n_invalid_actions, replace=False)

    def action_masks(self) -> List[bool]:
        return [action not in self.invalid_actions for action in self.possible_actions]


class InvalidActionEnvMultiBinary(IdentityEnv):
    """
    Identity env with a multibinary action space. Supports action masking.
    """

    def __init__(
        self,
        dims: Optional[int] = None,
        ep_length: int = 100,
        n_invalid_actions: int = 0,
    ):
        if dims is None:
            dims = 1

        if n_invalid_actions > dims:
            raise ValueError(f"Cannot find a valid action for each dim. Set n_invalid_actions <= {dims}")

        space = spaces.MultiBinary(dims)
        self.n_invalid_actions = n_invalid_actions
        self.possible_actions = np.arange(2 * dims)
        self.invalid_actions: List[int] = []
        super().__init__(space=space, ep_length=ep_length)

    def _choose_next_state(self) -> None:
        self.state = self.action_space.sample()

        converted_state: List[int] = []
        running_total = 0
        for i in range(self.action_space.n):
            converted_state.append(running_total + self.state[i])
            running_total += 2

        # Randomly choose invalid actions that are not the current state
        potential_invalid_actions = [i for i in self.possible_actions if i not in converted_state]
        self.invalid_actions = np.random.choice(potential_invalid_actions, self.n_invalid_actions, replace=False)

    def action_masks(self) -> List[bool]:
        return [action not in self.invalid_actions for action in self.possible_actions]
