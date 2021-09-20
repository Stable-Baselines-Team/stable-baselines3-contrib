from typing import List, Optional

import numpy as np
# from gym import Env, Space
from gym.spaces import Discrete  # TODO: add env for MultiBinary, MultiDiscrete
from stable_baselines3.common.envs import IdentityEnv


class InvalidActionEnvDiscrete(IdentityEnv):
    def __init__(
        self,
        dim: Optional[int] = None,
        ep_length: int = 100,
        n_invalid_actions: int = 0,
    ):
        if dim is None:
            dim = 1
        assert n_invalid_actions < dim, f"Too many invalid actions: {n_invalid_actions} < {dim}"

        space = Discrete(dim)
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
