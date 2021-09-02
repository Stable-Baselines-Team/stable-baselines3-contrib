from typing import List, Optional, Union

import numpy as np
from gym import Env, Space
from gym.spaces import Discrete  # TODO: add env for MultiBinary, MultiDiscrete
from stable_baselines3.common.type_aliases import GymObs, GymStepReturn


class InvalidActionEnv(Env):
    """
    Identity environment with invalid actions for testing purposes

    :param dim: the size of the action and observation dimension you want
        to learn. Provide at most one of ``dim`` and ``space``. If both are
        None, then initialization proceeds with ``dim=1`` and ``space=None``.
    :param space: the action and observation space. Provide at most one of
        ``dim`` and ``space``.
    :param ep_length: the length of each episode in timesteps
    :param n_invalid_actions: Number of invalid action that will be masked.
        Must be smaller than `dim - 1`
    """

    def __init__(
        self,
        dim: Optional[int] = None,
        space: Optional[Space] = None,
        ep_length: int = 100,
        n_invalid_actions: int = 0,
    ):
        if space is None:
            if dim is None:
                dim = 1
            space = Discrete(dim)
        else:
            assert dim is None, "arguments for both 'dim' and 'space' provided: at most one allowed"

        assert n_invalid_actions <= dim - 1, f"Too many invalid actions: {n_invalid_actions} <= {dim - 1}"

        self.invalid_actions = []
        self.possble_actions = np.arange(dim)
        self.dim = dim
        self.n_invalid_actions = n_invalid_actions
        self.action_space = self.observation_space = space
        self.ep_length = ep_length
        self.current_step = 0
        self.num_resets = -1  # Becomes 0 after __init__ exits.
        self.reset()

    def seed(self, seed: int) -> None:
        self.action_space.seed(seed)
        np.random.seed(seed)

    def reset(self) -> GymObs:
        self.current_step = 0
        self.num_resets += 1
        self._choose_next_state()
        return self.state

    def step(self, action: Union[int, np.ndarray]) -> GymStepReturn:
        reward = self._get_reward(action)
        self._choose_next_state()
        self.current_step += 1
        done = self.current_step >= self.ep_length
        return self.state, reward, done, {}

    def _choose_next_state(self) -> None:
        self.state = self.action_space.sample()
        # Randomly choose invalid actions that are not the current state
        potential_invalid_actions = [i for i in range(self.dim) if i != self.state]
        self.invalid_actions = np.random.choice(potential_invalid_actions, self.n_invalid_actions, replace=False)

    def action_masks(self) -> List[bool]:
        return [action not in self.invalid_actions for action in self.possble_actions]

    def _get_reward(self, action: Union[int, np.ndarray]) -> float:
        return 1.0 if np.all(self.state == action) else 0.0

    def render(self, mode: str = "human") -> None:
        pass
