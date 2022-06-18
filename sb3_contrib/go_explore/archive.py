from typing import Union

import numpy as np
import torch
from gym import spaces
from stable_baselines3 import HerReplayBuffer
from stable_baselines3.common.vec_env import VecEnv
from stable_baselines3.her.goal_selection_strategy import GoalSelectionStrategy

from sb3_contrib.go_explore.cells import CellFactory


def multinomial(weights: np.ndarray) -> int:
    p = weights / weights.sum()
    r = np.random.multinomial(1, p, size=1)
    idx = np.nonzero(r)[1][0]
    return idx


class ArchiveBuffer(HerReplayBuffer):
    """
    Archive buffer.

    :param buffer_size: Max number of element in the buffer
    :param observation_space: Observation space
    :param action_space: Action space
    :param env: The training environment
    :param inverse_model: Inverse model used to compute embeddings
    :param distance_threshold: The goal is reached when the distance between the current embedding
        and the goal embedding is under this threshold
    :param device: PyTorch device
    :param n_envs: Number of parallel environments
    :param n_sampled_goal: Number of virtual transitions to create per real transition,
        by sampling new goals.
    :param goal_selection_strategy: Strategy for sampling goals for replay.
        One of ['episode', 'final', 'future']
    """

    def __init__(
        self,
        buffer_size: int,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        env: VecEnv,
        cell_factory: CellFactory,
        device: Union[torch.device, str] = "cpu",
        n_envs: int = 1,
        optimize_memory_usage: bool = False,
        handle_timeout_termination: bool = True,
        n_sampled_goal: int = np.inf,
        goal_selection_strategy: Union[GoalSelectionStrategy, str] = "future",
    ) -> None:
        super().__init__(
            buffer_size,
            observation_space,
            action_space,
            env,
            device=device,
            n_envs=n_envs,
            n_sampled_goal=n_sampled_goal,
            goal_selection_strategy=goal_selection_strategy,
        )

        # For cell management
        self.cell_factory = cell_factory
        self.weights = None

    def recompute_cells(self) -> None:
        """
        Recompute cells.
        """
        upper_bound = self.pos if not self.full else self.buffer_size
        nb_obs = upper_bound * self.n_envs

        if upper_bound == 0:
            return  # no trajectory yet

        self.cells = self.cell_factory(self.next_observations["observation"][:upper_bound])
        flat_cells = self.cells.reshape((nb_obs, -1))  # shape from (pos, env_idx, *cell_shape) to (idx, *cell_shape)
        # Compute the unique cells.
        # cells_uid is a tensor of shape (nb_obs,) mapping observation index to its cell index.
        # unique_cells is a tensor of shape (nb_cells, *cell_shape) mapping cell index to the cell.
        unique_cells, cells_uid, counts = np.unique(flat_cells, return_inverse=True, return_counts=True, axis=0)
        nb_cells = unique_cells.shape[0]  # number of unique cells
        flat_pos = np.arange(upper_bound).repeat(self.n_envs)  # [0, 0, 1, 1, 2, ...] if n_envs == 2
        flat_ep_start = self.ep_start[:upper_bound].flatten()  # shape from (pos, env_idx) to (idx,)
        flat_timestep = flat_pos - flat_ep_start  # timestep within the episode

        earliest_cell_occurence = np.zeros(nb_cells, dtype=np.int64)
        for cell_uid in range(nb_cells):
            cell_idxs = np.where(cell_uid == cells_uid)[0]  # index of observations that are in the cell
            all_cell_occurences_timestep = flat_timestep[cell_idxs]  # the cell has been visited after all these timesteps
            earliest = np.argmin(all_cell_occurences_timestep)  # focus on the time when the cell is visited the earliest
            earliest_cell_occurence[cell_uid] = cell_idxs[earliest]

        self.weights = 1 / np.sqrt(counts + 1)
        self.earliest_cell_env = earliest_cell_occurence % self.n_envs
        self.earliest_cell_pos = earliest_cell_occurence // self.n_envs

    def sample_trajectory(self) -> np.ndarray:
        """
        Sample a trajcetory of observations based on the cells counts and trajectories.
        A goal cell is sampled with weight 1/count**count_pow. Then the shortest
        trajectory to the cell is computed and returned.
        :return: A list of observations as array
        """
        if self.weights is None:  # no cells yet
            goal = self.observation_space["goal"].sample()
            return [goal]

        cell_uid = multinomial(self.weights)
        # Get the env_idx, the pos in the buffer and the position of the start of the trajectory
        env_idx = self.earliest_cell_env[cell_uid]
        goal_pos = self.earliest_cell_pos[cell_uid]
        start = self.ep_start[goal_pos, env_idx]
        # Loop to avoid consecutive repetition
        trajectory = [self.next_observations["observation"][start, env_idx]]
        for pos in range(start + 1, goal_pos + 1):
            previous_cell = self.cells[pos - 1, env_idx]
            cell = self.cells[pos, env_idx]
            if (previous_cell != cell).any():
                obs = self.next_observations["observation"][pos, env_idx]
                trajectory.append(obs)
        return np.array(trajectory)
