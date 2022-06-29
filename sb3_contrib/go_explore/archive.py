from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch
from gym import spaces
from stable_baselines3 import HerReplayBuffer
from stable_baselines3.common.type_aliases import DictReplayBufferSamples
from stable_baselines3.common.vec_env import VecEnv
from stable_baselines3.common.vec_env.vec_normalize import VecNormalize
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
    :param cell_factory: The cell factory
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
        cell_dim: int,
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

        self.cells = np.zeros((self.buffer_size, self.n_envs, cell_dim), dtype=np.float32)

    def add(
        self,
        obs: Dict[str, np.ndarray],
        next_obs: Dict[str, np.ndarray],
        action: np.ndarray,
        reward: np.ndarray,
        done: np.ndarray,
        infos: List[Dict[str, Any]],
        is_virtual: bool = False,
    ) -> None:
        self.cells[self.pos] = self.cell_factory(next_obs["observation"])
        super().add(obs, next_obs, action, reward, done, infos, is_virtual)

    def recompute_cells(self) -> None:
        """
        Recompute cells.
        """
        upper_bound = self.pos if not self.full else self.buffer_size
        nb_obs = upper_bound * self.n_envs

        if upper_bound == 0:
            return  # no trajectory yet

        # shape from (pos, env_idx, *cell_shape) to (idx, *cell_shape)
        flat_cells = self.cells[:upper_bound].reshape((nb_obs, -1))
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
        trajectory = [self.cells[start, env_idx]]
        for pos in range(start + 1, goal_pos + 1):
            previous_cell = self.cells[pos - 1, env_idx]
            cell = self.cells[pos, env_idx]
            if (previous_cell != cell).any():
                trajectory.append(cell)
        return np.array(trajectory)

    def _get_virtual_samples(
        self, batch_inds: np.ndarray, env_indices: np.ndarray, env: Optional[VecNormalize] = None
    ) -> DictReplayBufferSamples:
        """
        Get the samples corresponding to the batch and environment indices.

        :param batch_inds: Indices of the transitions
        :param env_indices: Indices of the envrionments
        :param env: associated gym VecEnv to normalize the observations/rewards when sampling, defaults to None
        :return: Samples
        """
        # Get infos and obs
        obs = {key: obs[batch_inds, env_indices, :] for key, obs in self.observations.items()}
        next_obs = {key: obs[batch_inds, env_indices, :] for key, obs in self.next_observations.items()}
        cell = self.cell_factory(obs["observation"])

        # Sample and set new goals
        new_goals = self._sample_goals(batch_inds, env_indices)
        obs["goal"] = new_goals
        # The goal for the next observation must be the same as the previous one. TODO: Why ?
        next_obs["goal"] = new_goals

        # Compute new reward
        is_success = (cell == new_goals).all(-1)
        rewards = is_success.astype(np.float32) - 1

        obs = self._normalize_obs(obs)
        next_obs = self._normalize_obs(next_obs)

        # Convert to torch tensor
        observations = {key: self.to_torch(obs) for key, obs in obs.items()}
        next_observations = {key: self.to_torch(obs) for key, obs in next_obs.items()}

        return DictReplayBufferSamples(
            observations=observations,
            actions=self.to_torch(self.actions[batch_inds, env_indices]),
            next_observations=next_observations,
            # Only use dones that are not due to timeouts
            # deactivated by default (timeouts is initialized as an array of False)
            dones=self.to_torch(self.dones[batch_inds, env_indices] * (1 - self.timeouts[batch_inds, env_indices])).reshape(
                -1, 1
            ),
            rewards=self.to_torch(self._normalize_reward(rewards.reshape(-1, 1), env)),
        )

    def _sample_goals(self, batch_inds: np.ndarray, env_indices: np.ndarray) -> np.ndarray:
        """
        Sample goals based on goal_selection_strategy.

        :param trans_coord: Coordinates of the transistions within the buffer
        :return: Return sampled goals
        """
        batch_ep_start = self.ep_start[batch_inds, env_indices]
        batch_ep_length = self.ep_length[batch_inds, env_indices]

        if self.goal_selection_strategy == GoalSelectionStrategy.FINAL:
            # replay with final state of current episode
            transition_indices_in_episode = batch_ep_length - 1

        elif self.goal_selection_strategy == GoalSelectionStrategy.FUTURE:
            # replay with random state which comes from the same episode and was observed after current transition
            current_indices_in_episode = batch_inds - batch_ep_start
            transition_indices_in_episode = np.random.randint(current_indices_in_episode, batch_ep_length)

        elif self.goal_selection_strategy == GoalSelectionStrategy.EPISODE:
            # replay with random state which comes from the same episode as current transition
            transition_indices_in_episode = np.random.randint(0, batch_ep_length)

        else:
            raise ValueError(f"Strategy {self.goal_selection_strategy} for sampling goals not supported!")

        transition_indices = (transition_indices_in_episode + batch_ep_start) % self.buffer_size
        return self.cells[transition_indices, env_indices]
