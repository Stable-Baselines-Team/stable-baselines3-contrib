import random
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch as th
from gym import Env, spaces
from stable_baselines3 import HerReplayBuffer
from stable_baselines3.her.goal_selection_strategy import GoalSelectionStrategy


class Cell:
    """
    Cells is used as dict key, thus it must be hashable.

    :param arr: array
    """

    def __init__(self, arr: np.ndarray) -> None:
        self.arr = arr.astype(np.float64) + np.zeros_like(arr)  # avoid -0.
        self._hash = hash(self.arr.tobytes())

    def __hash__(self) -> int:
        return self._hash

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, Cell):
            return False
        return hash(self) == hash(other)

    def __neq__(self, other: Any) -> bool:
        return not self == other

    def __repr__(self) -> str:
        return "Cell({})".format(str(self.arr))


class CellFactory(ABC):
    """
    Abstract class used to compute the cell of an observation.

    To define a new cell factory, you only need to inherit this class with your `_process` method.
    """

    @abstractmethod
    def _process(self, observations: np.ndarray) -> np.ndarray:
        """
        Process the observation.

        :param observations: The observations to be processed. Each row is one observation
        :return: The processed observations
        """
        raise NotImplementedError()

    def __call__(self, observation: np.ndarray, dim: Optional[int] = None) -> Union[Cell, List[Cell]]:
        """
        Compute the observation into its cell.

        :param observation: The observation
        :param dim: When more than one observation is computed, set dim to 1, defaults to 0
        :return: The cell, or the list of cells
        """
        if dim is None:
            processed_obs = self._process(observation)
            return Cell(processed_obs)
        if dim == 0:
            return [self(obs) for obs in observation]
        else:
            raise ValueError("dim must be > 0")


class DownscaleCellFactory(CellFactory):
    """
    Reduce and downscale the observation: floor(coef*x/std).

    :param std: The observation standard deviation
    :param coef: The multiplication coeficient applied before computing floor. The higher, the more cells
    """

    def __init__(self, std: np.ndarray, coef: np.ndarray) -> None:
        super().__init__()
        self.std = std
        self.coef = coef

    def _process(self, observations: np.ndarray) -> np.ndarray:
        downscaled_obs = np.floor(self.coef * observations / self.std)
        return downscaled_obs


class ArchiveBuffer(HerReplayBuffer):
    """
    ReplayBuffer that keep track of cells and cell trajectories.

    :param buffer_size: Max number of element in the buffer
    :param observation_space: Observation space
    :param action_space: Action space
    :param env: The envrionment
    :param cell_factory: The cell factory
    :param count_pow: The cell weight is 1 / count**count_pow
    :param device:
    :param n_envs: Number of parallel environments
    :param optimize_memory_usage: Enable a memory efficient variant
        of the replay buffer which reduces by almost a factor two the memory used,
        at a cost of more complexity.
        See https://github.com/DLR-RM/stable-baselines3/issues/37#issuecomment-637501195
        and https://github.com/DLR-RM/stable-baselines3/pull/28#issuecomment-637559274
    :param handle_timeout_termination: Handle timeout termination (due to timelimit)
        separately and treat the task as infinite horizon task.
        https://github.com/DLR-RM/stable-baselines3/issues/284
    """

    def __init__(
        self,
        buffer_size: int,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        env: Env,
        cell_factory: CellFactory,
        count_pow: float = 0,
        device: Union[th.device, str] = "cpu",
        n_envs: int = 1,
        optimize_memory_usage: bool = False,
        handle_timeout_termination: bool = True,
        n_sampled_goal: int = 4,
        goal_selection_strategy: Union[GoalSelectionStrategy, str] = "future",
        online_sampling: bool = True,
    ) -> None:
        super().__init__(
            buffer_size,
            observation_space,
            action_space,
            env,
            device=device,
            n_envs=n_envs,
            optimize_memory_usage=optimize_memory_usage,
            handle_timeout_termination=handle_timeout_termination,
            n_sampled_goal=n_sampled_goal,
            goal_selection_strategy=goal_selection_strategy,
            online_sampling=online_sampling,
        )
        assert n_envs == 1, "The trajectory manager is not compatible with multiprocessing"
        self.count_pow = count_pow
        self.trajectory_manager = TrajectoryManager(cell_factory)

    def add(
        self,
        obs: np.ndarray,
        next_obs: np.ndarray,
        action: np.ndarray,
        reward: np.ndarray,
        done: np.ndarray,
        infos: List[Dict[str, Any]],
    ) -> None:
        """
        Add an element to the buffer.

        :param obs: the current observation
        :param next_obs: the next observation
        :param action: the action
        :param reward: the reward
        :param done: whether the env is done
        :param infos: infos
        :param episode_start: whether the episode starts, defaults to False
        """
        super().add(obs, next_obs, action, reward, done, infos)
        self.trajectory_manager.add(next_obs["achieved_goal"][0])
        if done[0]:
            self.trajectory_manager.start_new_trajectory()

    def sample_goal_trajectory(self) -> List[np.ndarray]:
        """
        Sample a trajectory of goals.

        :param count_pow: Weight of a cell to be sampled is 1/count**count_pow
        :return: The trajectory of goals
        """
        if self.trajectory_manager.nb_cells == 0:  # Edge case, when reset before any interaction
            return [self.observation_space["desired_goal"].sample()]
        return self.trajectory_manager.sample_goal_trajectory(self.count_pow)


Trajectory = List[Cell]


class TrajectoryManager:
    """
    Manage trajectrories.
    Main difference is that storing works cell by cell.

    Strong assumption: every episode starts in the same state.

    To add a obs:
    >>> trajectory_manager.add(obs)

    When a new trajectory starts:
    >>> trajectory_manager.start_new_trajectory()

    To sample a goal trajectory:
    >>> trajectory = trajectory_manager.sample_goal_trajectory(count_pow)
    """

    def __init__(self, cell_factory: CellFactory) -> None:
        self.cell_factory = cell_factory

        # Maps cell to a list of encountered observation in that cell
        self._cell_to_obss = {}  # type: Dict[Cell, List[np.ndarray]]

        self.trajectories = []  # type: List[Trajectory]
        self._cell_to_trajectory_idx = {}  # type: Dict[Cell, int] # Maps goal cell to shortest trajectory

        self.start_new_trajectory()

    @property
    def nb_cells(self):
        all_cells = self._cell_to_trajectory_idx.keys()
        return len(all_cells)

    def add(self, obs: np.ndarray) -> None:
        """
        Add a cell to the running trajectory.

        :param obs: The observation
        """
        # Add cell if needed
        cell = self.cell_factory(obs)
        if cell not in self._cell_to_obss.keys():
            self._cell_to_obss[cell] = []
        # Store the observation
        self._cell_to_obss[cell].append(obs)

        # If the agent is still on the last cell, do nothing (avoid cell repetition)
        if len(self._running_trajectory) > 0 and cell == self._running_trajectory[-1]:
            return
        self._running_trajectory.append(cell)
        self._process_last_cell()

    def start_new_trajectory(self) -> None:
        """
        Call this method when a new trajectory starts.
        """
        self.trajectories.append([])
        self._running_trajectory = self.trajectories[-1]

    def _process_last_cell(self) -> None:
        """
        Get the last cell encountered and determine if we just found a shorter trajectory to it.
        If so, we store the ``cell: trajectory_idx`` mapping in ``self._cell_to_trajectory_idx``.
        """
        # We process the last cell encoutered, so the last cell of the last stored trajectory
        cell_idx = len(self._running_trajectory) - 1
        trajectory_idx = len(self.trajectories) - 1
        # Get the cell
        cell = self._running_trajectory[cell_idx]
        # Determine if the cell has already been encountered
        is_known = cell in self._cell_to_trajectory_idx
        # If the cell has already been encountered, get the
        # number of intermediate cells required to reach it
        if is_known:
            lenght_shortest_trajectory = len(self._shortest_trajectory(cell))
        # If the cell is new, or if we've just find a shortest cell path to reach it,
        # store this path.
        if not is_known or cell_idx < lenght_shortest_trajectory:
            self._cell_to_trajectory_idx[cell] = trajectory_idx

    def sample_goal_trajectory(self, count_pow: float) -> List[np.ndarray]:
        """
        Sample a trajectory of goals.

        :param count_pow: Weight of a cell to be sampled is 1/count**count_pow
        :return: The trajectory of goals
        """
        all_cells = list(self._cell_to_trajectory_idx.keys())
        nb_cells = len(all_cells)
        counts = np.array([len(obs_list) for obs_list in self._cell_to_obss.values()])
        weights = 1 / (counts ** float(count_pow))
        p = weights / weights.sum()
        # randomly choose a goal cell
        goal_cell_idx = np.random.choice(nb_cells, p=p)
        goal_cell = all_cells[goal_cell_idx]
        cell_trajectory = self._shortest_trajectory(goal_cell)
        obs_trajectory = [self._sample_obs_from_cell(cell) for cell in cell_trajectory]
        return obs_trajectory

    def _shortest_trajectory(self, cell: Cell) -> Trajectory:
        """
        Returns the shortest trajectory known toward the given cell.

        :param cell: The goal cell
        :return: The shortest known trajectory
        """
        trajectory_idx = self._cell_to_trajectory_idx[cell]
        trajectory = self.trajectories[trajectory_idx]
        cell_idx = trajectory.index(cell)
        return trajectory[: cell_idx + 1]

    def _sample_obs_from_cell(self, cell: Cell) -> np.ndarray:
        """
        Randomly choose an observation among the observations of the given cell.

        :param cell: The cell
        """
        obs = random.choice(self._cell_to_obss[cell])
        return obs
