import random
from abc import ABC, abstractmethod
from logging import warning
from typing import Any, Dict, List, Union

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
        :return: The processed observations.
        """
        raise NotImplementedError()

    def __call__(self, observation: np.ndarray, dim: int = 0) -> Union[Cell, List[Cell]]:
        if dim > 0:
            return [self(obs, dim - 1) for obs in observation]
        elif dim == 0:
            processed_obs = self._process(observation)
            return Cell(processed_obs)
        else:
            raise ValueError("dim must be >0")


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
    ReplayBuffer that keep track of cells.

    :param buffer_size: Max number of element in the buffer
    :param observation_space: Observation space
    :param action_space: Action space
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

        self.cell_factory = cell_factory
        self.count_pow = count_pow

        self._cells = []  # type: List[Cell]
        self._cell_to_idx = {}  # type: Dict[Cell, int] # much faster than using self._cells.index(cell)

        # A dict mapping cell to a list of every encountered observation in that cell
        self._cell_to_obss = {}  # type: Dict[Cell, List[np.ndarray]]
        self._counts = np.zeros(shape=(0,))  # A counter of the number of visits per cell

        self.trajectory_manager = TrajectoryManager()
        self._running_trajectory = []  # type: List[Cell]

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
        self._store_cell(next_obs["achieved_goal"])
        self._store_trajectory(next_obs["achieved_goal"], done)

    def _store_cell(self, obs: np.ndarray) -> None:
        cell = self.cell_factory(obs)
        if cell not in self._cells:
            self._new_cell_found(cell)
        idx = self._cell_to_idx[cell]
        # update counts and obs list
        self._counts[idx] += 1
        self._cell_to_obss[cell].append(obs)

    def _new_cell_found(self, cell: Cell) -> None:
        """
        Call this when you have found a new cell.

        It expands the csgraph, counts, cell_to_obss, cell_to_idx and idx_to_cell
        """
        idx = len(self._cells)
        self._cells.append(cell)
        self._cell_to_idx[cell] = idx
        self._cell_to_obss[cell] = []
        self._counts = np.pad(self._counts, (0, 1), constant_values=0)

    def _store_trajectory(self, obs: np.ndarray, done: bool) -> None:
        self._running_trajectory.append(self.cell_factory(obs))
        if done:
            self.trajectory_manager.add_trajectory(self._running_trajectory)
        self._running_trajectory = []

    def _cell_to_obs(self, cell: Cell) -> np.ndarray:
        """
        Randomly choose an observation among the observations of the given cell.

        :param cell: The cell
        """
        obs = random.choice(self._cell_to_obss[cell])
        return obs

    def sample_goal_trajctory(self) -> List[np.ndarray]:
        """
        Sample a trajectory of goals, as a list f goal observations.

        :return: A list of observations
        """
        nb_cells = len(self._cells)
        if nb_cells == 0:
            return [self.observation_space["desired_goal"].sample()]
        weights = 1 / (self._counts**self.count_pow)
        p = weights / weights.sum()
        # randomly choose a final cell

        goal_cell_idx = np.random.choice(nb_cells, p=p)
        goal_cell = self._cells[goal_cell_idx]
        cell_trajectory = self.trajectory_manager.shortest_trajectory(goal_cell)
        obs_trajectory = [self._cell_to_obs(cell) for cell in cell_trajectory]
        return obs_trajectory


Trajectory = List[Cell]


class TrajectoryManager:
    """
    Manage trajectrories.

    Strong assumption: every episode starts in the same state.

    To add a trajectory:
    >>> trajectory_manager.add_trajectory(trajectory)

    To get the shortest trajectory toward a cell:
    >>> trajectory = trajectory_manager.shortest_trajectory(cell)
    """

    def __init__(self) -> None:
        self.trajectories = []  # type: List[Trajectory]
        self.cell_to_trajectory_idx = {}  # type: Dict[Cell, int] # Maps goal cell to shortest trajectory

    def add_trajectory(self, trajectory: Trajectory) -> None:
        """
        Add a trajecotry.
        """
        trajectory = self._colapse(trajectory)
        self.trajectories.append(trajectory)
        self._process(trajectory, trajectory_idx=len(self.trajectories) - 1)

    def _colapse(self, trajectory: Trajectory) -> Trajectory:
        """
        Remove repetition. Inplace computing.

        Examples: (with integers, but the principles remains the same)
        >>> colapse([1, 2, 2, 3, 3])
        [1, 2, 3]
        >>> colapse([1, 2, 2, 1, 2, 3, 3, 3])
        [1, 2, 1, 2, 3]

        :param trajectory: Trajectory as a list of cells
        :return:The colapsed trajectory
        """
        k = 0
        while k < len(trajectory) - 1:
            if trajectory[k] == trajectory[k + 1]:
                trajectory.pop(k)
            else:
                k += 1
        return trajectory

    def _process(self, trajectory: Trajectory, trajectory_idx: int):
        for k in range(1, len(trajectory)):
            cell = trajectory[k]
            is_known = cell in self.cell_to_trajectory_idx
            if is_known:
                lenght_shortest_trajectory = len(self.shortest_trajectory(cell))
            if not is_known or k < lenght_shortest_trajectory:
                self.cell_to_trajectory_idx[cell] = trajectory_idx

    def shortest_trajectory(self, cell: Cell) -> Trajectory:
        trajectory_idx = self.cell_to_trajectory_idx[cell]
        trajectory = self.trajectories[trajectory_idx]
        cell_idx = trajectory.index(cell)
        return trajectory[:cell_idx]


if __name__ == "__main__":
    import gym
    import gym_continuous_maze
    from stable_baselines3.common.vec_env import DummyVecEnv

    from sb3_contrib.go_explore.go_explore import Goalify

    # Parallel environments

    env = gym.make("ContinuousMaze-v0")
    cell_factory = DownscaleCellFactory(1, 1)
    env = Goalify(env, cell_factory)
    archive = ArchiveBuffer(1000, env.observation_space, env.action_space, env, cell_factory)
    env.set_archive(archive)
    obs = env.reset()
    env = DummyVecEnv([lambda: env])
    done = False
    while not done:
        action = env.action_space.sample()
        next_obs, reward, done, info = env.step(action)
        archive.add(obs, next_obs, action, reward, done, info)
        obs = next_obs

    ## ERROR BECAUSE RESET IS DONE BEFORE RETURNING IN DUMMYVEENV
