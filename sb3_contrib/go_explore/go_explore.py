import copy
from typing import Any, Dict, Mapping, Tuple, Type

import gym
import numpy as np
from gym import Env, spaces
from stable_baselines3 import SAC

from sb3_contrib.go_explore import ArchiveBuffer, CellFactory


class Goalify(gym.GoalEnv, gym.Wrapper):
    """
    Wrap the env into a GoalEnv.

    :param env: The environment
    :param cell_factory: The cell factory
    :param nb_random_exploration_steps: Number of random exploration steps after the goal is reached, defaults to 30
    :param window_size: Agent can skip goals in the goal trajectory within the limit of ``window_size``
        goals ahead, defaults to 10
    """

    def __init__(
        self,
        env: Env,
        cell_factory: CellFactory,
        nb_random_exploration_steps: int = 30,
        window_size: int = 10,
    ) -> None:
        super().__init__(env)
        # Set a goalconditionned observation space
        self.observation_space = spaces.Dict(
            {
                "observation": copy.deepcopy(self.env.observation_space),
                "desired_goal": copy.deepcopy(self.env.observation_space),
                "achieved_goal": copy.deepcopy(self.env.observation_space),
            }
        )
        self.archive = None  # type: ArchiveBuffer

        self.cell_factory = cell_factory
        self.nb_random_exploration_steps = nb_random_exploration_steps
        self.window_size = window_size

    def set_archive(self, archive: ArchiveBuffer) -> None:
        """
        Set the archive.

        :param archive: The archive
        """
        self.archive = archive

    def reset(self) -> Dict[str, np.ndarray]:
        obs = self.env.reset()
        assert self.archive is not None, "you need to set the archive before reset. Use set_archive()"
        self.goal_trajectory = self.archive.sample_goal_trajctory()
        self._goal_idx = 0
        self.done_countdown = self.nb_random_exploration_steps
        self._is_last_goal_reached = False  # useful flag
        obs = self._get_obs(obs)  # turn into dict
        return obs

    def _get_obs(self, obs: np.ndarray) -> Dict[str, np.ndarray]:
        return {
            "observation": obs.copy(),
            "achieved_goal": obs.copy(),
            "desired_goal": self.goal_trajectory[self._goal_idx].copy(),
        }

    def step(self, action: np.ndarray) -> Tuple[Dict[str, np.ndarray], float, bool, Dict[str, Any]]:
        wrapped_obs, reward, done, info = self.env.step(action)

        # Compute reward (has to be done before moving to next goal)
        desired_goal = self.goal_trajectory[self._goal_idx]
        reward = self.compute_reward(wrapped_obs, desired_goal, {})

        # Move to next goal here.
        self.maybe_move_to_next_goal(wrapped_obs)  # modify self._goal_idx

        # When the last goal is reached, delay the done to allow some random actions
        if self._is_last_goal_reached:
            if self.done_countdown != 0:
                info["use_random_action"] = True
                self.done_countdown -= 1
            else:  # self.done_countdown == 0:
                done = True

        obs = self._get_obs(wrapped_obs)
        return obs, reward, done, info

    def compute_reward(self, achieved_goal: object, desired_goal: object, info: Mapping[str, Any]) -> float:
        return self.is_success(achieved_goal, desired_goal) - 1

    def is_success(self, obs: np.ndarray, goal: np.ndarray) -> bool:
        """
        Return True when the obs and the goal are in the same cell.

        :param obs: The observation
        :param goal: The goal
        :return: Succes or not
        """
        return self.cell_factory(obs) == self.cell_factory(goal)

    def maybe_move_to_next_goal(self, obs: np.ndarray) -> None:
        """
        Set the next goal idx if necessary.

        From the paper:
        "When a cell that was reached occurs multiple times in the window, the next goal
        is the one that follows the last occurence of this repeated goal cell."

        :param obs: The observation
        """
        upper_idx = min(self._goal_idx + self.window_size, len(self.goal_trajectory))
        for goal_idx in range(self._goal_idx, upper_idx):
            goal = self.goal_trajectory[goal_idx]
            if self.is_success(obs, goal):
                self._goal_idx = goal_idx + 1
        # Update the flag _is_last_goal_reached
        if self._goal_idx == len(self.goal_trajectory):
            self._is_last_goal_reached = True
            self._goal_idx -= 1


class GoExplore:
    """
    Go-Explore implementation as described in [1].

    This is a simplified version, which does not include a number of tricks
    whose impact on performance we do not know. The goal is to implement the
    general principle of the algorithm and not all the little tricks.
    In particular, we do not implement:
    - everything related with domain knowledge,
    - action repetition
    - dynamic representation
    - self-imitation learning
    - parallelized exploration phase
    """

    def __init__(
        self,
        env: Env,
        cell_factory: CellFactory,
        count_pow: float = 1,
    ):
        env = Goalify(env, cell_factory)
        self.model = SAC(
            "MultiInputPolicy",
            env,
            replay_buffer_class=ArchiveBuffer,
            replay_buffer_kwargs={"cell_factory": cell_factory, "count_pow": count_pow},
        )


if __name__ == "__main__":
    import gym_continuous_maze

    from sb3_contrib.go_explore.archive import DownscaleCellFactory

    env = gym.make("ContinuousMaze-v0")
    cell_factory = DownscaleCellFactory(1, 1)

    algo = GoExplore(env, cell_factory)

    # archive = ArchiveBuffer(100, env.observation_space, env.action_space, cell_factory)
    # env = Goalify(env, archive, cell_factory)
    # # env.set_archive(archive)
    # env.reset()
    # done = False
    # while not done:
    #     action = np.array([-0.1, 0.0])
    #     obs, reward, done, info = env.step(action)
    #     print(obs, reward)
