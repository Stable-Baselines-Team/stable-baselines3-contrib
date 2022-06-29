import copy
from typing import Any, Callable, Dict, Optional, Tuple, Type

import gym
import numpy as np
from gym import Env, spaces
from stable_baselines3.common.base_class import maybe_make_env
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.off_policy_algorithm import OffPolicyAlgorithm

from sb3_contrib.go_explore.archive import ArchiveBuffer
from sb3_contrib.go_explore.cells import CellFactory
from sb3_contrib.go_explore.feature_extractor import GoExploreExtractor


class Goalify(gym.Wrapper):
    """
    Wrap the env into a GoalEnv.

    :param env: The environment
    :param nb_random_exploration_steps: Number of random exploration steps after the goal is reached, defaults to 30
    :param window_size: Agent can skip goals in the goal trajectory within the limit of ``window_size``
        goals ahead, defaults to 10
    """

    def __init__(
        self, env: Env, cell_factory: CellFactory, nb_random_exploration_steps: int = 30, window_size: int = 10
    ) -> None:
        super().__init__(env)
        # Set a goal-conditionned observation space
        self.observation_space = spaces.Dict(
            {
                "observation": copy.deepcopy(self.env.observation_space),
                "goal": copy.deepcopy(self.env.observation_space),
            }
        )
        self.archive = None  # type: ArchiveBuffer
        self.cell_factory = cell_factory
        self.nb_random_exploration_steps = nb_random_exploration_steps
        self.window_size = window_size

    def set_archive(self, archive: ArchiveBuffer) -> None:
        """
        Set the archive.

        The archive is used to compute goal trajectories, and to compute the cell for the reward.

        :param archive: The archive
        """
        self.archive = archive

    def reset(self) -> Dict[str, np.ndarray]:
        obs = self.env.reset()
        assert self.archive is not None, "You need to set the archive before reset. Use set_archive()"
        self.goal_cell_trajectory = self.archive.sample_trajectory()
        self._goal_idx = 0
        self.done_countdown = self.nb_random_exploration_steps
        self._is_last_goal_reached = False  # useful flag
        dict_obs = self._get_dict_obs(obs)  # turn into dict
        return dict_obs

    def _get_dict_obs(self, obs: np.ndarray) -> Dict[str, np.ndarray]:
        return {
            "observation": obs.astype(np.float32),
            "goal": self.goal_cell_trajectory[self._goal_idx].astype(np.float32),
        }

    def step(self, action: np.ndarray) -> Tuple[Dict[str, np.ndarray], float, bool, Dict[str, Any]]:
        obs, reward, done, info = self.env.step(action)
        obs = obs.astype(np.float32)
        current_cell = self.cell_factory(obs)
        # Move to next goal here (by modifying self._goal_idx and self._is_last_goal_reached)
        upper_idx = min(self._goal_idx + self.window_size, len(self.goal_cell_trajectory))
        current_cell = self.cell_factory(obs)
        future_goal_cells = self.goal_cell_trajectory[self._goal_idx : upper_idx]

        future_successes = (current_cell == future_goal_cells).all(-1)
        if future_successes.any():
            furthest_futur_success = np.where(future_successes)[0].max()
            self._goal_idx += furthest_futur_success + 1
        if self._goal_idx == len(self.goal_cell_trajectory):
            self._is_last_goal_reached = True
            self._goal_idx -= 1

        if future_successes[0]:
            # Agent has just reached the current goal
            reward = 0
        else:
            # Agent has reached another goal, or no goal at all
            reward = -1

        # When the last goal is reached, delay the done to allow some random actions
        if self._is_last_goal_reached:
            info["is_success"] = True
            if self.done_countdown != 0:
                info["action_repeat"] = action
                self.done_countdown -= 1
            else:  # self.done_countdown == 0:
                done = True
        else:
            info["is_success"] = False

        dict_obs = self._get_dict_obs(obs)
        return dict_obs, reward, done, info


class CallEveryNTimesteps(BaseCallback):
    """
    Callback that calls a function every ``call_freq`` timesteps.

    :param func: The function to call
    :param call_freq: The call timestep frequency, defaults to 1
    :param verbose: Verbosity level 0: not output 1: info 2: debug, defaults to 0
    """

    def __init__(self, func: Callable[[], None], call_freq: int = 1, verbose=0) -> None:
        super(CallEveryNTimesteps, self).__init__(verbose)
        self.func = func
        self.call_freq = call_freq

    def _on_step(self) -> bool:
        """
        This method will be called by the model after each call to `env.step()`.

        For child callback (of an `EventCallback`), this will be called
        when the event is triggered.

        :return: (bool) If the callback returns False, training is aborted early.
        """
        if self.num_timesteps % self.call_freq == 0:
            self.func()

        return True


class GoExplore:
    """
    This is a simplified version of Go-Explore from the original paper, which does not include
    a number of tricks which impacts the performance.
    The goal is to implement the general principle of the algorithm and not all the little tricks.
    In particular, we do not implement:
    - everything related with domain knowledge,
    - self-imitation learning,
    - parallelized exploration phase
    """

    def __init__(
        self,
        model_class: Type[OffPolicyAlgorithm],
        env: Env,
        cell_factory: CellFactory,
        n_envs: int = 1,
        replay_buffer_kwargs: Optional[Dict[str, Any]] = None,
        model_kwargs: Optional[Dict[str, Any]] = None,
        verbose: int = 0,
    ) -> None:
        self.wrapped_env = env
        # Wrap the env
        def env_func():
            return Goalify(
                maybe_make_env(env, verbose),
                cell_factory,
            )

        env = make_vec_env(env_func, n_envs=n_envs)
        cell_dim = cell_factory(self.wrapped_env.reset()).shape[0]
        replay_buffer_kwargs = {} if replay_buffer_kwargs is None else replay_buffer_kwargs
        replay_buffer_kwargs.update(dict(cell_factory=cell_factory, cell_dim=cell_dim))

        policy_kwargs = dict(
            features_extractor_class=GoExploreExtractor,
            features_extractor_kwargs=dict(cell_dim=cell_dim),
        )
        model_kwargs = {} if model_kwargs is None else model_kwargs
        model_kwargs["learning_starts"] = 10_000
        self.model = model_class(
            "MultiInputPolicy",
            env,
            replay_buffer_class=ArchiveBuffer,
            replay_buffer_kwargs=replay_buffer_kwargs,
            policy_kwargs=policy_kwargs,
            verbose=verbose,
            **model_kwargs,
        )
        self.archive = self.model.replay_buffer  # type: ArchiveBuffer
        for _env in self.model.env.envs:
            _env.set_archive(self.archive)

    def explore(self, total_timesteps: int, update_cell_factory_freq=1_000, reset_num_timesteps: bool = False) -> None:
        """
        Run exploration.

        :param total_timesteps: Total timestep of exploration
        :param update_freq: Cells update frequency
        :param reset_num_timesteps: Whether or not to reset the current timestep number (used in logging), defaults to False
        """
        cell_factory_updater = CallEveryNTimesteps(self.archive.recompute_cells, update_cell_factory_freq)
        self.model.learn(total_timesteps, callback=cell_factory_updater, reset_num_timesteps=reset_num_timesteps)
