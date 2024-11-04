from typing import Any, Dict, SupportsFloat, Tuple, Union

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from gymnasium.core import ActType

TimeFeatureObs = Union[np.ndarray, Dict[str, np.ndarray]]


class TimeFeatureWrapper(gym.Wrapper[TimeFeatureObs, ActType, TimeFeatureObs, ActType]):
    """
    Add remaining, normalized time to observation space for fixed length episodes.
    See https://arxiv.org/abs/1712.00378 and https://github.com/aravindr93/mjrl/issues/13.

    .. note::

        Only ``gym.spaces.Box`` and ``gym.spaces.Dict`` (``gym.GoalEnv``) 1D observation spaces
        are supported for now.

    :param env: Gym env to wrap.
    :param max_steps: Max number of steps of an episode
        if it is not wrapped in a ``TimeLimit`` object.
    :param test_mode: In test mode, the time feature is constant,
        equal to zero. This allow to check that the agent did not overfit this feature,
        learning a deterministic pre-defined sequence of actions.
    """

    def __init__(self, env: gym.Env, max_steps: int = 1000, test_mode: bool = False):
        assert isinstance(
            env.observation_space, (spaces.Box, spaces.Dict)
        ), "`TimeFeatureWrapper` only supports `gym.spaces.Box` and `spaces.Dict` (`gym.GoalEnv`) observation spaces."

        # Add a time feature to the observation
        if isinstance(env.observation_space, spaces.Dict):
            assert "observation" in env.observation_space.spaces, "No `observation` key in the observation space"
            obs_space = env.observation_space.spaces["observation"]
        else:
            obs_space = env.observation_space

        assert isinstance(obs_space, gym.spaces.Box), "`TimeFeatureWrapper` only supports `gym.spaces.Box` observation space."
        assert len(obs_space.shape) == 1, "Only 1D observation spaces are supported"

        low, high = obs_space.low, obs_space.high
        low, high = np.concatenate((low, [0.0])), np.concatenate((high, [1.0]))  # type: ignore[arg-type]
        self.dtype = obs_space.dtype
        low, high = low.astype(self.dtype), high.astype(self.dtype)

        if isinstance(env.observation_space, spaces.Dict):
            env.observation_space.spaces["observation"] = spaces.Box(
                low=low,
                high=high,
                dtype=self.dtype,  # type: ignore[arg-type]
            )
        else:
            env.observation_space = spaces.Box(low=low, high=high, dtype=self.dtype)  # type: ignore[arg-type]

        super().__init__(env)

        # Try to infer the max number of steps per episode
        try:
            self._max_steps = env.spec.max_episode_steps  # type: ignore[union-attr]
        except AttributeError:
            self._max_steps = None

        # Fallback to provided value
        if self._max_steps is None:
            self._max_steps = max_steps

        self._current_step = 0
        self._test_mode = test_mode

    def reset(self, **kwargs) -> Tuple[TimeFeatureObs, Dict[str, Any]]:
        self._current_step = 0
        obs, info = self.env.reset(**kwargs)
        return self._get_obs(obs), info

    def step(self, action: ActType) -> Tuple[TimeFeatureObs, SupportsFloat, bool, bool, Dict[str, Any]]:
        self._current_step += 1
        obs, reward, done, truncated, info = self.env.step(action)
        return self._get_obs(obs), reward, done, truncated, info

    def _get_obs(self, obs: Union[np.ndarray, Dict[str, np.ndarray]]) -> Union[np.ndarray, Dict[str, np.ndarray]]:
        """
        Concatenate the time feature to the current observation.

        :param obs:
        :return:
        """
        # for mypy
        assert self._max_steps is not None
        # Remaining time is more general
        time_feature = 1 - (self._current_step / self._max_steps)
        if self._test_mode:
            time_feature = 1.0
        time_feature = np.array(time_feature, dtype=self.dtype)  # type: ignore[assignment]

        if isinstance(obs, dict):
            obs["observation"] = np.append(obs["observation"], time_feature)
            return obs
        return np.append(obs, time_feature)
