from collections import OrderedDict

from stable_baselines3.common.vec_env.dummy_vec_env import DummyVecEnv
from stable_baselines3.common.vec_env.base_vec_env import VecEnv, VecEnvObs
from stable_baselines3.common.vec_env.util import copy_obs_dict, dict_to_obs, obs_space_info
from typing import Any, Callable, List, Optional, Sequence, Type, Union

import gym
import numpy as np


class SeqVecEnv(DummyVecEnv):
    
    def __init__(self, env_fns: List[Callable[[], gym.Env]]):
        self.envs = [fn() for fn in env_fns]
        if len(set([id(env.unwrapped) for env in self.envs])) != len(self.envs):
            raise ValueError(
                "You tried to create multiple environments, but the function to create them returned the same instance "
                "instead of creating different objects. "
                "You are probably using `make_vec_env(lambda: env)` or `DummyVecEnv([lambda: env] * n_envs)`. "
                "You should replace `lambda: env` by a `make_env` function that "
                "creates a new instance of the environment at every call "
                "(using `gym.make()` for instance). You can take a look at the documentation for an example. "
                "Please read https://github.com/DLR-RM/stable-baselines3/issues/1151 for more information."
            )
        env = self.envs[0]
        obs_space = env.observation_space
        assert isinstance(obs_space, gym.spaces.Sequence)
        VecEnv.__init__(self, len(env_fns), obs_space.feature_space, env.action_space, env.render_mode)
        self.keys, _, _ = obs_space_info(obs_space.feature_space)
        assert self.keys[0] is None

        self.buf_obs = OrderedDict([(k, [[None] for _ in range(self.num_envs)]) for k in self.keys])
        self.buf_dones = np.zeros((self.num_envs,), dtype=bool)
        self.buf_rews = np.zeros((self.num_envs,), dtype=np.float32)
        self.buf_infos = [{} for _ in range(self.num_envs)]
        self.actions = None
        self.metadata = env.metadata