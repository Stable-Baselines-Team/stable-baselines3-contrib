from typing import Callable, Optional

import gym
import numpy as np
from stable_baselines3.common.type_aliases import GymEnv
from stable_baselines3.common.vec_env import VecEnv


def get_action_masks(env: GymEnv) -> np.ndarray:
    method_name = "action_masks"

    if isinstance(env, VecEnv):
        return np.stack(env.env_method(method_name))
    else:
        return getattr(env, method_name)()


def get_action_mask_fn(env: GymEnv) -> Optional[Callable[[gym.Env], np.ndarray]]:
    attr_name = "action_mask_fn"

    try:
        if isinstance(env, VecEnv):
            return env.get_attr(attr_name)[0]
        else:
            return getattr(env, attr_name)
    except AttributeError:
        return None


def is_masking_supported(env: GymEnv) -> bool:
    return get_action_mask_fn(env) is not None
