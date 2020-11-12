import gym
import numpy as np
import pytest
from gym import spaces
from stable_baselines3.common.bit_flipping_env import BitFlippingEnv
from stable_baselines3.common.env_checker import check_env

from sb3_contrib.common.wrappers import TimeFeatureWrapper


class CustomGoalEnv(gym.GoalEnv):
    """docstring for CustomGoalEnv."""

    def __init__(self):
        super(CustomGoalEnv, self).__init__()
        self.observation_space = spaces.Dict(
            {
                "observation": spaces.Box(low=-np.inf, high=np.inf, shape=(3,)),
                "achieved_goal": spaces.Box(low=-np.inf, high=np.inf, shape=(3,)),
                "desired_goal": spaces.Box(low=-np.inf, high=np.inf, shape=(3,)),
            }
        )
        self.action_space = spaces.Box(low=-1, high=1, shape=(5,), dtype=np.float32)

    def reset(self):
        return self.observation_space.sample()

    def step(self, action):
        return self.observation_space.sample(), 0.0, False, {}


def check_time_feature(obs, timestep, max_timesteps):
    assert np.allclose(obs[-1], 1.0 - timestep / max_timesteps)


def test_time_feature():
    env = gym.make("Pendulum-v0")
    env = TimeFeatureWrapper(env)
    check_env(env, warn=False)
    obs = env.reset()
    check_time_feature(obs, timestep=0, max_timesteps=200)
    obs, _, _, _ = env.step(env.action_space.sample())
    check_time_feature(obs, timestep=1, max_timesteps=200)

    env = BitFlippingEnv()
    with pytest.raises(AssertionError):
        env = TimeFeatureWrapper(env)

    env = CustomGoalEnv()
    env = TimeFeatureWrapper(env, max_steps=500)
    obs = env.reset()
    check_time_feature(obs["observation"], timestep=0, max_timesteps=500)
    obs, _, _, _ = env.step(env.action_space.sample())
    check_time_feature(obs["observation"], timestep=1, max_timesteps=500)

    # In test mode, the time feature must be constant
    env = gym.make("Pendulum-v0")
    env = TimeFeatureWrapper(env, test_mode=True)
    obs = env.reset()
    check_time_feature(obs, timestep=0, max_timesteps=200)
    obs, _, _, _ = env.step(env.action_space.sample())
    # Should be the same
    check_time_feature(obs, timestep=0, max_timesteps=200)
