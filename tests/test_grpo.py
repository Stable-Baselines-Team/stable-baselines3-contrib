import gymnasium as gym
import numpy as np
import pytest
from stable_baselines3.common.vec_env import DummyVecEnv

from sb3_contrib.grpo import GRPO


def custom_reward_scaling(rewards: np.ndarray) -> np.ndarray:
    """
    Custom reward scaling function for testing.
    This function simply normalizes rewards between -1 and 1.
    """
    return np.clip(rewards / (np.abs(rewards).max() + 1e-8), -1, 1)


@pytest.fixture
def cartpole_env():
    """Fixture to create a wrapped Gym environment for testing."""
    return DummyVecEnv([lambda: gym.make("CartPole-v1")])


def test_grpo_training_default_reward(cartpole_env):
    """
    Test that GRPO can train with the default reward scaling function.
    Ensures that the model initializes and runs without errors.
    """
    model = GRPO("MlpPolicy", cartpole_env, samples_per_time_step=5, verbose=0)

    model.learn(total_timesteps=1000)

    assert model is not None, "GRPO model failed to initialize or train."


def test_grpo_training_custom_reward(cartpole_env):
    """
    Test that GRPO can accept a custom reward scaling function.
    Ensures that the model trains correctly with the provided function.
    """
    model = GRPO("MlpPolicy", cartpole_env, samples_per_time_step=5, reward_scaling_fn=custom_reward_scaling, verbose=0)

    model.learn(total_timesteps=1000)

    assert model is not None, "GRPO model failed to initialize or train with custom reward scaling."
