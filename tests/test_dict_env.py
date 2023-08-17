from typing import Dict, Optional

import gymnasium as gym
import numpy as np
import pytest
from gymnasium import spaces
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.envs import SimpleMultiObsEnv
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack, VecNormalize

from sb3_contrib import QRDQN, TQC, TRPO


class DummyDictEnv(gym.Env):
    """Custom Environment for testing purposes only"""

    metadata = {"render.modes": ["human"]}

    def __init__(
        self,
        use_discrete_actions=False,
        channel_last=False,
        nested_dict_obs=False,
        vec_only=False,
    ):
        super().__init__()
        if use_discrete_actions:
            self.action_space = spaces.Discrete(3)
        else:
            self.action_space = spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32)
        N_CHANNELS = 1
        HEIGHT = 36
        WIDTH = 36

        if channel_last:
            obs_shape = (HEIGHT, WIDTH, N_CHANNELS)
        else:
            obs_shape = (N_CHANNELS, HEIGHT, WIDTH)

        self.observation_space = spaces.Dict(
            {
                # Image obs
                "img": spaces.Box(low=0, high=255, shape=obs_shape, dtype=np.uint8),
                # Vector obs
                "vec": spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32),
                # Discrete obs
                "discrete": spaces.Discrete(4),
            }
        )

        # For checking consistency with normal MlpPolicy
        if vec_only:
            self.observation_space = spaces.Dict(
                {
                    # Vector obs
                    "vec": spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32),
                }
            )

        if nested_dict_obs:
            # Add dictionary observation inside observation space
            self.observation_space.spaces["nested-dict"] = spaces.Dict({"nested-dict-discrete": spaces.Discrete(4)})

    def step(self, action):
        reward = 0.0
        done = truncated = False
        return self.observation_space.sample(), reward, done, truncated, {}

    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict] = None):
        if seed is not None:
            self.observation_space.seed(seed)
        return self.observation_space.sample(), {}

    def render(self):
        pass


@pytest.mark.parametrize("use_discrete_actions", [True, False])
@pytest.mark.parametrize("channel_last", [True, False])
@pytest.mark.parametrize("nested_dict_obs", [True, False])
@pytest.mark.parametrize("vec_only", [True, False])
def test_env(use_discrete_actions, channel_last, nested_dict_obs, vec_only):
    # Check the env used for testing
    if nested_dict_obs:
        with pytest.warns(UserWarning, match="Nested observation spaces are not supported"):
            check_env(DummyDictEnv(use_discrete_actions, channel_last, nested_dict_obs, vec_only))
    else:
        check_env(DummyDictEnv(use_discrete_actions, channel_last, nested_dict_obs, vec_only))


@pytest.mark.parametrize("model_class", [QRDQN, TQC, TRPO])
def test_consistency(model_class):
    """
    Make sure that dict obs with vector only vs using flatten obs is equivalent.
    This ensures notable that the network architectures are the same.
    """
    use_discrete_actions = model_class == QRDQN
    dict_env = DummyDictEnv(use_discrete_actions=use_discrete_actions, vec_only=True)
    dict_env = gym.wrappers.TimeLimit(dict_env, 100)
    env = gym.wrappers.FlattenObservation(dict_env)
    obs, _ = dict_env.reset(seed=10)

    kwargs = {}
    n_steps = 256

    if model_class in {TRPO}:
        kwargs = dict(
            n_steps=128,
        )
    else:
        # Avoid memory error when using replay buffer
        # Reduce the size of the features and make learning faster
        kwargs = dict(
            buffer_size=250,
            train_freq=8,
            gradient_steps=1,
        )
        if model_class == QRDQN:
            kwargs["learning_starts"] = 0

    dict_model = model_class("MultiInputPolicy", dict_env, gamma=0.5, seed=1, **kwargs)
    action_before_learning_1, _ = dict_model.predict(obs, deterministic=True)
    dict_model.learn(total_timesteps=n_steps)

    normal_model = model_class("MlpPolicy", env, gamma=0.5, seed=1, **kwargs)
    action_before_learning_2, _ = normal_model.predict(obs["vec"], deterministic=True)
    normal_model.learn(total_timesteps=n_steps)

    action_1, _ = dict_model.predict(obs, deterministic=True)
    action_2, _ = normal_model.predict(obs["vec"], deterministic=True)

    assert np.allclose(action_before_learning_1, action_before_learning_2)
    assert np.allclose(action_1, action_2)


@pytest.mark.parametrize("model_class", [QRDQN, TQC, TRPO])
@pytest.mark.parametrize("channel_last", [False, True])
def test_dict_spaces(model_class, channel_last):
    """
    Additional tests to check observation space support
    with mixed observation.
    """
    use_discrete_actions = model_class not in [TQC]
    env = DummyDictEnv(use_discrete_actions=use_discrete_actions, channel_last=channel_last)
    env = gym.wrappers.TimeLimit(env, 100)

    kwargs = {}
    n_steps = 256

    if model_class in {TRPO}:
        kwargs = dict(
            n_steps=128,
            policy_kwargs=dict(
                net_arch=dict(pi=[32], vf=[32]),
                features_extractor_kwargs=dict(cnn_output_dim=32),
            ),
        )
    else:
        # Avoid memory error when using replay buffer
        # Reduce the size of the features and make learning faster
        kwargs = dict(
            buffer_size=250,
            policy_kwargs=dict(
                net_arch=[32],
                features_extractor_kwargs=dict(cnn_output_dim=32),
                n_quantiles=20,
            ),
            train_freq=8,
            gradient_steps=1,
        )
        if model_class == QRDQN:
            kwargs["learning_starts"] = 0

    model = model_class("MultiInputPolicy", env, gamma=0.5, seed=1, **kwargs)

    model.learn(total_timesteps=n_steps)

    evaluate_policy(model, env, n_eval_episodes=5, warn=False)


@pytest.mark.parametrize("model_class", [QRDQN, TQC, TRPO])
@pytest.mark.parametrize("channel_last", [False, True])
def test_dict_vec_framestack(model_class, channel_last):
    """
    Additional tests to check observation space support
    for Dictionary spaces and VecEnvWrapper using MultiInputPolicy.
    """
    use_discrete_actions = model_class not in [TQC]
    channels_order = {"vec": None, "img": "last" if channel_last else "first"}
    env = DummyVecEnv(
        [lambda: SimpleMultiObsEnv(random_start=True, discrete_actions=use_discrete_actions, channel_last=channel_last)]
    )

    env = VecFrameStack(env, n_stack=3, channels_order=channels_order)

    kwargs = {}
    n_steps = 256

    if model_class in {TRPO}:
        kwargs = dict(
            n_steps=128,
            policy_kwargs=dict(
                net_arch=dict(pi=[32], vf=[32]),
                features_extractor_kwargs=dict(cnn_output_dim=32),
            ),
        )
    else:
        # Avoid memory error when using replay buffer
        # Reduce the size of the features and make learning faster
        kwargs = dict(
            buffer_size=250,
            policy_kwargs=dict(
                net_arch=[32],
                features_extractor_kwargs=dict(cnn_output_dim=32),
                n_quantiles=20,
            ),
            train_freq=8,
            gradient_steps=1,
        )
        if model_class == QRDQN:
            kwargs["learning_starts"] = 0

    model = model_class("MultiInputPolicy", env, gamma=0.5, seed=1, **kwargs)

    model.learn(total_timesteps=n_steps)

    evaluate_policy(model, env, n_eval_episodes=5, warn=False)


@pytest.mark.parametrize("model_class", [QRDQN, TQC, TRPO])
def test_vec_normalize(model_class):
    """
    Additional tests to check observation space support
    for GoalEnv and VecNormalize using MultiInputPolicy.
    """
    env = DummyVecEnv([lambda: gym.wrappers.TimeLimit(DummyDictEnv(use_discrete_actions=model_class == QRDQN), 100)])
    env = VecNormalize(env, norm_obs_keys=["vec"])

    kwargs = {}
    n_steps = 256

    if model_class in {TRPO}:
        kwargs = dict(
            n_steps=128,
            policy_kwargs=dict(
                net_arch=dict(pi=[32], vf=[32]),
            ),
        )
    else:
        # Avoid memory error when using replay buffer
        # Reduce the size of the features and make learning faster
        kwargs = dict(
            buffer_size=250,
            policy_kwargs=dict(
                net_arch=[32],
            ),
            train_freq=8,
            gradient_steps=1,
        )
        if model_class == QRDQN:
            kwargs["learning_starts"] = 0

    model = model_class("MultiInputPolicy", env, gamma=0.5, seed=1, **kwargs)

    model.learn(total_timesteps=n_steps)

    evaluate_policy(model, env, n_eval_episodes=5, warn=False)
