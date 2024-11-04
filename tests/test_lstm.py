from typing import Dict, Optional

import gymnasium as gym
import numpy as np
import pytest
from gymnasium import spaces
from gymnasium.envs.classic_control import CartPoleEnv
from gymnasium.wrappers import TimeLimit
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.envs import FakeImageEnv
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import VecNormalize

from sb3_contrib import RecurrentPPO


class ToDictWrapper(gym.Wrapper):
    """
    Simple wrapper to test MultInputPolicy on Dict obs.
    """

    def __init__(self, env):
        super().__init__(env)
        self.observation_space = spaces.Dict({"obs": self.env.observation_space})

    def reset(self, **kwargs):
        return {"obs": self.env.reset(**kwargs)[0]}, {}

    def step(self, action):
        obs, reward, done, truncated, infos = self.env.step(action)
        return {"obs": obs}, reward, done, truncated, infos


class CartPoleNoVelEnv(CartPoleEnv):
    """Variant of CartPoleEnv with velocity information removed. This task requires memory to solve."""

    def __init__(self):
        super().__init__()
        high = np.array(
            [
                self.x_threshold * 2,
                self.theta_threshold_radians * 2,
            ]
        ).astype(np.float32)
        self.observation_space = spaces.Box(-high, high, dtype=np.float32)

    @staticmethod
    def _pos_obs(full_obs):
        xpos, _xvel, thetapos, _thetavel = full_obs
        return np.array([xpos, thetapos])

    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict] = None):
        full_obs, info = super().reset(seed=seed, options=options)
        return CartPoleNoVelEnv._pos_obs(full_obs), info

    def step(self, action):
        full_obs, rew, terminated, truncated, info = super().step(action)
        return CartPoleNoVelEnv._pos_obs(full_obs), rew, terminated, truncated, info


def test_env():
    check_env(CartPoleNoVelEnv())


@pytest.mark.parametrize(
    "policy_kwargs",
    [
        {},
        {"share_features_extractor": False},
        dict(shared_lstm=True, enable_critic_lstm=False),
        dict(
            enable_critic_lstm=True,
            lstm_hidden_size=4,
            lstm_kwargs=dict(dropout=0.5),
            n_lstm_layers=2,
        ),
        dict(
            enable_critic_lstm=False,
            lstm_hidden_size=4,
            lstm_kwargs=dict(dropout=0.5),
            n_lstm_layers=2,
        ),
        dict(
            enable_critic_lstm=False,
            lstm_hidden_size=4,
            share_features_extractor=False,
        ),
    ],
)
def test_cnn(policy_kwargs):
    model = RecurrentPPO(
        "CnnLstmPolicy",
        FakeImageEnv(screen_height=40, screen_width=40, n_channels=3),
        n_steps=16,
        seed=0,
        policy_kwargs=dict(**policy_kwargs, features_extractor_kwargs=dict(features_dim=32)),
        n_epochs=2,
    )

    model.learn(total_timesteps=32)


@pytest.mark.parametrize(
    "policy_kwargs",
    [
        {},
        dict(shared_lstm=True, enable_critic_lstm=False),
        dict(
            enable_critic_lstm=True,
            lstm_hidden_size=4,
            lstm_kwargs=dict(dropout=0.5),
            n_lstm_layers=2,
        ),
        dict(
            enable_critic_lstm=False,
            lstm_hidden_size=4,
            lstm_kwargs=dict(dropout=0.5),
            n_lstm_layers=2,
        ),
    ],
)
def test_policy_kwargs(policy_kwargs):
    model = RecurrentPPO(
        "MlpLstmPolicy",
        "CartPole-v1",
        n_steps=16,
        seed=0,
        policy_kwargs=policy_kwargs,
    )

    model.learn(total_timesteps=32)


def test_check():
    policy_kwargs = dict(shared_lstm=True, enable_critic_lstm=True)
    with pytest.raises(AssertionError):
        RecurrentPPO(
            "MlpLstmPolicy",
            "CartPole-v1",
            n_steps=16,
            seed=0,
            policy_kwargs=policy_kwargs,
        )

    policy_kwargs = dict(shared_lstm=True, enable_critic_lstm=False, share_features_extractor=False)
    with pytest.raises(AssertionError):
        RecurrentPPO(
            "MlpLstmPolicy",
            "CartPole-v1",
            n_steps=16,
            seed=0,
            policy_kwargs=policy_kwargs,
        )


@pytest.mark.parametrize("env", ["Pendulum-v1", "CartPole-v1"])
def test_run(env):
    model = RecurrentPPO(
        "MlpLstmPolicy",
        env,
        n_steps=16,
        seed=0,
    )

    model.learn(total_timesteps=32)


def test_run_sde():
    model = RecurrentPPO(
        "MlpLstmPolicy",
        "Pendulum-v1",
        n_steps=16,
        seed=0,
        sde_sample_freq=4,
        use_sde=True,
        clip_range_vf=0.1,
    )

    model.learn(total_timesteps=200)


@pytest.mark.parametrize(
    "policy_kwargs",
    [
        {},
        dict(shared_lstm=True, enable_critic_lstm=False),
        dict(
            enable_critic_lstm=True,
            lstm_hidden_size=4,
            lstm_kwargs=dict(dropout=0.5),
            n_lstm_layers=2,
        ),
        dict(
            enable_critic_lstm=False,
            lstm_hidden_size=4,
            lstm_kwargs=dict(dropout=0.5),
            n_lstm_layers=2,
        ),
    ],
)
def test_dict_obs(policy_kwargs):
    env = make_vec_env("CartPole-v1", n_envs=1, wrapper_class=ToDictWrapper)
    model = RecurrentPPO("MultiInputLstmPolicy", env, n_steps=32, policy_kwargs=policy_kwargs).learn(64)
    evaluate_policy(model, env, warn=False)


@pytest.mark.slow
def test_ppo_lstm_performance():
    # env = make_vec_env("CartPole-v1", n_envs=16)
    def make_env():
        env = CartPoleNoVelEnv()
        env = TimeLimit(env, max_episode_steps=500)
        return env

    env = VecNormalize(make_vec_env(make_env, n_envs=8))

    eval_callback = EvalCallback(
        VecNormalize(make_vec_env(make_env, n_envs=4), training=False, norm_reward=False),
        n_eval_episodes=20,
        eval_freq=5000 // env.num_envs,
    )

    model = RecurrentPPO(
        "MlpLstmPolicy",
        env,
        n_steps=128,
        learning_rate=0.0007,
        verbose=1,
        batch_size=256,
        seed=1,
        n_epochs=10,
        max_grad_norm=1,
        gae_lambda=0.98,
        policy_kwargs=dict(
            net_arch=dict(vf=[64], pi=[]),
            lstm_hidden_size=64,
            ortho_init=False,
            enable_critic_lstm=True,
        ),
    )

    model.learn(total_timesteps=50_000, callback=eval_callback)
    # Maximum episode reward is 500.
    # In CartPole-v1, a non-recurrent policy can easily get >= 450.
    # In CartPoleNoVelEnv, a non-recurrent policy doesn't get more than ~50.
    evaluate_policy(model, env, reward_threshold=450)
