import gym
import numpy as np
import pytest
from gym import spaces
from gym.envs.classic_control import CartPoleEnv
from gym.wrappers.time_limit import TimeLimit
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy

from sb3_contrib import RecurrentPPO


class ToDictWrapper(gym.Wrapper):
    """
    Simple wrapper to test MultInputPolicy on Dict obs.
    """

    def __init__(self, env):
        super().__init__(env)
        self.observation_space = gym.spaces.Dict({"obs": self.env.observation_space})

    def reset(self):
        return {"obs": self.env.reset()}

    def step(self, action):
        obs, reward, done, infos = self.env.step(action)
        return {"obs": obs}, reward, done, infos


class CartPoleNoVelEnv(CartPoleEnv):
    """Variant of CartPoleEnv with velocity information removed. This task requires memory to solve."""

    def __init__(self):
        super().__init__()
        high = np.array(
            [
                self.x_threshold * 2,
                self.theta_threshold_radians * 2,
            ]
        )
        self.observation_space = spaces.Box(-high, high, dtype=np.float32)

    @staticmethod
    def _pos_obs(full_obs):
        xpos, _xvel, thetapos, _thetavel = full_obs
        return xpos, thetapos

    def reset(self):
        full_obs = super().reset()
        return CartPoleNoVelEnv._pos_obs(full_obs)

    def step(self, action):
        full_obs, rew, done, info = super().step(action)
        return CartPoleNoVelEnv._pos_obs(full_obs), rew, done, info


def test_cnn():
    model = RecurrentPPO(
        "CnnLstmPolicy",
        "Breakout-v0",
        n_steps=16,
        seed=0,
        policy_kwargs=dict(features_extractor_kwargs=dict(features_dim=32)),
    )

    model.learn(total_timesteps=32)


@pytest.mark.parametrize("policy_kwargs", [{}, dict(shared_lstm=True), dict(enable_critic_lstm=True, lstm_hidden_size=4)])
def test_policy_kwargs(policy_kwargs):
    model = RecurrentPPO(
        "MlpLstmPolicy",
        "CartPole-v1",
        n_steps=16,
        seed=0,
        policy_kwargs=policy_kwargs,
    )

    model.learn(total_timesteps=32)


@pytest.mark.parametrize("env", ["Pendulum-v0", "CartPole-v1"])
def test_run(env):
    model = RecurrentPPO(
        "MlpLstmPolicy",
        env,
        n_steps=16,
        seed=0,
        create_eval_env=True,
    )

    model.learn(total_timesteps=32, eval_freq=16)


def test_run_sde():
    model = RecurrentPPO(
        "MlpLstmPolicy",
        "Pendulum-v0",
        n_steps=16,
        seed=0,
        create_eval_env=True,
        sde_sample_freq=4,
        use_sde=True,
    )

    model.learn(total_timesteps=32, eval_freq=16)


def test_dict_obs():
    env = make_vec_env("CartPole-v1", n_envs=1, wrapper_class=ToDictWrapper)
    model = RecurrentPPO("MultiInputLstmPolicy", env, n_steps=32).learn(64)
    evaluate_policy(model, env, warn=False)


@pytest.mark.slow
def test_ppo_lstm_performance():
    # env = make_vec_env("CartPole-v1", n_envs=16)
    def make_env():
        env = CartPoleNoVelEnv()
        env = TimeLimit(env, max_episode_steps=500)
        return env

    env = make_vec_env(make_env, n_envs=8)

    eval_callback = EvalCallback(
        make_vec_env(make_env, n_envs=4),
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
        seed=0,
        n_epochs=10,
        max_grad_norm=1,
        gae_lambda=0.98,
        policy_kwargs=dict(net_arch=[dict(vf=[64])], ortho_init=False),
    )

    model.learn(total_timesteps=50_000, callback=eval_callback)
    # Maximum episode reward is 500.
    # In CartPole-v1, a non-recurrent policy can easily get >= 450.
    # In CartPoleNoVelEnv, a non-recurrent policy doesn't get more than ~50.
    # LSTM policies can reach above 400, but it varies a lot between runs; consistently get >=150.
    evaluate_policy(model, env, reward_threshold=160)
