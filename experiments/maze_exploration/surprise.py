import gym_continuous_maze
from render_maze import render_and_save
from stable_baselines3 import SAC
from stable_baselines3.common.env_util import make_vec_env

from sb3_contrib import  Surprise

env = make_vec_env("ContinuousMaze-v0", n_envs=8)
surprise = Surprise(
    env.observation_space.shape[0],
    env.action_space.shape[0],
    feature_dim=8,
    eta_0=0.1,
    train_freq=4,
    lr=0.0001,
    batch_size=256,
)
model = SAC(
    "MlpPolicy",
    env,
    policy_kwargs=dict(net_arch=[256, 256, 256]),
    surgeon=surprise,
    verbose=1,
    gradient_steps=8,
)
model.learn(500000)
render_and_save(model.env.envs, "surprise_500k.png")
