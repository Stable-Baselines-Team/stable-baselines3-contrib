import gym_continuous_maze
from render_maze import render_and_save
from stable_baselines3 import SAC
from stable_baselines3.common.env_util import make_vec_env

from sb3_contrib import RND

env = make_vec_env("ContinuousMaze-v0", n_envs=8)
rnd = RND(
    scaling_factor=0.5,
    obs_dim=env.observation_space.shape[0],
    net_arch=[256, 256],
    out_dim=2,
    train_freq=128,
)
model = SAC("MlpPolicy", env, policy_kwargs=dict(net_arch=[256, 256, 256]), surgeon=rnd, gradient_steps=8, verbose=1)
model.learn(500000)
im = render_and_save(model.env.envs, "rnd_500k.png")
