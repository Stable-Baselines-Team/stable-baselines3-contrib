import gym_continuous_maze
from render_maze import render_and_save
from stable_baselines3 import SAC
from stable_baselines3.common.env_util import make_vec_env

from sb3_contrib import ICM

env = make_vec_env("ContinuousMaze-v0", n_envs=8)
icm = ICM(
    scaling_factor=1.0,
    actor_loss_coef=0.05,
    inverse_loss_coef=0.1,
    forward_loss_coef=2.0,
    obs_dim=env.observation_space.shape[0],
    action_dim=env.action_space.shape[0],
)
model = SAC("MlpPolicy", env, policy_kwargs=dict(net_arch=[256, 256, 256]), surgeon=icm, gradient_steps=8, verbose=1)
model.learn(500000)
im = render_and_save(model.env.envs, "icm_500k.png")
