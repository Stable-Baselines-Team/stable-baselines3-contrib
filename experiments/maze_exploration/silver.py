import gym
import gym_continuous_maze
import numpy as np
from render_maze import render_and_save
from stable_baselines3 import SAC

from sb3_contrib import Silver

silver = Silver(
    SAC,
    "ContinuousMaze-v0",
    n_envs=4,
    update_goal_set_freq=2000,
    model_kwargs=dict(
        learning_starts=400,
        policy_kwargs=dict(net_arch=[256, 256, 256]),
    ),
    verbose=1,
)
silver.explore(500000)
goals = np.array([silver.model.replay_buffer.sample_goal() for _ in range(500)])
im = render_and_save(silver.model.env.envs, "silver_500k.png", goals)
