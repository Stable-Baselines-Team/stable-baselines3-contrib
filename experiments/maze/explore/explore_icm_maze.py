import os

import gym
import gym_continuous_maze
import numpy as np
from stable_baselines3 import SAC
from toolbox.maze_grid import compute_coverage

from sb3_contrib import ICM

NUM_TIMESTEPS = 30_000
NUM_RUN = 5

for run_idx in range(NUM_RUN):
    env = gym.make("ContinuousMaze-v0")
    icm = ICM(
        scaling_factor=0.1,
        actor_loss_coef=100,
        inverse_loss_coef=10,
        forward_loss_coef=10,
        obs_dim=env.observation_space.shape[0],
        action_dim=env.action_space.shape[0],
    )
    model = SAC("MlpPolicy", env, surgeon=icm, verbose=1)
    model.learn(NUM_TIMESTEPS)
    buffer = model.replay_buffer
    observations = buffer.next_observations[: buffer.pos if not buffer.full else buffer.buffer_size]
    coverage = compute_coverage(observations) / (24 * 24) * 100
    coverage = np.expand_dims(coverage, 0)

    filename = "results/icm_maze.npy"
    if os.path.exists(filename):
        previous_coverage = np.load(filename)
        counts = np.concatenate((previous_coverage, coverage))
    np.save(filename, coverage)