import os

import gym
import gym_continuous_maze
import numpy as np
from stable_baselines3 import SAC
from toolbox.maze_grid import compute_coverage

from sb3_contrib import GoExplore
from sb3_contrib.go_explore.cells import Downscale

NUM_TIMESTEPS = 100_000
NUM_RUN = 5

for run_idx in range(NUM_RUN):
    env = gym.make("ContinuousMaze-v0")
    cell_factory = Downscale(np.log(2.0) / np.log(10))
    model = GoExplore(SAC, env, cell_factory, verbose=1)
    model.explore(NUM_TIMESTEPS)
    buffer = model.archive
    observations = buffer.next_observations["observation"][: buffer.pos if not buffer.full else buffer.buffer_size]
    coverage = compute_coverage(observations) / (24 * 24) * 100
    coverage = np.expand_dims(coverage, 0)

    filename = "results/go_explore_maze.npy"
    if os.path.exists(filename):
        previous_coverage = np.load(filename)
        coverage = np.concatenate((previous_coverage, coverage))
    np.save(filename, coverage)
