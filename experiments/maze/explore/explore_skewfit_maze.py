import os

import gym
import gym_continuous_maze
import numpy as np
from toolbox.maze_grid import compute_coverage

from sb3_contrib import SkewFit

NUM_TIMESTEPS = 100_000
NUM_RUN = 5

for run_idx in range(NUM_RUN):
    env = gym.make("ContinuousMaze-v0")
    model = SkewFit(env, nb_models=200, power=-1.0, num_presampled_goals=64, verbose=1)
    model.learn(NUM_TIMESTEPS)
    buffer = model.replay_buffer
    observations = buffer.next_observations["observation"][: buffer.pos if not buffer.full else buffer.buffer_size]
    coverage = compute_coverage(observations) / (24 * 24) * 100
    coverage = np.expand_dims(coverage, 0)

    filename = "results/skewfit_maze.npy"
    if os.path.exists(filename):
        previous_coverage = np.load(filename)
        coverage = np.concatenate((previous_coverage, coverage))
    np.save(filename, coverage)
