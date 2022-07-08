import os

import gym
import gym_continuous_maze
import numpy as np
from toolbox.maze_grid import compute_coverage

from sb3_contrib import DIAYN

NUM_TIMESTEPS = 30_000
NUM_RUN = 5

for run_idx in range(NUM_RUN):
    env = gym.make("ContinuousMaze-v0")
    model = DIAYN(env, nb_skills=32, verbose=1)
    model.learn(NUM_TIMESTEPS)
    buffer = model.replay_buffer
    observations = buffer.next_observations["observation"][: buffer.pos if not buffer.full else buffer.buffer_size]
    coverage = compute_coverage(observations) / (24 * 24) * 100
    coverage = np.expand_dims(coverage, 0)

    filename = "results/diayn_maze.npy"
    if os.path.exists(filename):
        previous_coverage = np.load(filename)
        counts = np.concatenate((previous_coverage, coverage))
    np.save(filename, coverage)
