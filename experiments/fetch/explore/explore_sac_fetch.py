import os

import gym
import gym_robotics
import numpy as np
from stable_baselines3 import SAC
from toolbox.fetch_utils import cumulative_object_coverage

NUM_TIMESTEPS = 1_000_000
NUM_RUN = 5

for run_idx in range(NUM_RUN):
    env = gym.make("__root__/FetchNoTask-v1")
    model = SAC("MlpPolicy", env, verbose=1)
    model.learn(NUM_TIMESTEPS)
    buffer = model.replay_buffer
    observations = buffer.next_observations[: buffer.pos if not buffer.full else buffer.buffer_size]
    coverage = cumulative_object_coverage(observations)
    coverage = np.expand_dims(coverage, 0)

    filename = "results/sac_fetch.npy"
    if os.path.exists(filename):
        previous_coverage = np.load(filename)
        coverage = np.concatenate((previous_coverage, coverage))
    np.save(filename, coverage)
