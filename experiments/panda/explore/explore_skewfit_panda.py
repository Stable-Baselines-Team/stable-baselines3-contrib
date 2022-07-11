import os

import gym
import panda_gym
import numpy as np
from toolbox.fetch_utils import cumulative_object_coverage

from sb3_contrib import SkewFit

NUM_TIMESTEPS = 1_000_000
NUM_RUN = 5

for run_idx in range(NUM_RUN):
    env = gym.make("PandaNoTask-v0", nb_objects=1)
    model = SkewFit(env, nb_models=200, power=-1.0, num_presampled_goals=64, verbose=1)
    model.learn(NUM_TIMESTEPS)
    buffer = model.replay_buffer
    observations = buffer.next_observations["observation"][: buffer.pos if not buffer.full else buffer.buffer_size]
    coverage = cumulative_object_coverage(observations)
    coverage = np.expand_dims(coverage, 0)

    filename = "results/skewfit_panda.npy"
    if os.path.exists(filename):
        previous_coverage = np.load(filename)
        coverage = np.concatenate((previous_coverage, coverage))
    np.save(filename, coverage)
