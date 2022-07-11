import os

import gym
import panda_gym
import numpy as np
from stable_baselines3 import SAC
from toolbox.fetch_utils import cumulative_object_coverage

from sb3_contrib import Surprise

NUM_TIMESTEPS = 1_000_000
NUM_RUN = 5

for run_idx in range(NUM_RUN):
    env = gym.make("PandaNoTask-v0", nb_objects=1)
    surprise = Surprise(
        env.observation_space.shape[0],
        env.action_space.shape[0],
        eta_0=0.01,
        feature_dim=2,
        lr=1e-5,
        train_freq=64,
    )
    model = SAC("MlpPolicy", env, surgeon=surprise, verbose=1)
    model.learn(NUM_TIMESTEPS)
    buffer = model.replay_buffer
    observations = buffer.next_observations[: buffer.pos if not buffer.full else buffer.buffer_size]
    coverage = cumulative_object_coverage(observations)
    coverage = np.expand_dims(coverage, 0)

    filename = "results/surprise_panda.npy"
    if os.path.exists(filename):
        previous_coverage = np.load(filename)
        coverage = np.concatenate((previous_coverage, coverage))
    np.save(filename, coverage)
