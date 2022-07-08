import gym
import gym_continuous_maze
import numpy as np
import optuna
from stable_baselines3 import SAC
from toolbox.maze_grid import compute_coverage

from sb3_contrib import Surprise

NUM_TIMESTEPS = 30_000
NUM_RUN = 5


def objective(trial: optuna.Trial) -> float:
    feature_dim = trial.suggest_categorical("feature_dim", [2, 4, 8, 16, 32])
    eta_0 = trial.suggest_categorical("eta_0", [0.01, 0.1, 1.0, 10])
    train_freq = trial.suggest_categorical("train_freq", [2, 4, 8, 16, 32, 64, 128])
    lr = trial.suggest_categorical("lr", [1e-6, 1e-5, 1e-4, 1e-3, 1e-2])

    coverage = np.zeros((NUM_RUN, NUM_TIMESTEPS))
    for run_idx in range(NUM_RUN):
        env = gym.make("ContinuousMaze-v0")
        surprise = Surprise(
            obs_dim=env.observation_space.shape[0],
            action_dim=env.action_space.shape[0],
            feature_dim=feature_dim,
            eta_0=eta_0,
            train_freq=train_freq,
            lr=lr,
        )

        model = SAC("MlpPolicy", env, surgeon=surprise, verbose=1)
        model.learn(NUM_TIMESTEPS)
        buffer = model.replay_buffer
        observations = buffer.next_observations[: buffer.pos if not buffer.full else buffer.buffer_size]
        coverage[run_idx] = compute_coverage(observations) / (24 * 24) * 100

    score = np.median(coverage[:, -1])
    return score


if __name__ == "__main__":
    study = optuna.create_study(
        storage="sqlite:///optuna.db", study_name="surprise_maze", load_if_exists=True, direction="maximize"
    )
    study.optimize(objective, n_trials=30)
    print(study.best_params, study.best_value)
