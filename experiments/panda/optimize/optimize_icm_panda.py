import gym
import panda_gym
import numpy as np
import optuna
from stable_baselines3 import SAC
from toolbox.fetch_utils import cumulative_object_coverage

from sb3_contrib import ICM

NUM_TIMESTEPS = 300_000
NUM_RUN = 5


def objective(trial: optuna.Trial) -> float:
    scaling_factor = trial.suggest_categorical("scaling_factor", [0.001, 0.01, 0.1, 1, 10, 100])
    actor_loss_coef = trial.suggest_categorical("actor_loss_coef", [0.001, 0.01, 0.1, 1, 10, 100])
    inverse_loss_coef = trial.suggest_categorical("inverse_loss_coef", [0.001, 0.01, 0.1, 1, 10, 100])
    forward_loss_coef = trial.suggest_categorical("forward_loss_coef", [0.001, 0.01, 0.1, 1, 10, 100])

    coverage = np.zeros((NUM_RUN, NUM_TIMESTEPS))
    for run_idx in range(NUM_RUN):
        env = gym.make("PandaNoTask-v0", nb_objects=1)
        icm = ICM(
            scaling_factor,
            actor_loss_coef,
            inverse_loss_coef,
            forward_loss_coef,
            obs_dim=env.observation_space.shape[0],
            action_dim=env.action_space.shape[0],
        )
        model = SAC("MlpPolicy", env, surgeon=icm, verbose=1)
        model.learn(NUM_TIMESTEPS)
        buffer = model.replay_buffer
        observations = buffer.next_observations[: buffer.pos if not buffer.full else buffer.buffer_size]
        coverage[run_idx] = cumulative_object_coverage(observations) / (24 * 24) * 100

    score = np.median(coverage[:, -1])
    return score


if __name__ == "__main__":
    study = optuna.create_study(
        storage="sqlite:///optuna.db", study_name="icm_panda", load_if_exists=True, direction="maximize"
    )
    study.optimize(objective, n_trials=30)
    print(study.best_params, study.best_value)
