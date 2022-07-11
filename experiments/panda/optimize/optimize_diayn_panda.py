import gym
import panda_gym
import numpy as np
import optuna
from toolbox.fetch_utils import cumulative_object_coverage

from sb3_contrib import DIAYN

NUM_TIMESTEPS = 300_000
NUM_RUN = 5


def objective(trial: optuna.Trial) -> float:
    nb_skills = trial.suggest_categorical("nb_skills", [4, 8, 16, 32, 64, 128])

    coverage = np.zeros((NUM_RUN, NUM_TIMESTEPS))
    for run_idx in range(NUM_RUN):
        env = gym.make("PandaNoTask-v0", nb_objects=1)
        model = DIAYN(env, nb_skills, verbose=1)
        model.learn(NUM_TIMESTEPS)
        buffer = model.replay_buffer
        observations = buffer.next_observations["observation"][: buffer.pos if not buffer.full else buffer.buffer_size]
        coverage[run_idx] = cumulative_object_coverage(observations) / (24 * 24) * 100

    score = np.median(coverage[:, -1])
    return score


if __name__ == "__main__":
    study = optuna.create_study(
        storage="sqlite:///optuna.db", study_name="diayn_panda", load_if_exists=True, direction="maximize"
    )
    study.optimize(objective, n_trials=30)
    print(study.best_params, study.best_value)
