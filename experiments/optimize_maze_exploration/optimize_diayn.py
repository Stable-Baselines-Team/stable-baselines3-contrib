import gym
import gym_continuous_maze
import numpy as np
import optuna

from sb3_contrib import DIAYN
from sb3_contrib.silver.silver import kde


def objective(trial: optuna.Study):
    nb_skills = trial.suggest_categorical("nb_skills", [8, 16, 32, 64, 128, 256, 512])

    env = gym.make("ContinuousMaze-v0")
    model = DIAYN(
        env,
        nb_skills=nb_skills,
        policy_kwargs=dict(net_arch=[256, 256, 256]),
        verbose=1,
    )
    model.learn(20000)

    x = np.array(model.env.envs[0].all_pos)
    density = kde(x, x)
    entropy = 1 / np.mean(density)
    return entropy


if __name__ == "__main__":
    study = optuna.create_study(
        storage="sqlite:///optuna.db",
        study_name="ContinuousMaze_DIAYN",
        load_if_exists=True,
        direction="maximize",
    )
    study.optimize(objective, n_trials=10)
    importances = optuna.importance.get_param_importances(study)
    print(importances)
