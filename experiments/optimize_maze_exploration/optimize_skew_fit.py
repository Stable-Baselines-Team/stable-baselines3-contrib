import gym
import gym_continuous_maze
import numpy as np
import optuna

from sb3_contrib import SkewFit
from sb3_contrib.silver.silver import kde


def objective(trial: optuna.Study):
    nb_models = trial.suggest_categorical("nb_models", [8, 16, 32, 64, 128, 256, 512])
    power = trial.suggest_categorical("power", [-0.2, -0.5, -0.8, -1.0, -1.2, -1.5, -1.8, -2.0])
    num_presampled_goals = trial.suggest_categorical("num_presampled_goals", [256, 512, 1024, 2048, 4096, 8192])

    env = gym.make("ContinuousMaze-v0")
    model = SkewFit(
        env,
        nb_models=nb_models,
        power=power,
        num_presampled_goals=num_presampled_goals,
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
        study_name="ContinuousMaze_SkewFit",
        load_if_exists=True,
        direction="maximize",
    )
    study.optimize(objective, n_trials=10)
    importances = optuna.importance.get_param_importances(study)
    print(importances)
