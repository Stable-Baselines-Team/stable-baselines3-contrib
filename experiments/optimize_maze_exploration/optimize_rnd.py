import gym
import gym_continuous_maze
import numpy as np
import optuna
from stable_baselines3 import SAC

from sb3_contrib import RND
from sb3_contrib.silver.silver import kde


def objective(trial: optuna.Study):
    scaling_factor = trial.suggest_categorical(
        "scaling_factor",
        [1e-3, 2e-3, 5e-3, 1e-2, 2e-2, 5e-2, 1e-1, 2e-1, 5e-1, 1e0, 2e0, 5e0, 1e1, 2e1, 5e1, 1e2, 2e2, 5e2, 1e3],
    )
    train_freq = trial.suggest_categorical("out_dim", [1, 2, 4, 8, 16, 32, 64, 128, 256, 512])
    out_dim = trial.suggest_categorical("train_freq", [16, 32, 64, 128, 256, 512])

    env = gym.make("ContinuousMaze-v0")
    rnd = RND(
        scaling_factor=scaling_factor,
        obs_dim=env.observation_space.shape[0],
        train_freq=train_freq,
        out_dim=out_dim,
    )
    model = SAC("MlpPolicy", env, policy_kwargs=dict(net_arch=[256, 256, 256]), surgeon=rnd, verbose=0)
    model.learn(20000)

    x = np.array(model.env.envs[0].all_pos)
    density = kde(x, x)
    entropy = 1 / np.mean(density)
    return entropy


if __name__ == "__main__":
    study = optuna.create_study(
        storage="sqlite:///optuna.db",
        study_name="ContinuousMaze_RND",
        load_if_exists=True,
        direction="maximize",
    )
    study.optimize(objective, n_trials=10)
    importances = optuna.importance.get_param_importances(study)
    print(importances)
