import gym
import gym_continuous_maze
import numpy as np
import optuna

from stable_baselines3 import SAC
from sb3_contrib import Surprise
from sb3_contrib.silver.silver import kde


def objective(trial: optuna.Study):
    feature_dim = trial.suggest_categorical("feature_dim", [8, 16, 32, 64, 128, 256, 512])
    eta_0 = trial.suggest_categorical("eta_0", [0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0])
    train_freq = trial.suggest_categorical("train_freq", [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024])
    lr = trial.suggest_categorical("lr", [1e-5, 1e-4, 1e-3, 1e-2])
    batch_size = trial.suggest_categorical("batch_size", [64, 128, 256, 512, 1024])

    env = gym.make("ContinuousMaze-v0")
    surprise = Surprise(
        env.observation_space.shape[0],
        env.action_space.shape[0],
        feature_dim=feature_dim,
        eta_0=eta_0,
        train_freq=train_freq,
        lr=lr,
        batch_size=batch_size,
    )
    model = SAC(
        "MlpPolicy",
        env,
        policy_kwargs=dict(net_arch=[256, 256, 256]),
        surgeon=surprise,
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
        study_name="ContinuousMaze_Surprise",
        load_if_exists=True,
        direction="maximize",
    )
    study.optimize(objective, n_trials=10)
    importances = optuna.importance.get_param_importances(study)
    print(importances)
