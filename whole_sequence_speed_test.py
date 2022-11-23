from sb3_contrib import RecurrentPPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor

import gym


def make_env(**kwargs):
    def _init():
        env = gym.make(**kwargs)
        return env
    return _init

def get_vectorized_envs(n_cpus, **kwargs):
    envs_no_log = SubprocVecEnv([make_env(**kwargs) for _ in range(n_cpus)])
    envs = VecMonitor(envs_no_log)
    return envs


if __name__ == "__main__":
    envs = get_vectorized_envs(n_cpus=32, id="BipedalWalker-v3")

    model = RecurrentPPO(
        "MlpLstmPolicy",
        envs,
        n_steps=2048,
        batch_size=64,
        gae_lambda=0.95,
        gamma=0.999,
        whole_sequences=False,
        n_epochs=10,
        ent_coef=0.0,
        learning_rate=0.0003,
        clip_range=0.18,
        tensorboard_log="temp/",
        verbose=1
    )
    model.learn(500000, tb_log_name="speed test sb3-contrib default 2d batching")


    model = RecurrentPPO(
        "MlpLstmPolicy",
        envs,
        n_steps=2048,
        batch_size=4,
        gae_lambda=0.95,
        gamma=0.999,
        whole_sequences=True,
        n_epochs=10,
        ent_coef=0.0,
        learning_rate=0.0003,
        clip_range=0.18,
        tensorboard_log="temp/",
        verbose=1
    )
    model.learn(500000, tb_log_name="speed test sb3-contrib whole sequence 3d batching")