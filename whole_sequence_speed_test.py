import gym
import numpy as np
import torch.nn as nn
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor, VecNormalize

class MaskVelocityWrapper(gym.ObservationWrapper):
    """
    Gym environment observation wrapper used to mask velocity terms in
    observations. The intention is the make the MDP partially observable.
    Adapted from https://github.com/LiuWenlin595/FinalProject.
    :param env: Gym environment
    """

    # Supported envs
    velocity_indices = {
        "CartPole-v1": np.array([1, 3]),
        "MountainCar-v0": np.array([1]),
        "MountainCarContinuous-v0": np.array([1]),
        "Pendulum-v1": np.array([2]),
        "LunarLander-v2": np.array([2, 3, 5]),
        "LunarLanderContinuous-v2": np.array([2, 3, 5]),
    }

    def __init__(self, env: gym.Env):
        super().__init__(env)

        env_id: str = env.unwrapped.spec.id
        # By default no masking
        self.mask = np.ones_like((env.observation_space.sample()))
        try:
            # Mask velocity
            self.mask[self.velocity_indices[env_id]] = 0.0
        except KeyError:
            raise NotImplementedError(f"Velocity masking not implemented for {env_id}")

    def observation(self, observation: np.ndarray) -> np.ndarray:
        return observation * self.mask


def make_env(mask_vel=False, **kwargs):
    def _init():
        env = gym.make(**kwargs)
        if mask_vel:
            env = MaskVelocityWrapper(env)
        return env
    return _init


def get_vectorized_envs(n_cpus, **kwargs):
    envs_no_log = SubprocVecEnv([make_env(**kwargs) for _ in range(n_cpus)])
    envs = VecNormalize(VecMonitor(envs_no_log))
    return envs


if __name__ == "__main__":
    ############################################################
    # BipedalWalker-v3
    ############################################################

    n_cpus = 32
    envs = get_vectorized_envs(n_cpus=n_cpus, id="BipedalWalker-v3")
    model = RecurrentPPO(
        "MlpLstmPolicy",
        envs,
        n_steps=256,
        batch_size=256,
        gae_lambda=0.95,
        gamma=0.999,
        whole_sequences=False,
        n_epochs=10,
        ent_coef=0.0,
        learning_rate=0.0003,
        clip_range=0.18,
        policy_kwargs={
            "ortho_init": False,
            "activation_fn": nn.ReLU,
            "lstm_hidden_size": 64,
            "enable_critic_lstm": True,
            "net_arch": [dict(pi=[64], vf=[64])]
        },
        tensorboard_log="temp/",
        verbose=1
    )
    model.learn(5e6, tb_log_name="BipedalWalker-v3_sb3_standard")


    model = RecurrentPPO(
        "MlpLstmPolicy",
        envs,
        n_steps=256,
        batch_size=4,
        gae_lambda=0.95,
        gamma=0.999,
        whole_sequences=True, # This sets use of whole sequence batching
        n_epochs=10,
        ent_coef=0.0,
        learning_rate=0.0003,
        clip_range=0.18,
        policy_kwargs={
            "ortho_init": False,
            "activation_fn": nn.ReLU,
            "lstm_hidden_size": 64,
            "enable_critic_lstm": True,
            "net_arch": [dict(pi=[64], vf=[64])]
        },
        tensorboard_log="temp/",
        verbose=1
    )
    model.learn(5e6, tb_log_name="BipedalWalker-v3_whole_sequences")


    ############################################################
    # PendulumNoVel
    ############################################################

    n_cpus = 4
    envs = get_vectorized_envs(n_cpus=n_cpus, id="Pendulum-v1", mask_vel=True)

    model = RecurrentPPO(
        "MlpLstmPolicy",
        envs,
        n_steps=1024,
        # batch_size=256,
        gae_lambda=0.95,
        gamma=0.9,
        whole_sequences=False,
        n_epochs=10,
        ent_coef=0.0,
        learning_rate=0.001,
        clip_range=0.2,
        policy_kwargs={
            "ortho_init": False,
            "activation_fn": nn.ReLU,
            "lstm_hidden_size": 64,
            "enable_critic_lstm": True,
            "net_arch": [dict(pi=[64], vf=[64])]
        },
        tensorboard_log="temp/",
        verbose=1
    )
    model.learn(2e5, tb_log_name="PendulumNoVel-v1_sb3_standard")

    for batch_size in [2, 4, 8]:
        model = RecurrentPPO(
            "MlpLstmPolicy",
            envs,
            n_steps=1024,
            batch_size=batch_size,
            gae_lambda=0.95,
            gamma=0.9,
            whole_sequences=True, # This sets use of whole sequence batching
            n_epochs=10,
            ent_coef=0.0,
            learning_rate=0.001,
            clip_range=0.2,
            policy_kwargs={
                "ortho_init": False,
                "activation_fn": nn.ReLU,
                "lstm_hidden_size": 64,
                "enable_critic_lstm": True,
                "net_arch": [dict(pi=[64], vf=[64])]
            },
            tensorboard_log="temp/",
            verbose=1
        )
        model.learn(2e5, tb_log_name=f"PendulumNoVel-v1_whole_sequences_batch_size{batch_size}")