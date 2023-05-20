import numpy as np
import pytest
from stable_baselines3.common.envs import IdentityEnv, IdentityEnvMultiBinary, IdentityEnvMultiDiscrete
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import DummyVecEnv

from sb3_contrib import QRDQN, TRPO

DIM = 4


@pytest.mark.parametrize("model_class", [QRDQN, TRPO])
@pytest.mark.parametrize("env", [IdentityEnv(DIM), IdentityEnvMultiDiscrete(DIM), IdentityEnvMultiBinary(DIM)])
def test_discrete(model_class, env):
    vec_env = DummyVecEnv([lambda: env])
    kwargs = {}
    n_steps = 1500
    if model_class == QRDQN:
        kwargs = dict(
            learning_starts=0,
            policy_kwargs=dict(n_quantiles=25, net_arch=[32]),
            target_update_interval=10,
            train_freq=2,
            batch_size=256,
        )
        n_steps = 1500
        # DQN only support discrete actions
        if isinstance(env, (IdentityEnvMultiDiscrete, IdentityEnvMultiBinary)):
            return
    elif n_steps == TRPO:
        kwargs = dict(n_steps=256, cg_max_steps=5)

    model = model_class("MlpPolicy", vec_env, learning_rate=1e-3, gamma=0.4, seed=0, **kwargs).learn(n_steps)

    evaluate_policy(model, vec_env, n_eval_episodes=20, reward_threshold=90, warn=False)
    obs = vec_env.reset()

    assert np.shape(model.predict(obs)[0]) == np.shape(obs)
