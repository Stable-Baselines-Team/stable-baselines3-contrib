import pytest
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.vec_env import VecNormalize

from sb3_contrib import ARS, QRDQN, TQC
from sb3_contrib.common.vec_env import AsyncEval

N_STEPS_TRAINING = 500
SEED = 0
ARS_MULTI = "ars_multi"


@pytest.mark.parametrize("algo", [ARS, QRDQN, TQC, ARS_MULTI])
def test_deterministic_training_common(algo):
    results = [[], []]
    rewards = [[], []]
    # Smaller network
    kwargs = {"policy_kwargs": dict(net_arch=[64])}
    env_id = "Pendulum-v0"
    if algo == ARS_MULTI:
        algo = ARS
        ars_multi = True
    else:
        ars_multi = False

    if algo in [TQC]:
        kwargs.update({"action_noise": NormalActionNoise(0.0, 0.1), "learning_starts": 100, "train_freq": 4})
    else:
        if algo == QRDQN:
            env_id = "CartPole-v1"
            kwargs.update({"learning_starts": 100, "target_update_interval": 100})
        elif algo == ARS:
            kwargs.update({"n_delta": 2})

    for i in range(2):
        model = algo("MlpPolicy", env_id, seed=SEED, **kwargs)

        learn_kwargs = {"total_timesteps": N_STEPS_TRAINING}
        if ars_multi:
            learn_kwargs["async_eval"] = AsyncEval(
                [lambda: VecNormalize(make_vec_env(env_id, seed=SEED, n_envs=1)) for _ in range(2)],
                model.policy,
            )

        model.learn(**learn_kwargs)
        env = model.get_env()
        obs = env.reset()
        for _ in range(100):
            action, _ = model.predict(obs, deterministic=False)
            obs, reward, _, _ = env.step(action)
            results[i].append(action)
            rewards[i].append(reward)
    assert sum(results[0]) == sum(results[1]), results
    assert sum(rewards[0]) == sum(rewards[1]), rewards
