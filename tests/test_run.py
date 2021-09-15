import pytest

from sb3_contrib import QRDQN, TQC, ARS


@pytest.mark.parametrize("ent_coef", ["auto", 0.01, "auto_0.01"])
def test_tqc(ent_coef):
    model = TQC(
        "MlpPolicy",
        "Pendulum-v0",
        policy_kwargs=dict(net_arch=[64, 64]),
        learning_starts=100,
        verbose=1,
        create_eval_env=True,
        ent_coef=ent_coef,
    )
    model.learn(total_timesteps=300, eval_freq=250)


@pytest.mark.parametrize("n_critics", [1, 3])
def test_n_critics(n_critics):
    # Test TQC with different number of critics
    model = TQC(
        "MlpPolicy",
        "Pendulum-v0",
        policy_kwargs=dict(net_arch=[64], n_critics=n_critics),
        learning_starts=100,
        verbose=1,
    )
    model.learn(total_timesteps=300)


def test_sde():
    model = TQC(
        "MlpPolicy",
        "Pendulum-v0",
        policy_kwargs=dict(net_arch=[64], sde_net_arch=[8]),
        use_sde=True,
        learning_starts=100,
        verbose=1,
    )
    model.learn(total_timesteps=300)
    model.policy.reset_noise()
    model.policy.actor.get_std()


def test_qrdqn():
    model = QRDQN(
        "MlpPolicy",
        "CartPole-v1",
        policy_kwargs=dict(n_quantiles=25, net_arch=[64, 64]),
        learning_starts=100,
        buffer_size=500,
        learning_rate=3e-4,
        verbose=1,
        create_eval_env=True,
    )
    model.learn(total_timesteps=500, eval_freq=250)


@pytest.mark.parametrize("n_delta", [5,8])
def test_ars_n_delta(n_delta):
    from stable_baselines3.common.env_util import make_vec_env
    from stable_baselines3.common.vec_env import SubprocVecEnv

    env = make_vec_env("Pendulum-v0", 4, vec_env_cls=SubprocVecEnv)

    model = ARS(
        "MlpPolicy",
        env,
        n_delta = n_delta
    )

    model.learn(total_timesteps=500)


@pytest.mark.parametrize("n_top", [2, 8])
def test_ars_n_top(n_top):
    from stable_baselines3.common.env_util import make_vec_env
    from stable_baselines3.common.vec_env import SubprocVecEnv
    env = make_vec_env("Pendulum-v0", 4, vec_env_cls=SubprocVecEnv)
    model = ARS(
        "MlpPolicy",
        env,
        n_delta = 3,
        n_top = 4
    )

    model.learn(total_timesteps=500)
