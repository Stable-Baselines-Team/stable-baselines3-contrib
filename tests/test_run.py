import gym
import pytest
from stable_baselines3.common.env_util import make_vec_env

from sb3_contrib import ARS, QRDQN, TQC


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
        policy_kwargs=dict(net_arch=[64]),
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


@pytest.mark.parametrize("env_str", ["CartPole-v1", "Pendulum-v0"])
@pytest.mark.parametrize("policy_str", ["LinearPolicy", "MlpPolicy"])
def test_ars(policy_str, env_str):
    model = ARS(policy_str, env_str, n_delta=1, verbose=1, seed=0)
    model.learn(total_timesteps=500, log_interval=1, eval_freq=250)


@pytest.mark.parametrize("n_envs", [1, 2])
def test_ars_n_env(n_envs):
    from stable_baselines3.common.env_util import make_vec_env

    env = make_vec_env("Pendulum-v0", n_envs=n_envs)

    if n_envs > 1:
        with pytest.warns(UserWarning):
            model = ARS("MlpPolicy", env, n_delta=1)
            model.learn(total_timesteps=500)
    else:
        model = ARS("MlpPolicy", env, n_delta=1)
        model.learn(total_timesteps=500)


@pytest.mark.parametrize("n_top", [2, 8])
def test_ars_n_top(n_top):
    n_delta = 3
    if n_top > n_delta:
        with pytest.warns(UserWarning):
            model = ARS("MlpPolicy", "Pendulum-v0", n_delta=n_delta, n_top=n_top)
            model.learn(total_timesteps=500)
    else:
        model = ARS("MlpPolicy", "Pendulum-v0", n_delta=n_delta, n_top=n_top)
        model.learn(total_timesteps=500)


@pytest.mark.parametrize("model_class", [TQC, QRDQN])
def test_offpolicy_multi_env(model_class):
    if model_class in [TQC]:
        env_id = "Pendulum-v0"
        policy_kwargs = dict(net_arch=[64], n_critics=1)
    else:
        env_id = "CartPole-v1"
        policy_kwargs = dict(net_arch=[64])

    def make_env():
        env = gym.make(env_id)
        # to check that the code handling timeouts runs
        env = gym.wrappers.TimeLimit(env, 50)
        return env

    env = make_vec_env(make_env, n_envs=2)
    model = model_class(
        "MlpPolicy",
        env,
        policy_kwargs=policy_kwargs,
        learning_starts=100,
        buffer_size=10000,
        verbose=0,
        train_freq=5,
    )
    model.learn(total_timesteps=150)
