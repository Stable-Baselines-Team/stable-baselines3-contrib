import gymnasium as gym
import pytest
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize

from sb3_contrib import ARS, QRDQN, TQC, TRPO, CrossQ, MaskablePPO
from sb3_contrib.common.envs import InvalidActionEnvDiscrete
from sb3_contrib.common.vec_env import AsyncEval


def test_crossq():
    model = CrossQ(
        "MlpPolicy",
        "Pendulum-v1",
        learning_starts=100,
        verbose=1,
        policy_kwargs=dict(net_arch=[32], renorm_warmup_steps=1),
    )
    model.learn(total_timesteps=110)


@pytest.mark.parametrize("ent_coef", ["auto", 0.01, "auto_0.01"])
def test_tqc(ent_coef):
    model = TQC(
        "MlpPolicy",
        "Pendulum-v1",
        policy_kwargs=dict(net_arch=[64, 64]),
        learning_starts=100,
        verbose=1,
        ent_coef=ent_coef,
    )
    model.learn(total_timesteps=110, progress_bar=True)


@pytest.mark.parametrize("n_critics", [1, 3])
def test_n_critics(n_critics):
    # Test TQC with different number of critics
    model = TQC(
        "MlpPolicy",
        "Pendulum-v1",
        policy_kwargs=dict(net_arch=[64], n_critics=n_critics),
        learning_starts=100,
        verbose=1,
    )
    model.learn(total_timesteps=110)


@pytest.mark.parametrize("model_class", [TQC, CrossQ])
def test_sde(model_class):
    model = model_class(
        "MlpPolicy",
        "Pendulum-v1",
        policy_kwargs=dict(net_arch=[16]),
        use_sde=True,
        learning_starts=100,
        verbose=1,
    )
    model.learn(total_timesteps=110)
    model.policy.reset_noise()
    model.policy.actor.get_std()


def test_qrdqn():
    model = QRDQN(
        "MlpPolicy",
        "CartPole-v1",
        policy_kwargs=dict(n_quantiles=25, net_arch=[64]),
        learning_starts=100,
        buffer_size=500,
        learning_rate=3e-4,
        verbose=1,
    )
    model.learn(total_timesteps=500)


@pytest.mark.parametrize("env_id", ["CartPole-v1", "Pendulum-v1"])
def test_trpo(env_id):
    model = TRPO("MlpPolicy", env_id, n_steps=128, seed=0, policy_kwargs=dict(net_arch=[16]), verbose=1)
    model.learn(total_timesteps=500)


def test_trpo_params():
    # Test with gSDE and subsampling
    model = TRPO(
        "MlpPolicy",
        "Pendulum-v1",
        n_steps=64,
        batch_size=32,
        use_sde=True,
        sub_sampling_factor=4,
        seed=0,
        policy_kwargs=dict(net_arch=dict(pi=[32], vf=[32])),
        verbose=1,
    )
    model.learn(total_timesteps=500)


@pytest.mark.parametrize("env_id", ["CartPole-v1", "Pendulum-v1"])
@pytest.mark.parametrize("policy_str", ["LinearPolicy", "MlpPolicy"])
def test_ars(policy_str, env_id):
    model = ARS(policy_str, env_id, n_delta=1, verbose=1, seed=0)
    model.learn(total_timesteps=500, log_interval=1)


def test_ars_multi_env():
    env = make_vec_env("Pendulum-v1", n_envs=2)
    model = ARS("MlpPolicy", env, n_delta=1)
    model.learn(total_timesteps=250)

    env = VecNormalize(make_vec_env("Pendulum-v1", n_envs=1))
    model = ARS("MlpPolicy", env, n_delta=2, seed=0)
    # with parallelism
    async_eval = AsyncEval([lambda: VecNormalize(make_vec_env("Pendulum-v1", n_envs=1)) for _ in range(2)], model.policy)
    async_eval.seed(0)
    async_eval.set_options()
    model.learn(500, async_eval=async_eval)


@pytest.mark.parametrize("n_top", [2, 8])
def test_ars_n_top(n_top):
    n_delta = 3
    if n_top > n_delta:
        with pytest.warns(UserWarning):
            model = ARS("MlpPolicy", "Pendulum-v1", n_delta=n_delta, n_top=n_top)
            model.learn(total_timesteps=500)
    else:
        model = ARS("MlpPolicy", "Pendulum-v1", n_delta=n_delta, n_top=n_top)
        model.learn(total_timesteps=500)


@pytest.mark.parametrize("model_class", [TQC, QRDQN])
def test_offpolicy_multi_env(model_class):
    if model_class in [TQC]:
        env_id = "Pendulum-v1"
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


@pytest.mark.parametrize("normalize_advantage", [False, True])
def test_advantage_normalization(normalize_advantage):
    env = InvalidActionEnvDiscrete(dim=80, n_invalid_actions=60)
    model = MaskablePPO("MlpPolicy", env, n_steps=64, normalize_advantage=normalize_advantage)
    model.learn(64)


@pytest.mark.parametrize("algo", [TRPO, QRDQN, MaskablePPO])
@pytest.mark.parametrize("stats_window_size", [1, 42])
def test_ep_buffers_stats_window_size(algo, stats_window_size):
    """Set stats_window_size for logging to non-default value and check if
    ep_info_buffer and ep_success_buffer are initialized to the correct length"""
    env = InvalidActionEnvDiscrete() if algo == MaskablePPO else "CartPole-v1"
    model = algo("MlpPolicy", env, stats_window_size=stats_window_size)
    model.learn(total_timesteps=10)
    assert model.ep_info_buffer.maxlen == stats_window_size
    assert model.ep_success_buffer.maxlen == stats_window_size
