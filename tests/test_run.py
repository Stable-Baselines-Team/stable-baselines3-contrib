import pytest

from sb3_contrib import TQC


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
    model.learn(total_timesteps=1000, eval_freq=500)


@pytest.mark.parametrize("n_critics", [1, 3])
def test_n_critics(n_critics):
    # Test TQC with different number of critics
    model = TQC(
        "MlpPolicy", "Pendulum-v0", policy_kwargs=dict(net_arch=[64, 64], n_critics=n_critics), learning_starts=100, verbose=1
    )
    model.learn(total_timesteps=1000)


# "CartPole-v1"
# @pytest.mark.parametrize("env_id", ["MountainCarContinuous-v0"])
# def test_cmaes(env_id):
#     if CMAES is None:
#         return
#     model = CMAES("MlpPolicy", env_id, seed=0, policy_kwargs=dict(net_arch=[64]), verbose=1, create_eval_env=True)
#     model.learn(total_timesteps=50000, eval_freq=10000)


@pytest.mark.parametrize("strategy", ["exp", "bc", "binary"])
@pytest.mark.parametrize("reduce", ["mean", "max"])
def test_crr(tmp_path, strategy, reduce):
    model = TQC(
        "MlpPolicy",
        "Pendulum-v0",
        policy_kwargs=dict(net_arch=[64]),
        learning_starts=1000,
        verbose=1,
        create_eval_env=True,
        action_noise=None,
        use_sde=False,
    )

    model.learn(total_timesteps=1000, eval_freq=0)
    for n_action_samples in [1, 2, -1]:
        model.pretrain(gradient_steps=32, batch_size=32, n_action_samples=n_action_samples, strategy=strategy, reduce=reduce)
