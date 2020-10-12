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
    model.learn(total_timesteps=500, eval_freq=250)


@pytest.mark.parametrize("n_critics", [1, 3])
def test_n_critics(n_critics):
    # Test TQC with different number of critics
    model = TQC(
        "MlpPolicy", "Pendulum-v0", policy_kwargs=dict(net_arch=[64], n_critics=n_critics), learning_starts=100, verbose=1
    )
    model.learn(total_timesteps=500)

def test_sde():
    model = TQC(
        "MlpPolicy", "Pendulum-v0", policy_kwargs=dict(net_arch=[64]), use_sde=True, learning_starts=100, verbose=1
    )
    model.learn(total_timesteps=500)
