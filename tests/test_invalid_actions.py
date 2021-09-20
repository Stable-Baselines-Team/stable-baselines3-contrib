import random

import pytest
from stable_baselines3.common.envs import FakeImageEnv, IdentityEnvBox
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.policies import ActorCriticPolicy

from sb3_contrib import MaskablePPO
from sb3_contrib.common.envs import InvalidActionEnvDiscrete
from sb3_contrib.common.maskable.callbacks import MaskableEvalCallback
from sb3_contrib.common.maskable.evaluation import evaluate_policy
from sb3_contrib.common.wrappers import ActionMasker


def make_env():
    return InvalidActionEnvDiscrete(dim=20, n_invalid_actions=10)


def test_supports_multi_envs():
    """
    Learning and evaluation works with VecEnvs
    """

    env = make_vec_env(make_env, n_envs=2)
    model = MaskablePPO("MlpPolicy", env, n_steps=256, gamma=0.4, seed=32, verbose=1)
    model.learn(100)
    evaluate_policy(model, env, warn=False)


def test_identity():
    """
    Masking forces choices when there's only one valid action
    """

    # Mask all actions except the good one, a random model should succeed
    env = InvalidActionEnvDiscrete(dim=20, n_invalid_actions=19)
    model = MaskablePPO("MlpPolicy", env, seed=8)
    evaluate_policy(model, env, reward_threshold=99, warn=False)


def test_callback():
    """
    No errors using MaskableEvalCallback during learning
    """

    env = make_env()
    eval_env = make_env()
    model = MaskablePPO("MlpPolicy", env, n_steps=64, gamma=0.4, seed=32, verbose=1)
    model.learn(100, callback=MaskableEvalCallback(eval_env, eval_freq=100, warn=False))

    model.learn(100, callback=MaskableEvalCallback(Monitor(eval_env), eval_freq=100, warn=False))


def test_masked_evaluation():
    """
    Masking can be enabled or disabled for evaluation, but masking should perform better.
    """

    env = InvalidActionEnvDiscrete(dim=20, n_invalid_actions=19)
    model = MaskablePPO("MlpPolicy", env, seed=8)
    masked_avg_rew, _ = evaluate_policy(model, env, warn=False)
    unmasked_avg_rew, _ = evaluate_policy(model, env, warn=False, use_masking=False)
    assert masked_avg_rew >= unmasked_avg_rew


def test_maskable_policy_required():
    """
    MaskablePPO requires a policy that subclasses MaskableActorCriticPolicy
    """

    env = make_env()
    with pytest.raises(ValueError):
        model = MaskablePPO(ActorCriticPolicy, env)


def test_discrete_action_space_required():
    """
    MaskablePPO requires an env with a discrete (ie non-continuous) action space
    """

    env = IdentityEnvBox()
    with pytest.raises(AssertionError):
        model = MaskablePPO("MlpPolicy", env)


def test_cnn():
    def action_mask_fn(env):
        random_invalid_action = random.randrange(env.action_space.n)
        return [i != random_invalid_action for i in range(env.action_space.n)]

    env = FakeImageEnv()
    env = ActionMasker(env, action_mask_fn)

    model = MaskablePPO("CnnPolicy", env, n_steps=256, seed=32, verbose=1)
    model.learn(100)
    evaluate_policy(model, env, warn=False)


# TODO works with discrete, multidiscrete, multibinary action spaces

# TODO?
# def test_dict_obs():
#     pass
# def test_multi_env_eval():
#     pass
