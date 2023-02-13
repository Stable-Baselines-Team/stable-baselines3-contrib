import random
from typing import Dict, Tuple

import gymnasium as gym
import numpy as np
import pytest
from gymnasium import spaces
from stable_baselines3.common.callbacks import EventCallback, StopTrainingOnNoModelImprovement
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.envs import FakeImageEnv, IdentityEnv, IdentityEnvBox
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.policies import ActorCriticPolicy

from sb3_contrib import MaskablePPO
from sb3_contrib.common.envs import InvalidActionEnvDiscrete, InvalidActionEnvMultiBinary, InvalidActionEnvMultiDiscrete
from sb3_contrib.common.maskable.callbacks import MaskableEvalCallback
from sb3_contrib.common.maskable.evaluation import evaluate_policy
from sb3_contrib.common.maskable.utils import is_masking_supported
from sb3_contrib.common.wrappers import ActionMasker


def make_env():
    return InvalidActionEnvDiscrete(dim=20, n_invalid_actions=10)


class ToDictWrapper(gym.Wrapper):
    """
    Simple wrapper to test MultInputPolicy on Dict obs.
    """

    def __init__(self, env):
        super().__init__(env)
        self.observation_space = spaces.Dict({"obs": self.env.observation_space})

    def reset(self, **kwargs) -> Tuple[Dict[str, np.ndarray], Dict]:
        return {"obs": self.env.reset(seed=kwargs.get("seed", 0))[0]}, {}  # type: ignore[dict-item]

    def step(self, action):
        obs, reward, terminated, truncated, infos = self.env.step(action)
        return {"obs": obs}, reward, terminated, truncated, infos


def test_identity():
    """
    Performance test.
    A randomly initialized model cannot solve that task (score ~=6),
    nor a model without invalid action masking (score ~=30 after training)
    which such a low training budget.
    """
    env = InvalidActionEnvDiscrete(dim=70, n_invalid_actions=55)
    model = MaskablePPO(
        "MlpPolicy",
        env,
        gamma=0.4,
        seed=32,
        verbose=0,
    )
    model.learn(3000)
    evaluate_policy(model, env, n_eval_episodes=20, reward_threshold=90, warn=False)


def test_bootstraping():
    # Max ep length = 100 by default
    env = InvalidActionEnvDiscrete(dim=20, n_invalid_actions=10)
    env = gym.wrappers.TimeLimit(env, 30)
    model = MaskablePPO("MlpPolicy", env, n_steps=64, seed=8)
    model.learn(128)


def test_supports_discrete_action_space():
    """
    No errors using algorithm with an env that has a discrete action space
    """

    env = InvalidActionEnvDiscrete(dim=20, n_invalid_actions=10)
    model = MaskablePPO("MlpPolicy", env, n_steps=64, seed=8)
    model.learn(100)
    evaluate_policy(model, env, warn=False)

    # Mask all actions except the good one, a random model should succeed
    env = InvalidActionEnvDiscrete(dim=20, n_invalid_actions=19)
    model = MaskablePPO("MlpPolicy", env, seed=8)
    evaluate_policy(model, env, reward_threshold=99, warn=False)


def test_supports_multi_discrete_action_space():
    """
    No errors using algorithm with an env that has a multidiscrete action space
    """

    env = InvalidActionEnvMultiDiscrete(dims=[2, 3], n_invalid_actions=1)
    model = MaskablePPO("MlpPolicy", env, n_steps=64, seed=8)
    model.learn(100)
    evaluate_policy(model, env, warn=False)

    # Mask all actions except the good ones, a random model should succeed
    env = InvalidActionEnvMultiDiscrete(dims=[2, 3], n_invalid_actions=3)
    model = MaskablePPO("MlpPolicy", env, seed=8)
    evaluate_policy(model, env, reward_threshold=99, warn=False)


def test_supports_multi_binary_action_space():
    """
    No errors using algorithm with an env that has a multidiscrete action space
    """

    env = InvalidActionEnvMultiBinary(dims=3, n_invalid_actions=1)
    model = MaskablePPO("MlpPolicy", env, n_steps=64, seed=8)
    model.learn(100)
    evaluate_policy(model, env, warn=False)

    # Mask all actions except the good ones, a random model should succeed
    env = InvalidActionEnvMultiBinary(dims=3, n_invalid_actions=3)
    model = MaskablePPO("MlpPolicy", env, seed=8)
    evaluate_policy(model, env, reward_threshold=99, warn=False)


def test_disabling_masking():
    """
    Behave like normal PPO if masking is disabled, which allows for envs that don't provide masks
    """

    env = InvalidActionEnvDiscrete(dim=20, n_invalid_actions=19)
    model = MaskablePPO("MlpPolicy", env, seed=8)
    evaluate_policy(model, env, reward_threshold=99, warn=False)

    # With masking disabled, perfect performance disappears
    with pytest.raises(AssertionError):
        evaluate_policy(model, env, reward_threshold=99, warn=False, use_masking=False)

    # Without masking disabled, learning/evaluation will fail if the env doesn't provide masks
    env = IdentityEnv(dim=2)
    model = MaskablePPO("MlpPolicy", env, n_steps=64, seed=8)
    with pytest.raises(ValueError):
        model.learn(100)
    with pytest.raises(ValueError):
        evaluate_policy(model, env, warn=False)

    model.learn(100, use_masking=False)
    evaluate_policy(model, env, warn=False, use_masking=False)


def test_masked_evaluation():
    """
    Masking can be enabled or disabled for evaluation, but masking should perform better.
    """

    env = InvalidActionEnvDiscrete(dim=20, n_invalid_actions=19)
    model = MaskablePPO("MlpPolicy", env, seed=8)
    masked_avg_rew, _ = evaluate_policy(model, env, warn=False)
    unmasked_avg_rew, _ = evaluate_policy(model, env, warn=False, use_masking=False)
    assert masked_avg_rew > unmasked_avg_rew


def test_supports_multi_envs():
    """
    Learning and evaluation works with VecEnvs
    """

    env = make_vec_env(make_env, n_envs=2)
    assert is_masking_supported(env)
    model = MaskablePPO("MlpPolicy", env, n_steps=256, gamma=0.4, seed=32, verbose=1)
    model.learn(100)
    evaluate_policy(model, env, warn=False)

    env = make_vec_env(IdentityEnv, n_envs=2, env_kwargs={"dim": 2})
    assert not is_masking_supported(env)
    model = MaskablePPO("MlpPolicy", env, n_steps=256, gamma=0.4, seed=32, verbose=1)
    with pytest.raises(ValueError):
        model.learn(100)
    with pytest.raises(ValueError):
        evaluate_policy(model, env, warn=False)
    model.learn(100, use_masking=False)
    evaluate_policy(model, env, warn=False, use_masking=False)


def test_callback(tmp_path):
    """
    No errors using MaskableEvalCallback during learning
    """

    env = make_env()
    eval_env = make_env()
    model = MaskablePPO("MlpPolicy", env, n_steps=64, gamma=0.4, seed=32, verbose=1)
    model.learn(100, callback=MaskableEvalCallback(eval_env, eval_freq=100, warn=False, log_path=tmp_path))

    model.learn(100, callback=MaskableEvalCallback(Monitor(eval_env), eval_freq=100, warn=False), progress_bar=True)


def test_child_callback():
    """
    Stop callback and callback on new best rewards
    """

    env = make_env()
    eval_env = make_env()
    model = MaskablePPO("MlpPolicy", env, n_steps=64, n_epochs=1)
    stop_callback = StopTrainingOnNoModelImprovement(1, 2)
    new_best_mean_callback = EventCallback()
    eval_callback = MaskableEvalCallback(
        Monitor(eval_env),
        eval_freq=64,
        callback_after_eval=stop_callback,
        callback_on_new_best=new_best_mean_callback,
    )
    model.learn(128, callback=eval_callback)
    assert new_best_mean_callback.n_calls > 0
    assert stop_callback.n_calls > 0
    assert stop_callback.n_calls >= new_best_mean_callback.n_calls


def test_maskable_policy_required():
    """
    MaskablePPO requires a policy that subclasses MaskableActorCriticPolicy
    """

    env = make_env()
    with pytest.raises(ValueError):
        MaskablePPO(ActorCriticPolicy, env)


def test_discrete_action_space_required():
    """
    MaskablePPO requires an env with a discrete (ie non-continuous) action space
    """

    env = IdentityEnvBox()
    with pytest.raises(AssertionError):
        MaskablePPO("MlpPolicy", env)


@pytest.mark.parametrize("share_features_extractor", [True, False])
def test_cnn(share_features_extractor):
    def action_mask_fn(env):
        random_invalid_action = random.randrange(env.action_space.n)
        return [i != random_invalid_action for i in range(env.action_space.n)]

    env = FakeImageEnv()
    env = ActionMasker(env, action_mask_fn)

    model = MaskablePPO(
        "CnnPolicy",
        env,
        n_steps=64,
        seed=32,
        verbose=1,
        policy_kwargs=dict(
            features_extractor_kwargs=dict(features_dim=32),
            share_features_extractor=share_features_extractor,
        ),
    )
    model.learn(100)
    evaluate_policy(model, env, warn=False)


def test_dict_obs():
    env = InvalidActionEnvDiscrete(dim=20, n_invalid_actions=10)
    env = ToDictWrapper(env)
    model = MaskablePPO("MultiInputPolicy", env, n_steps=64, seed=8)
    model.learn(64)
    evaluate_policy(model, env, warn=False)

    # Mask all actions except the good one, a random model should succeed
    env = InvalidActionEnvDiscrete(dim=20, n_invalid_actions=19)
    env = ToDictWrapper(env)
    model = MaskablePPO("MultiInputPolicy", env, seed=8)
    evaluate_policy(model, env, reward_threshold=99, warn=False)
    # MultiDiscrete
    env = InvalidActionEnvMultiDiscrete(dims=[2, 3], n_invalid_actions=1)
    env = ToDictWrapper(env)
    model = MaskablePPO("MultiInputPolicy", env, n_steps=32, seed=8)
    model.learn(32)
    # MultiBinary
    env = InvalidActionEnvMultiBinary(dims=3, n_invalid_actions=1)
    env = ToDictWrapper(env)
    model = MaskablePPO("MultiInputPolicy", env, n_steps=32, seed=8)
    model.learn(32)
