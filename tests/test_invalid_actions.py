from sb3_contrib import MaskablePPO
from sb3_contrib.common.envs import InvalidActionEnv
from sb3_contrib.common.maskable.evaluation import evaluate_policy

# TODO: add test for ActionMasker
# from sb3_contrib.common.wrappers import ActionMasker


def test_identity():
    env = InvalidActionEnv(dim=80, n_invalid_actions=60)
    model = MaskablePPO("MlpPolicy", env, gamma=0.4, seed=32, verbose=1)
    model.learn(5000)
    evaluate_policy(model, env, n_eval_episodes=20, reward_threshold=90, warn=False)

    # Mask all actions except the good one, a random model should succeed
    env = InvalidActionEnv(dim=20, n_invalid_actions=19)
    model = MaskablePPO("MlpPolicy", env, seed=8)
    evaluate_policy(model, env, n_eval_episodes=20, reward_threshold=99, warn=False)


def test_callback():
    env = InvalidActionEnv(dim=20, n_invalid_actions=10)
    model = MaskablePPO("MlpPolicy", env, n_steps=64, gamma=0.4, seed=32, verbose=1, create_eval_env=True)
    model.learn(300, eval_freq=100)


# TODO
# def test_cnn():
#     pass
#
# def test_dict_obs():
#     pass
