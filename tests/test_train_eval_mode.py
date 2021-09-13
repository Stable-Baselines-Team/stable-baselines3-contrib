import gym
import numpy as np
import pytest
import torch as th
import torch.nn as nn
from stable_baselines3.common.preprocessing import get_flattened_obs_dim
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

from sb3_contrib import QRDQN, TQC


class FlattenBatchNormDropoutExtractor(BaseFeaturesExtractor):
    """
    Feature extract that flatten the input and applies batch normalization and dropout.
    Used as a placeholder when feature extraction is not needed.
    :param observation_space:
    """

    def __init__(self, observation_space: gym.Space):
        super(FlattenBatchNormDropoutExtractor, self).__init__(
            observation_space,
            get_flattened_obs_dim(observation_space),
        )
        self.flatten = nn.Flatten()
        self.batch_norm = nn.BatchNorm1d(self._features_dim)
        self.dropout = nn.Dropout(0.5)

    def forward(self, observations: th.Tensor) -> th.Tensor:
        result = self.flatten(observations)
        result = self.batch_norm(result)
        result = self.dropout(result)
        return result


def clone_batch_norm_stats(batch_norm: nn.BatchNorm1d) -> (th.Tensor, th.Tensor):
    """
    Clone the bias and running mean from the given batch norm layer.
    :param batch_norm:
    :return: the bias and running mean
    """
    return batch_norm.bias.clone(), batch_norm.running_mean.clone()


def clone_qrdqn_batch_norm_stats(model: QRDQN) -> (th.Tensor, th.Tensor, th.Tensor, th.Tensor):
    """
    Clone the bias and running mean from the quantile network and quantile-target network.
    :param model:
    :return: the bias and running mean from the quantile network and quantile-target network
    """
    quantile_net_batch_norm = model.policy.quantile_net.features_extractor.batch_norm
    quantile_net_bias, quantile_net_running_mean = clone_batch_norm_stats(quantile_net_batch_norm)

    quantile_net_target_batch_norm = model.policy.quantile_net_target.features_extractor.batch_norm
    quantile_net_target_bias, quantile_net_target_running_mean = clone_batch_norm_stats(quantile_net_target_batch_norm)

    return quantile_net_bias, quantile_net_running_mean, quantile_net_target_bias, quantile_net_target_running_mean


def clone_tqc_batch_norm_stats(
    model: TQC,
) -> (th.Tensor, th.Tensor, th.Tensor, th.Tensor, th.Tensor, th.Tensor):
    """
    Clone the bias and running mean from the actor and critic networks and critic-target networks.
    :param model:
    :return: the bias and running mean from the actor and critic networks and critic-target networks
    """
    actor_batch_norm = model.actor.features_extractor.batch_norm
    actor_bias, actor_running_mean = clone_batch_norm_stats(actor_batch_norm)

    critic_batch_norm = model.critic.features_extractor.batch_norm
    critic_bias, critic_running_mean = clone_batch_norm_stats(critic_batch_norm)

    critic_target_batch_norm = model.critic_target.features_extractor.batch_norm
    critic_target_bias, critic_target_running_mean = clone_batch_norm_stats(critic_target_batch_norm)

    return (actor_bias, actor_running_mean, critic_bias, critic_running_mean, critic_target_bias, critic_target_running_mean)


CLONE_HELPERS = {
    QRDQN: clone_qrdqn_batch_norm_stats,
    TQC: clone_tqc_batch_norm_stats,
}


def test_qrdqn_train_with_batch_norm():
    model = QRDQN(
        "MlpPolicy",
        "CartPole-v1",
        policy_kwargs=dict(net_arch=[16, 16], features_extractor_class=FlattenBatchNormDropoutExtractor),
        learning_starts=0,
        seed=1,
        tau=0,  # do not clone the target
    )

    (
        quantile_net_bias_before,
        quantile_net_running_mean_before,
        quantile_net_target_bias_before,
        quantile_net_target_running_mean_before,
    ) = clone_qrdqn_batch_norm_stats(model)

    model.learn(total_timesteps=200)

    (
        quantile_net_bias_after,
        quantile_net_running_mean_after,
        quantile_net_target_bias_after,
        quantile_net_target_running_mean_after,
    ) = clone_qrdqn_batch_norm_stats(model)

    assert ~th.isclose(quantile_net_bias_before, quantile_net_bias_after).all()
    assert ~th.isclose(quantile_net_running_mean_before, quantile_net_running_mean_after).all()

    assert th.isclose(quantile_net_target_bias_before, quantile_net_target_bias_after).all()
    assert th.isclose(quantile_net_target_running_mean_before, quantile_net_target_running_mean_after).all()


def test_tqc_train_with_batch_norm():
    model = TQC(
        "MlpPolicy",
        "Pendulum-v0",
        policy_kwargs=dict(net_arch=[16, 16], features_extractor_class=FlattenBatchNormDropoutExtractor),
        learning_starts=0,
        tau=0,  # do not copy the target
        seed=1,
    )

    (
        actor_bias_before,
        actor_running_mean_before,
        critic_bias_before,
        critic_running_mean_before,
        critic_target_bias_before,
        critic_target_running_mean_before,
    ) = clone_tqc_batch_norm_stats(model)

    model.learn(total_timesteps=200)

    (
        actor_bias_after,
        actor_running_mean_after,
        critic_bias_after,
        critic_running_mean_after,
        critic_target_bias_after,
        critic_target_running_mean_after,
    ) = clone_tqc_batch_norm_stats(model)

    assert ~th.isclose(actor_bias_before, actor_bias_after).all()
    assert ~th.isclose(actor_running_mean_before, actor_running_mean_after).all()

    assert ~th.isclose(critic_bias_before, critic_bias_after).all()
    assert ~th.isclose(critic_running_mean_before, critic_running_mean_after).all()

    assert th.isclose(critic_target_bias_before, critic_target_bias_after).all()
    assert th.isclose(critic_target_running_mean_before, critic_target_running_mean_after).all()


@pytest.mark.parametrize("model_class", [QRDQN, TQC])
def test_offpolicy_collect_rollout_batch_norm(model_class):
    if model_class in [QRDQN]:
        env_id = "CartPole-v1"
    else:
        env_id = "Pendulum-v0"

    clone_helper = CLONE_HELPERS[model_class]

    learning_starts = 10
    model = model_class(
        "MlpPolicy",
        env_id,
        policy_kwargs=dict(net_arch=[16, 16], features_extractor_class=FlattenBatchNormDropoutExtractor),
        learning_starts=learning_starts,
        seed=1,
        gradient_steps=0,
        train_freq=1,
    )

    batch_norm_stats_before = clone_helper(model)

    model.learn(total_timesteps=100)

    batch_norm_stats_after = clone_helper(model)

    # No change in batch norm params
    for param_before, param_after in zip(batch_norm_stats_before, batch_norm_stats_after):
        assert th.isclose(param_before, param_after).all()


@pytest.mark.parametrize("model_class", [QRDQN, TQC])
@pytest.mark.parametrize("env_id", ["Pendulum-v0", "CartPole-v1"])
def test_predict_with_dropout_batch_norm(model_class, env_id):
    if env_id == "CartPole-v1":
        if model_class in [TQC]:
            return
    elif model_class in [QRDQN]:
        return

    model_kwargs = dict(seed=1)
    clone_helper = CLONE_HELPERS[model_class]

    if model_class in [QRDQN, TQC]:
        model_kwargs["learning_starts"] = 0
    else:
        model_kwargs["n_steps"] = 64

    policy_kwargs = dict(
        features_extractor_class=FlattenBatchNormDropoutExtractor,
        net_arch=[16, 16],
    )
    model = model_class("MlpPolicy", env_id, policy_kwargs=policy_kwargs, verbose=1, **model_kwargs)

    batch_norm_stats_before = clone_helper(model)

    env = model.get_env()
    observation = env.reset()
    first_prediction, _ = model.predict(observation, deterministic=True)
    for _ in range(5):
        prediction, _ = model.predict(observation, deterministic=True)
        np.testing.assert_allclose(first_prediction, prediction)

    batch_norm_stats_after = clone_helper(model)

    # No change in batch norm params
    for param_before, param_after in zip(batch_norm_stats_before, batch_norm_stats_after):
        assert th.isclose(param_before, param_after).all()
