import numpy as np
import pytest
import torch
from gymnasium import spaces
from stable_baselines3.common.envs import FakeImageEnv
from stable_baselines3.common.vec_env import SubprocVecEnv

from sb3_contrib.rainbow.buffer import PER, SumTree
from sb3_contrib.rainbow.policy import (
    FactorizedNoisyLinear,
    NatureC51,
    RainbowPolicy,
)
from sb3_contrib.rainbow.rainbow import Rainbow, distr_projection


def make_env(
    image_width: int = 84,
    image_height: int = 84,
) -> FakeImageEnv:
    return FakeImageEnv(
        screen_height=image_height,
        screen_width=image_width,
        n_channels=4,
        channel_first=True,
        discrete=True,
    )


class TestDistributionProjection:
    def test_distribution_projection_sums_to_one(self):
        batch_size = 4
        n_atoms = 51

        next_distr = torch.rand(batch_size, n_atoms)
        next_distr /= next_distr.sum(dim=1, keepdim=True)

        rewards = torch.tensor([0.0, 1.0, -1.0, 0.5])
        dones = torch.tensor([False, False, False, False])

        projected = distr_projection(
            next_distr,
            rewards,
            dones,
            v_min=-10.0,
            v_max=10.0,
            n_atoms=n_atoms,
            gamma=0.99,
        )

        assert torch.allclose(
            projected.sum(dim=1),
            torch.ones(batch_size),
            atol=1e-6,
        )

    def test_distribution_projection_terminal_ignores_next_distribution(self):
        n_atoms = 51

        next_distr_1 = torch.zeros(1, n_atoms)
        next_distr_1[0, 0] = 1.0

        next_distr_2 = torch.zeros(1, n_atoms)
        next_distr_2[0, -1] = 1.0

        rewards = torch.tensor([1.0])
        dones = torch.tensor([True])

        projected_1 = distr_projection(
            next_distr_1,
            rewards,
            dones,
            v_min=-10.0,
            v_max=10.0,
            n_atoms=n_atoms,
            gamma=0.99,
        )

        projected_2 = distr_projection(
            next_distr_2,
            rewards,
            dones,
            v_min=-10.0,
            v_max=10.0,
            n_atoms=n_atoms,
            gamma=0.99,
        )

        assert torch.allclose(projected_1, projected_2, atol=1e-6)
        assert torch.allclose(projected_1.sum(dim=1), torch.ones(1))

    @pytest.mark.parametrize(
        ("reward", "expected_atom"),
        [
            (-100.0, 0),
            (100.0, 50),
        ],
    )
    def test_distribution_projection_clips_to_support(
        self,
        reward,
        expected_atom,
    ):
        n_atoms = 51

        next_distr = torch.full((1, n_atoms), 1.0 / n_atoms)
        rewards = torch.tensor([reward])
        dones = torch.tensor([True])

        projected = distr_projection(
            next_distr,
            rewards,
            dones,
            v_min=-10.0,
            v_max=10.0,
            n_atoms=n_atoms,
            gamma=0.99,
        )

        assert projected.argmax(dim=1).item() == expected_atom
        assert projected[0, expected_atom].item() == pytest.approx(1.0)


class TestNStepReturns:
    def test_n_step_return(self):
        buffer = PER(
            size=100,
            device="cpu",
            n_step=3,
            n_envs=1,
            gamma=0.9,
        )

        rewards = np.array([[1.0, 2.0, 3.0]])
        dones = np.array([[False, False, False]])
        truncs = np.array([[False, False, False]])

        returns, cumulative_dones, discounts = buffer.compute_discounted_rewards_batch(
            rewards,
            dones,
            truncs,
        )

        expected_return = 1.0 + 0.9 * 2.0 + 0.9**2 * 3.0

        assert returns[0] == pytest.approx(expected_return)
        assert not cumulative_dones[0]
        assert discounts[0] == pytest.approx(0.9**3)

    def test_n_step_return_stops_at_termination(self):
        buffer = PER(
            size=100,
            device="cpu",
            n_step=3,
            n_envs=1,
            gamma=0.9,
        )

        rewards = np.array([[1.0, 2.0, 3.0]])
        dones = np.array([[False, True, False]])
        truncs = np.array([[False, False, False]])

        returns, cumulative_dones, _ = buffer.compute_discounted_rewards_batch(
            rewards,
            dones,
            truncs,
        )

        expected_return = 1.0 + 0.9 * 2.0

        assert returns[0] == pytest.approx(expected_return)
        assert cumulative_dones[0]

    def test_n_step_truncation_preserves_bootstrap(self):
        buffer = PER(
            size=100,
            device="cpu",
            n_step=3,
            n_envs=1,
            gamma=0.9,
        )

        rewards = np.array([[1.0, 2.0, 3.0]])
        dones = np.array([[False, False, False]])
        truncs = np.array([[False, True, False]])

        returns, cumulative_dones, discounts = buffer.compute_discounted_rewards_batch(
            rewards,
            dones,
            truncs,
        )

        expected_return = 1.0 + 0.9 * 2.0

        assert returns[0] == pytest.approx(expected_return)
        assert not cumulative_dones[0]
        assert discounts[0] == pytest.approx(0.9**2)

    def test_one_step_discount(self):
        buffer = PER(
            size=100,
            device="cpu",
            n_step=1,
            n_envs=1,
            gamma=0.9,
        )

        dummy_state = np.zeros((4, 84, 84), dtype=np.uint8)

        for _ in range(10):
            buffer.append(
                dummy_state,
                0,
                1.0,
                dummy_state,
                False,
                False,
                stream=0,
            )

        batch = buffer.sample(4)

        assert torch.allclose(
            batch.discounts,
            torch.full((4,), 0.9),
        )


class TestSumTree:
    def test_sum_tree_total(self):
        tree = SumTree(4)

        tree.append(1.0)
        tree.append(2.0)
        tree.append(3.0)
        tree.append(4.0)

        assert tree.total() == pytest.approx(10.0)

    def test_sum_tree_update(self):
        tree = SumTree(4)

        tree.append(1.0)
        tree.append(2.0)
        tree.append(3.0)
        tree.append(4.0)

        leaf_index = tree.tree_start
        tree.update(
            np.array([leaf_index]),
            np.array([5.0]),
        )

        assert tree.sum_tree[leaf_index] == pytest.approx(5.0)
        assert tree.total() == pytest.approx(14.0)

    def test_sum_tree_wraps_at_capacity(self):
        tree = SumTree(3)

        tree.append(1.0)
        tree.append(2.0)
        tree.append(3.0)

        assert tree.full
        assert tree.index == 0

        tree.append(4.0)

        assert tree.index == 1
        assert tree.total() == pytest.approx(9.0)

    def test_sum_tree_find(self):
        tree = SumTree(4)

        tree.append(1.0)
        tree.append(2.0)
        tree.append(3.0)
        tree.append(4.0)

        values = np.array([0.5, 1.5, 3.5, 7.0])
        priorities, data_indices, tree_indices = tree.find(values)

        assert np.array_equal(data_indices, np.array([0, 1, 2, 3]))
        assert np.allclose(priorities, np.array([1.0, 2.0, 3.0, 4.0]))
        assert np.array_equal(
            tree_indices,
            data_indices + tree.tree_start,
        )


class TestFactorizedNoisyLinear:
    def test_reset_noise_changes_noise(self):
        layer = FactorizedNoisyLinear(16, 8)

        layer.reset_noise()

        weight_noise_before = layer.weight_epsilon.clone()
        bias_noise_before = layer.bias_epsilon.clone()

        layer.reset_noise()

        assert not torch.equal(
            weight_noise_before,
            layer.weight_epsilon,
        )
        assert not torch.equal(
            bias_noise_before,
            layer.bias_epsilon,
        )

    def test_disable_noise(self):
        layer = FactorizedNoisyLinear(16, 8)

        layer.reset_noise()
        layer.disable_noise()

        assert torch.all(layer.weight_epsilon == 0)
        assert torch.all(layer.bias_epsilon == 0)

    def test_self_norm_parameters_are_finite(self):
        layer = FactorizedNoisyLinear(
            16,
            8,
            self_norm=True,
        )

        for parameter in layer.parameters():
            assert torch.isfinite(parameter).all()

    def test_noisy_output_changes_after_reset(self):
        layer = FactorizedNoisyLinear(16, 8)
        input_tensor = torch.ones(2, 16)

        layer.reset_noise()
        output_1 = layer(input_tensor)

        layer.reset_noise()
        output_2 = layer(input_tensor)

        assert not torch.equal(output_1, output_2)

    def test_disabled_noise_is_deterministic(self):
        layer = FactorizedNoisyLinear(16, 8)
        input_tensor = torch.ones(2, 16)

        layer.disable_noise()
        output_1 = layer(input_tensor)

        layer.disable_noise()
        output_2 = layer(input_tensor)

        assert torch.equal(output_1, output_2)


class TestRainbowPolicy:

    def test_policy(self):
        obs_space = spaces.Box(low=0, high=255, shape=(4, 84, 84), dtype=np.uint8)
        action_space = spaces.Discrete(6)

        policy = RainbowPolicy(obs_space, action_space, lr_schedule=lambda x: 1e-4)

        obs = torch.randn(2, 4, 84, 84)
        action = policy._predict(obs)

        assert action.shape == (2,)

    def test_deterministic_prediction_disables_noise(self):
        obs_space = spaces.Box(
            low=0,
            high=255,
            shape=(4, 84, 84),
            dtype=np.uint8,
        )
        action_space = spaces.Discrete(6)

        policy = RainbowPolicy(
            obs_space,
            action_space,
            lr_schedule=lambda _: 1e-4,
        )

        observation = torch.zeros(2, 4, 84, 84)

        policy._predict(
            observation,
            deterministic=True,
        )

        for module in policy.q_net.modules():
            if isinstance(module, FactorizedNoisyLinear):
                assert torch.all(module.weight_epsilon == 0)
                assert torch.all(module.bias_epsilon == 0)

    def test_stochastic_prediction_resamples_noise(self):
        obs_space = spaces.Box(
            low=0,
            high=255,
            shape=(4, 84, 84),
            dtype=np.uint8,
        )
        action_space = spaces.Discrete(6)

        policy = RainbowPolicy(
            obs_space,
            action_space,
            lr_schedule=lambda _: 1e-4,
        )

        observation = torch.zeros(2, 4, 84, 84)

        policy._predict(
            observation,
            deterministic=False,
        )

        noisy_layers = [module for module in policy.q_net.modules() if isinstance(module, FactorizedNoisyLinear)]

        assert noisy_layers

        for module in noisy_layers:
            assert torch.any(module.weight_epsilon != 0)


class TestNatureC51:

    def test_network_output_shape(self):
        net = NatureC51(
            4,
            6,
            device="cpu",
        )

        observations = torch.zeros(2, 4, 84, 84)
        q_values = net.qvals(observations)

        assert q_values.shape == (2, 6)

    def test_network_supports_different_image_size(self):
        net = NatureC51(
            4,
            6,
            device="cpu",
            image_width=96,
            image_height=96,
        )

        observations = torch.zeros(2, 4, 96, 96)
        q_values = net.qvals(observations)

        assert q_values.shape == (2, 6)

    def test_categorical_probabilities_sum_to_one(self):
        net = NatureC51(
            4,
            6,
            device="cpu",
        )

        observations = torch.zeros(2, 4, 84, 84)

        logits = net(observations)
        probabilities = net.apply_softmax(logits)

        assert probabilities.shape == (2, 6, 51)
        assert torch.allclose(
            probabilities.sum(dim=2),
            torch.ones(2, 6),
            atol=1e-6,
        )


class TestRainbowMultiEnvBuffer:
    def test_environment_streams_are_independent(self):
        buffer = PER(
            size=100,
            device="cpu",
            n_step=3,
            n_envs=2,
            gamma=0.99,
        )

        state_0 = np.zeros((4, 84, 84), dtype=np.uint8)
        state_1 = np.full((4, 84, 84), 255, dtype=np.uint8)

        buffer.append(
            state_0,
            0,
            1.0,
            state_0,
            False,
            False,
            stream=0,
        )

        buffer.append(
            state_1,
            1,
            2.0,
            state_1,
            False,
            False,
            stream=1,
        )

        assert buffer.state_buffer[0] != buffer.state_buffer[1]
        assert buffer.reward_buffer[0] != buffer.reward_buffer[1]
        assert len(buffer.state_buffer[0]) == len(buffer.state_buffer[1])
        assert len(buffer.reward_buffer[0]) == len(buffer.reward_buffer[1])

    def test_termination_only_resets_terminated_stream(self):
        buffer = PER(
            size=100,
            device="cpu",
            n_step=3,
            n_envs=2,
            gamma=0.99,
        )

        state = np.zeros((4, 84, 84), dtype=np.uint8)

        buffer.append(
            state,
            0,
            1.0,
            state,
            False,
            False,
            stream=0,
        )

        buffer.append(
            state,
            0,
            1.0,
            state,
            False,
            False,
            stream=1,
        )

        buffer.append(
            state,
            0,
            1.0,
            state,
            True,
            False,
            stream=0,
        )

        assert buffer.state_buffer[0] == []
        assert buffer.reward_buffer[0] == []

        assert buffer.state_buffer[1]
        assert buffer.reward_buffer[1]


class TestRainbowBufferInterface:

    def test_replay_sample_structure(self):
        env = make_env()

        model = Rainbow(
            RainbowPolicy,
            env,
            learning_starts=10,
            batch_size=8,
            buffer_size=1000,
        )

        model.learn(200)

        batch = model.replay_buffer.sample(8)

        assert hasattr(batch, "observations")
        assert hasattr(batch, "actions")
        assert hasattr(batch, "next_observations")
        assert hasattr(batch, "dones")
        assert hasattr(batch, "rewards")
        assert hasattr(batch, "idxs")
        assert hasattr(batch, "weights")

    def test_replay_sample_shapes_and_types(self):
        env = make_env()

        model = Rainbow(
            RainbowPolicy,
            env,
            learning_starts=10,
            batch_size=8,
            buffer_size=1000,
        )

        model.learn(200)

        batch = model.replay_buffer.sample(8)

        assert batch.observations.shape[0] == 8
        assert batch.actions.shape[0] == 8
        assert batch.next_observations.shape[0] == 8
        assert batch.rewards.shape[0] == 8
        assert batch.dones.shape[0] == 8

        assert batch.observations.dtype == torch.float32
        assert batch.actions.dtype == torch.int64
        assert batch.rewards.dtype == torch.float32

    def test_replay_sample_device(self):
        env = make_env()

        model = Rainbow(
            RainbowPolicy,
            env,
            learning_starts=10,
            batch_size=8,
            buffer_size=1000,
        )

        model.learn(200)

        batch = model.replay_buffer.sample(8)

        assert batch.observations.device.type == model.device.type
        assert batch.next_observations.device.type == model.device.type
        assert batch.weights.device.type == model.device.type


class TestRainbowBufferPER:

    def test_replay(self):
        buffer = PER(size=100, device="cpu", n_step=3, n_envs=1, gamma=0.99)

        dummy_state = np.zeros((4, 84, 84), dtype=np.uint8)

        for _ in range(50):
            buffer.append(
                dummy_state,
                0,
                1.0,
                dummy_state,
                False,
                False,
                stream=0,
            )

        batch = buffer.sample(8)

        assert batch.observations.shape[0] == 8
        assert batch.actions.shape[0] == 8

    def test_per_priority_update(self):
        buffer = PER(
            size=100,
            device="cpu",
            n_step=3,
            n_envs=1,
            gamma=0.99,
            alpha=0.5,
        )

        dummy_state = np.zeros((4, 84, 84), dtype=np.uint8)

        for _ in range(20):
            buffer.append(
                dummy_state,
                0,
                1.0,
                dummy_state,
                False,
                False,
                stream=0,
            )

        batch = buffer.sample(8)
        priorities = np.full(len(batch.idxs), 4.0)

        buffer.update_priorities(batch.idxs, priorities)

        expected = (priorities + buffer.eps) ** buffer.alpha

        assert np.allclose(
            buffer.st.sum_tree[batch.idxs],
            expected,
        )

    def test_per_weights_bounds(self):
        env = make_env()

        model = Rainbow(
            RainbowPolicy,
            env,
            learning_starts=10,
            batch_size=32,
            buffer_size=2000,
        )

        model.learn(500)

        batch = model.replay_buffer.sample(32)

        assert torch.all(batch.weights > 0)
        assert torch.all(batch.weights <= 1.0)

    def test_per_beta_annealing(self):
        env = make_env()

        model = Rainbow(
            RainbowPolicy,
            env,
            learning_starts=10,
            batch_size=8,
            buffer_size=1000,
        )

        buffer = model.replay_buffer
        initial_beta = buffer.beta

        model.learn(500)

        assert buffer.beta > initial_beta
        assert buffer.beta <= 1.0


class TestRainbowTraining:

    def test_gradients_flow(self):
        env = make_env()

        model = Rainbow(
            RainbowPolicy,
            env,
            learning_starts=10,
            batch_size=8,
            buffer_size=1000,
        )

        model.learn(200)

        params = list(model.q_net.parameters())
        has_grad = any(p.grad is not None for p in params)

        assert has_grad

    def test_parameters_update(self):
        env = make_env()

        model = Rainbow(
            RainbowPolicy,
            env,
            learning_starts=10,
            batch_size=8,
            buffer_size=1000,
        )

        params_before = [p.clone().detach() for p in model.q_net.parameters()]

        model.learn(500)

        params_after = list(model.q_net.parameters())

        changed = False
        for p_before, p_after in zip(params_before, params_after, strict=True):
            if not torch.equal(p_before, p_after):
                changed = True
                break

        assert changed

    def test_loss_is_finite(self):
        env = make_env()

        model = Rainbow(
            RainbowPolicy,
            env,
            learning_starts=10,
            batch_size=8,
            buffer_size=1000,
        )

        model.learn(200)

        loss = model.last_loss

        assert not np.isnan(loss)
        assert not np.isinf(loss)

    def test_beta_increment_uses_number_of_environments(self):
        env = make_env()

        model = Rainbow(
            RainbowPolicy,
            env,
            learning_starts=10,
            batch_size=8,
            buffer_size=1000,
            device="cpu",
            compile_mode=None,
        )

        initial_beta = model.per_buffer.beta

        model._setup_learn(total_timesteps=1000)

        expected_increment = (1.0 - initial_beta) * model.n_envs / 1000

        assert model.per_buffer.beta_increment == pytest.approx(expected_increment)


def test_subproc_vec_env():
    n_envs = 2

    env = SubprocVecEnv(
        [
            lambda: FakeImageEnv(
                screen_height=84,
                screen_width=84,
                n_channels=4,
                channel_first=True,
                discrete=True,
            )
            for _ in range(n_envs)
        ]
    )

    model = Rainbow(
        "CnnPolicy",
        env,
        learning_starts=10,
        batch_size=8,
        buffer_size=1000,
        device="cpu",
        compile_mode=None,
    )

    model.learn(100)

    assert model.n_envs == n_envs

    env.close()
