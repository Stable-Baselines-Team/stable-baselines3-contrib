import pytest
import numpy as np
import torch
import torch as th
from gymnasium import spaces

from sb3_contrib.common.recurrent.buffers import RecurrentDictRolloutBuffer
from sb3_contrib.common.recurrent.type_aliases import RNNStates


import pytest
import numpy as np
import torch as th
from sb3_contrib.common.recurrent.buffers import pad, pad_and_flatten, create_sequencers


@pytest.mark.parametrize("n_envs", [1, 2, 5])
# @pytest.mark.parametrize("n_envs", [1,])
@pytest.mark.parametrize("buffer_size", [100, 200])
@pytest.mark.parametrize(
    "observation_space",
    [
        spaces.Dict({"obs": spaces.Box(low=-1, high=1, shape=(3,), dtype=np.float32)}),
        spaces.Dict({"obs": spaces.MultiDiscrete([1, 1, 1])}),
    ],
)
@pytest.mark.parametrize("action_space", [spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32)])
@pytest.mark.parametrize("hidden_state_size", [64, 128])
@pytest.mark.parametrize("batch_size", [64, 100])
@pytest.mark.parametrize("lstm_num_layers", [1, 3])
def test_recurrent_dict_rollout_buffer_get(
    n_envs,
    buffer_size,
    observation_space,
    action_space,
    hidden_state_size,
    batch_size,
    lstm_num_layers,
):
    single_hs_shape = (lstm_num_layers, n_envs, hidden_state_size)

    hidden_state_shape = (buffer_size, *single_hs_shape)
    rollout_buffer = RecurrentDictRolloutBuffer(
        buffer_size=buffer_size,
        observation_space=observation_space,
        action_space=action_space,
        hidden_state_shape=hidden_state_shape,
        device="cpu",
        gae_lambda=0.95,
        gamma=0.99,
        n_envs=n_envs,
    )
    buffer_size = rollout_buffer.buffer_size
    n_envs = rollout_buffer.n_envs
    assert rollout_buffer.hidden_states_pi.shape == (buffer_size, *single_hs_shape)
    assert rollout_buffer.observations["obs"].shape == (buffer_size, n_envs, 3)
    episode_start = np.array([False], dtype=bool)
    observation = {"obs": np.array([[0.1, 0.2, 0.3]] * n_envs, dtype=np.float32)}
    action = np.array([[0.1, -0.1]] * n_envs, dtype=np.float32)
    reward = np.array([0.5] * n_envs, dtype=np.float32)
    value = th.tensor([0.1] * n_envs)
    log_prob = th.tensor([-0.1] * n_envs)
    lstm_states = RNNStates(
        pi=(th.ones(single_hs_shape), th.ones(single_hs_shape)),
        vf=(th.ones(single_hs_shape), th.ones(single_hs_shape)),
    )
    lstm_states.pi[0][:, 0, :] *= 2
    lstm_states.pi[1][:, 0, :] *= 2
    lstm_states.vf[0][:, 0, :] *= 2
    lstm_states.vf[1][:, 0, :] *= 2
    for _ in range(buffer_size - 1):
        rollout_buffer.add(observation, action, reward, episode_start, value, log_prob, lstm_states=lstm_states)
        episode_start = np.array([False], dtype=bool)
    assert not rollout_buffer.full
    rollout_buffer.add(observation, action, reward, episode_start, value, log_prob, lstm_states=lstm_states)
    assert rollout_buffer.full
    assert rollout_buffer.hidden_states_pi.shape == (buffer_size, *single_hs_shape)
    assert rollout_buffer.cell_states_pi.shape == (buffer_size, *single_hs_shape)
    assert rollout_buffer.hidden_states_vf.shape == (buffer_size, *single_hs_shape)
    assert rollout_buffer.cell_states_vf.shape == (buffer_size, *single_hs_shape)

    for i in range(10):
        sample = next(rollout_buffer.get(batch_size=batch_size))
        assert sample.observations["obs"].shape[-1] == 3, f"obs epoch {i}"
        assert sample.actions.shape[-1] == 2, f"actions epoch {i}"
        assert len(set(torch.unique(sample.lstm_states.pi[0]))) == 1
        # todo: define behavior


