from collections.abc import Generator
import numpy as np
from stable_baselines3.common.buffers import RolloutBuffer
from gymnasium import spaces
from typing import Optional, Union, NamedTuple
import torch as th
from stable_baselines3.common.preprocessing import get_obs_shape
from stable_baselines3.common.utils import get_device
from stable_baselines3.common.vec_env import VecNormalize


class HybridActionsRolloutBufferSamples(NamedTuple):
    observations: th.Tensor
    actions_d: th.Tensor
    actions_c: th.Tensor
    old_values: th.Tensor
    old_log_prob_d: th.Tensor
    old_log_prob_c: th.Tensor
    advantages: th.Tensor
    returns: th.Tensor


def get_action_dim(action_space: spaces.Tuple) -> tuple[int, int]:
    """
    Get the dimension of the action space,
    assumed to be the one of HybridPPO (Tuple[MultiDiscrete, Box]).

    :param action_space: Tuple action space containing MultiDiscrete and Box spaces
    :return: (dim_d, dim_c) where dim_d is the discrete action dimension and dim_c the continuous action dimension.
    """
    assert isinstance(action_space, spaces.Tuple), "Action space must be a Tuple space"
    assert len(action_space.spaces) == 2, "Action space must contain exactly 2 subspaces"
    assert isinstance(action_space.spaces[0], spaces.MultiDiscrete), "First subspace must be MultiDiscrete"
    assert isinstance(action_space.spaces[1], spaces.Box), "Second subspace must be Box"
    return (
        len(action_space.nvec),  # discrete action dimension
        int(np.prod(action_space.shape)) # continuous action dimension
    )


class HybridActionsRolloutBuffer(RolloutBuffer):
    """
    Rollout buffer for hybrid action spaces (discrete + continuous).
    Stores separate actions and log probabilities for discrete and continuous parts.
    """

    actions: dict[str, np.ndarray]

    def __init__(
            self,
            buffer_size: int,
            observation_space: spaces.Space,
            action_space: spaces.Tuple,  # Type[spaces.MultiDiscrete, spaces.Box]
            device: Union[th.device, str] = "auto",
            gae_lambda: float = 1,
            gamma: float = 0.99,
            n_envs: int = 1,
    ):
        # NOTE: it would be nice to use RolloutBuffer.__init__(), but BaseBuffer calls
        # get_action_dim which is not compatible with Tuple action spaces.
        
        # BaseBuffer's constructor code (excluding call to ABC.__init__())
        self.buffer_size = buffer_size
        self.observation_space = observation_space
        self.action_space = action_space
        self.obs_shape = get_obs_shape(observation_space)  # type: ignore[assignment]

        self.action_dim = get_action_dim(action_space)
        self.pos = 0
        self.full = False
        self.device = get_device(device)
        self.n_envs = n_envs

        # RolloutBuffer's constructor code
        self.gae_lambda = gae_lambda
        self.gamma = gamma
        self.generator_ready = False
        self.reset()
    
    def reset(self) -> None:
        super().reset()
        # override actions and log_probs to handle hybrid actions
        self.actions = {
            'd': np.zeros((self.buffer_size, self.n_envs, self.action_dim[0]), dtype=np.float32),
            'c': np.zeros((self.buffer_size, self.n_envs, self.action_dim[1]), dtype=np.float32)
        }
        self.log_probs = {
            'd': np.zeros((self.buffer_size, self.n_envs), dtype=np.float32),
            'c': np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        }
    
    def add(
        self,
        obs: np.ndarray,
        action_d: np.ndarray,
        action_c: np.ndarray,
        reward: np.ndarray,
        episode_start: np.ndarray,
        value: th.Tensor,
        log_prob_d: th.Tensor,
        log_prob_c: th.Tensor,
    ) -> None:
        """
        :param obs: Observation
        :param action_d: Discrete action
        :param action_c: Continuous action
        :param reward:
        :param episode_start: Start of episode signal.
        :param value: estimated value of the current state
            following the current policy.
        :param log_prob_d: log probabilities of the discrete action following the current policy.
        :param log_prob_c: log probabilities of the continuous action following the current policy.
        """
        # Reshape 0-d tensor to avoid error
        if len(log_prob_d.shape) == 0:
            log_prob_d = log_prob_d.reshape(-1, 1)
        if len(log_prob_c.shape) == 0:
            log_prob_c = log_prob_c.reshape(-1, 1)
        
        # copied from RolloutBuffer:
        # Reshape needed when using multiple envs with discrete observations
        # as numpy cannot broadcast (n_discrete,) to (n_discrete, 1)
        if isinstance(self.observation_space, spaces.Discrete):
            obs = obs.reshape((self.n_envs, *self.obs_shape))
        
        # Adapted from RolloutBuffer:
        # Reshape to handle multi-dim and discrete action spaces, see GH #970 #1392
        action_d = action_d.reshape((self.n_envs, self.action_dim))
        action_c = action_c.reshape((self.n_envs, self.action_dim))
        
        self.observations[self.pos] = np.array(obs)
        self.actions['d'][self.pos] = np.array(action_d)
        self.actions['c'][self.pos] = np.array(action_c)
        self.rewards[self.pos] = np.array(reward)
        self.episode_starts[self.pos] = np.array(episode_start)
        self.values[self.pos] = value.clone().cpu().numpy().flatten()
        self.log_probs['d'][self.pos] = log_prob_d.clone().cpu().numpy()
        self.log_probs['c'][self.pos] = log_prob_c.clone().cpu().numpy()
        self.pos += 1
        if self.pos == self.buffer_size:
            self.full = True
    
    def get(self, batch_size: Optional[int] = None) -> Generator[HybridActionsRolloutBufferSamples, None, None]:
        assert self.full, ""
        indices = np.random.permutation(self.buffer_size * self.n_envs)
        # Prepare the data
        if not self.generator_ready:
            _tensor_names = [
                "observations",
                "actions_d",
                "actions_c",
                "values",
                "log_probs_d",
                "log_probs_c",
                "advantages",
                "returns",
            ]

            for tensor in _tensor_names:
                self.__dict__[tensor] = self.swap_and_flatten(self.__dict__[tensor])
            self.generator_ready = True
        
        # Return everything, don't create minibatches
        if batch_size is None:
            batch_size = self.buffer_size * self.n_envs

        start_idx = 0
        while start_idx < self.buffer_size * self.n_envs:
            yield self._get_samples(indices[start_idx : start_idx + batch_size])
            start_idx += batch_size
    
    def _get_samples(
            self,
            batch_inds: np.ndarray,
            env: Optional[VecNormalize] = None  # TODO: check type hint
        ) -> HybridActionsRolloutBufferSamples:
        data = (
            self.observations[batch_inds],
            self.actions['d'][batch_inds],
            self.actions['c'][batch_inds],
            self.values[batch_inds].flatten(),
            self.log_probs['d'][batch_inds].flatten(),
            self.log_probs['c'][batch_inds].flatten(),
            self.advantages[batch_inds].flatten(),
            self.returns[batch_inds].flatten(),
        )
        return HybridActionsRolloutBufferSamples(*tuple(map(self.to_torch, data)))


# TODO: implement
# class HybridActionsRolloutBuffer(HybridActionsRolloutBuffer):
