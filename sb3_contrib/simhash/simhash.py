from typing import Optional

import torch as th
from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.surgeon import Surgeon
from stable_baselines3.common.type_aliases import ReplayBufferSamples
from stable_baselines3.common.utils import get_device
from stable_baselines3.common.vec_env.base_vec_env import VecEnv


class SimHasher:
    """
    SimHash retrieves a binary code of observation.
        φ(obs) = sign(A*obs) ∈ {−1, 1}^granularity
    where A is a granularity × obs_size matrix with i.i.d. entries drawn from a standard
    Gaussian distribution (mean=0, std=1).
    :param obs_size: Size of observation.
    :type obs_size: int
    :param granularity: Granularity. Higher value lead to fewer collisions
        and are thus more likely to distinguish states.
    :type granularity: int
    """

    def __init__(self, obs_size: int, granularity: int) -> None:
        size = (granularity, obs_size)
        self.A = th.normal(mean=th.zeros(size), std=th.ones(size)).to(get_device("auto"))

    def __call__(self, obs: th.Tensor) -> th.Tensor:
        return th.sign(th.matmul(self.A, obs.T)).T


class SimHash(Surgeon):
    """
    SimHash motivation.
        intrinsic_reward += β / √n(φ(obs))
    where β>0 is the bonus coefficient, φ the hash function and n the count.
    Paper: https://arxiv.org/abs/1611.04717

    :param
    :param granularity: granularity; Higher value lead to fewer collisions
        and are thus more likely to distinguish states
    :param beta: The bonus coefficient (scaling factor)
    :param pure_exploration: If True, ignore environment reward
    :param env: The env, when normalization.
    """

    def __init__(self, granularity: int, beta: float, pure_exploration: bool = False, env: Optional[VecEnv] = None) -> None:
        self.buffer = None  # type: ReplayBuffer
        self.env = env
        self.beta = beta
        self.granularity = granularity
        self.pure_exploration = pure_exploration

    def set_buffer(self, buffer: ReplayBuffer) -> None:
        self.buffer = buffer
        self.hasher = SimHasher(self.buffer.obs_shape[0], self.granularity)

    def modify_reward(self, replay_data: ReplayBufferSamples) -> ReplayBufferSamples:
        next_obs_hash = self.hasher(replay_data.next_observations)
        pos = self.buffer.buffer_size if self.buffer.full else self.buffer.pos
        all_next_observations = th.from_numpy(self.buffer._normalize_obs(self.buffer.next_observations[:pos], self.env))
        all_next_observations = all_next_observations.view(-1, self.buffer.obs_shape[0]).to(get_device("auto"))
        all_hashes = self.hasher(all_next_observations)
        unique, all_counts = th.unique(all_hashes, dim=0, return_counts=True)
        count = th.zeros(next_obs_hash.shape[0]).to(get_device("auto"))
        for k, hash in enumerate(next_obs_hash):
            idx = (unique == hash).all(1)
            count[k] = all_counts[idx]
        intrinsic_reward = self.beta / th.sqrt(count)
        new_rewards = (1 - self.pure_exploration) * replay_data.rewards + intrinsic_reward.unsqueeze(1)
        new_replay_data = ReplayBufferSamples(
            replay_data.observations, replay_data.actions, replay_data.next_observations, replay_data.dones, new_rewards
        )
        self.logger.record("SimHash/rew_intr_mean", intrinsic_reward.mean().item())
        self.logger.record("SimHash/rew_extr_mean", replay_data.rewards.mean().item())
        return new_replay_data
