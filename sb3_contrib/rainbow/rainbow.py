"""
Rainbow DQN.
"""

import logging
import platform
from typing import Any, ClassVar

import torch
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.off_policy_algorithm import OffPolicyAlgorithm
from stable_baselines3.common.policies import BasePolicy
from stable_baselines3.common.type_aliases import GymEnv, Schedule
from torch import Tensor
from torch.nn import functional

from sb3_contrib.rainbow.buffer import PER, PERReplayBufferSamples
from sb3_contrib.rainbow.policy import (
    FactorizedNoisyLinear,
    NatureC51,
    RainbowPolicy,
)

logger = logging.getLogger(__name__)


class Rainbow(OffPolicyAlgorithm):
    """
    Rainbow DQN.

    This implementation combines the main Rainbow DQN extensions, including
    distributional Q-learning, Double DQN, dueling networks, noisy networks,
    prioritized experience replay and multi-step returns.
    """

    policy: RainbowPolicy
    replay_buffer: PER
    per_buffer: PER
    q_net: NatureC51
    q_net_target: NatureC51

    policy_aliases: ClassVar[dict[str, type[BasePolicy]]] = {
        "CnnPolicy": RainbowPolicy,
    }

    def __init__(
        self,
        policy: str | type[BasePolicy],
        env: GymEnv | str,
        target_update_interval: int = 2000,
        per_alpha: float = 0.5,
        gamma: float = 0.99,
        n_steps: int = 3,
        max_grad_norm: float = 10,
        replay_ratio: float = 0.25,
        learning_rate: float | Schedule = 6.25e-5,
        buffer_size: int = 1_000_000,
        learning_starts: int = 20_000,
        batch_size: int = 32,
        rgb: bool = False,
        framestack: int = 4,
        image_width: int = 84,
        image_height: int = 84,
        _init_setup_model: bool = True,
        policy_kwargs: dict[str, Any] | None = None,
        compile_mode: str | None = "max-autotune",
        **kwargs: Any,
    ) -> None:
        """
        :param policy: The policy model to use.
        :param env: The environment to learn from.
        :param target_update_interval: Number of gradient steps between target network
        updates.
        :param per_alpha: Priority exponent used by prioritized experience replay.
        :param gamma: Discount factor.
        :param n_steps: Number of steps used for multi-step returns.
        :param max_grad_norm: Maximum gradient norm.
        :param replay_ratio: Number of gradient updates per environment transition.
        :param learning_rate: Learning rate for the optimizer.
        :param buffer_size: Replay buffer capacity.
        :param learning_starts: Number of transitions collected before training
        starts.
        :param batch_size: Minibatch size for each gradient update.
        :param rgb: Whether observations contain RGB frames.
        :param framestack: Number of frames in each stacked observation.
        :param image_width: Deprecated. Image dimensions are now derived from the
            observation space.
        :param image_height: Deprecated. Image dimensions are now derived from the
            observation space.
        :param _init_setup_model: Whether to build the networks and replay buffer
        during initialization.
        :param policy_kwargs: Additional arguments passed to the policy.
        :param compile_mode: Mode passed to ``torch.compile``. ``None`` disables
        compilation.
        :param kwargs: Additional arguments passed to ``OffPolicyAlgorithm``.
        """

        policy_kwargs = policy_kwargs or {}
        if "linear_size" not in policy_kwargs:
            policy_kwargs["linear_size"] = 512

        super().__init__(
            policy=policy,
            env=env,
            learning_rate=learning_rate,
            buffer_size=buffer_size,
            learning_starts=learning_starts,
            batch_size=batch_size,
            gamma=gamma,
            train_freq=(1, "step"),
            gradient_steps=1,
            policy_kwargs=policy_kwargs,
            support_multi_env=True,
            **kwargs,
        )

        self.compile_mode = compile_mode

        assert replay_ratio > 0, "replay_ratio must be positive"
        self.replay_ratio = replay_ratio

        grads_per_vec_step = replay_ratio * self.n_envs

        if grads_per_vec_step >= 1:
            self.train_freq = (1, "step")
            self.gradient_steps = round(grads_per_vec_step)
        else:
            self.train_freq = (round(1 / grads_per_vec_step), "step")
            self.gradient_steps = 1

        effective = self.gradient_steps / (self.train_freq[0] * self.n_envs)

        if abs(effective - replay_ratio) / replay_ratio > 0.01:
            import warnings

            warnings.warn(
                f"replay_ratio={replay_ratio} is not achievable exactly with "
                f"{self.n_envs} envs; using {effective:.4g} "
                f"(train_freq={self.train_freq[0]} steps, gradient_steps={self.gradient_steps})"
            )
        self.grad_steps = 0
        self.replace_target_cnt = target_update_interval
        self.v_min = -10
        self.v_max = 10
        self.N_ATOMS = 51
        self.n_steps = n_steps
        self.grad_clip = max_grad_norm

        self.rgb = rgb

        self.per_alpha = per_alpha
        self.per_beta = 0.4
        self.framestack = framestack

        self.last_loss: float | None = None

        if _init_setup_model:
            self._setup_model()

    def _setup_model(self) -> None:
        """
        Create the replay buffer, policy networks and optional compiled forwards.
        """
        assert self.env is not None
        # Create the PER buffer before the base setup to avoid allocating the
        # default SB3 replay buffer.
        self.per_buffer = PER(
            size=self.buffer_size,
            observation_space=self.env.observation_space,
            action_space=self.env.action_space,
            device=self.device,
            rgb=self.rgb,
            n_step=self.n_steps,
            n_envs=self.env.num_envs,
            gamma=self.gamma,
            alpha=self.per_alpha,
            beta=self.per_beta,
            framestack=self.framestack,
        )
        self.replay_buffer = self.per_buffer

        super()._setup_model()

        self.q_net = self.policy.q_net
        self.q_net_target = self.policy.q_net_target

        if self.compile_mode is not None and self.device.type == "cuda" and platform.system() == "Linux":
            self.q_net.forward = torch.compile(self.q_net.forward, mode=self.compile_mode)  # type: ignore[method-assign]
            self.q_net_target.forward = torch.compile(self.q_net_target.forward, mode=self.compile_mode)  # type: ignore[method-assign]
        elif self.compile_mode is not None:
            logger.info(f"torch.compile skipped (needs CUDA + Linux; got {self.device.type} + {platform.system()})")

    def _setup_learn(self, total_timesteps: int, *args, **kwargs) -> tuple[int, BaseCallback]:
        """
        Set up training and configure PER beta annealing.

        The PER beta increment is scaled by the number of parallel environments
        because one replay-buffer ``add()`` call stores one transition per
        environment.

        :param total_timesteps: Number of environment transitions requested for training.
        :return: The result of the parent learning setup.
        """
        self.per_buffer.beta_increment = (1.0 - self.per_buffer.beta) * self.n_envs / total_timesteps
        return super()._setup_learn(total_timesteps, *args, **kwargs)

    def train(self, gradient_steps: int, batch_size: int) -> None:
        """
        Perform gradient updates.

        :param gradient_steps: Number of gradient steps to perform.
        :param batch_size: Minibatch size for each gradient update. This argument is
            retained for compatibility with the ``OffPolicyAlgorithm`` interface.
        """
        # Switch to train mode (this affects batch norm / dropout)
        self.policy.set_training_mode(True)
        # Update learning rate according to schedule
        self._update_learning_rate(self.policy.optimizer)

        for _ in range(gradient_steps):
            self._train_call(batch_size)

    @torch.no_grad()
    def reset_noise(self, net: torch.nn.Module) -> None:
        """
        Reset the noise in all noisy linear layers of a network.

        :param net: Network containing the noisy linear layers.
        """
        for m in net.modules():
            if isinstance(m, FactorizedNoisyLinear):
                m.reset_noise()

    @torch.no_grad()
    def disable_noise(self, net: torch.nn.Module) -> None:
        """
        Disable noise in all noisy linear layers of a network.

        :param net: Network containing the noisy linear layers.
        """
        for m in net.modules():
            if isinstance(m, FactorizedNoisyLinear):
                m.disable_noise()

    def replace_target_network(self) -> None:
        """
        Update the target network with the parameters of the online network.
        """
        self.q_net_target.load_state_dict(self.q_net.state_dict())

    def _sample_buffer(self, batch_size: int) -> PERReplayBufferSamples:
        """
        Sample a minibatch from the prioritised experience replay buffer.

        :param batch_size: Number of transitions to sample.
        :return: A minibatch of transitions including PER indices and
        importance-sampling weights.
        """
        return self.replay_buffer.sample(batch_size)

    def _train_call(self, batch_size: int) -> None:
        """
        Perform a single Rainbow DQN gradient update.

        Samples a minibatch from the prioritised replay buffer, computes the
        distributional loss, updates replay priorities and optimises the online
        network. The target network is updated at the configured interval.

        :param batch_size: Number of transitions to sample.
        """
        if self.num_timesteps < self.learning_starts:
            logger.debug("Skipping training: learning_starts not reached")
            return

        # NoisyNet: resample noise on both networks per gradient step
        self.reset_noise(self.q_net)
        self.reset_noise(self.q_net_target)

        if self.grad_steps % self.replace_target_cnt == 0:
            self.replace_target_network()

        batch = self._sample_buffer(batch_size)
        obs = batch.observations
        actions = batch.actions
        rewards = batch.rewards
        next_obs = batch.next_observations
        dones = batch.dones
        weights = batch.weights
        idxs = batch.idxs
        discounts = batch.discounts
        device = self.q_net.device

        obs = obs.to(device)
        actions = actions.to(device)
        rewards = rewards.to(device)
        next_obs = next_obs.to(device)
        dones = dones.to(device)
        weights = weights.to(device)
        discounts = discounts.to(device)

        batch_indices = torch.arange(
            actions.shape[0],
            device=actions.device,
        )

        self.policy.optimizer.zero_grad()
        distr_v, _ = self.q_net.both(obs)
        state_action_values = distr_v[batch_indices, actions]
        state_log_sm_v = functional.log_softmax(state_action_values, dim=1)

        with torch.no_grad():
            # this is using Double DQN
            next_distr_v, _ = self.q_net_target.both(next_obs)
            _, action_qvals_v = self.q_net.both(next_obs)

            next_actions_v = action_qvals_v.max(1)[1]

            next_best_distr_v = next_distr_v[batch_indices, next_actions_v.data]
            next_best_distr_v = self.q_net_target.apply_softmax(next_best_distr_v)
            next_best_distr = next_best_distr_v.detach()

            proj_distr = distr_projection(
                next_best_distr,
                rewards,
                dones,
                self.v_min,
                self.v_max,
                self.N_ATOMS,
                discounts,
            )

            proj_distr_v = proj_distr.to(self.q_net.device)

        state_log_sm_v = state_log_sm_v.to(self.q_net.device)
        kl_per_sample = (-state_log_sm_v * proj_distr_v).sum(dim=1)

        # update PER priorities with the raw (unweighted) KL
        if hasattr(self.replay_buffer, "update_priorities") and idxs is not None:
            self.replay_buffer.update_priorities(idxs, kl_per_sample.detach().cpu().numpy())

        weights = weights.squeeze().to(self.q_net.device)
        loss = (weights * kl_per_sample).mean()
        self.last_loss = loss.item()
        self.logger.record("train/loss", loss.item())
        self.logger.record("train/beta", self.replay_buffer.beta)
        self.logger.record("train/grad_steps", self.grad_steps)

        loss.backward()

        # this wasn't explicitly mentioned in the Rainbow DQN paper, but was used in DQN and was likely kept
        torch.nn.utils.clip_grad_norm_(self.q_net.parameters(), self.grad_clip)
        self.policy.optimizer.step()

        self.grad_steps += 1
        if self.grad_steps % 10_000 == 0:
            logger.info(f"Completed {self.grad_steps} gradient steps")
            logger.info(f"Beta: {self.replay_buffer.beta}")


def distr_projection(
    next_distr: Tensor,
    rewards: Tensor,
    dones: Tensor,
    v_min: float,
    v_max: float,
    n_atoms: int,
    gamma: float | Tensor,
) -> Tensor:
    """
    Project the target distribution onto the categorical support.

    Perform distribution projection aka Categorical Algorithm from the
    "A Distributional Perspective on RL" paper.

    :param next_distr: Probability distribution over atoms for the next states.
    :param rewards: Rewards for the sampled transitions.
    :param dones: Whether the sampled transitions are terminal.
    :param v_min: Minimum value of the categorical support.
    :param v_max: Maximum value of the categorical support.
    :param n_atoms: Number of atoms in the categorical support.
    :param gamma: Discount factor or per-sample bootstrap discounts.
    :return: Projected probability distribution over the categorical support.
    """
    device = next_distr.device
    batch_size = len(rewards)

    rewards = rewards.to(device).float()
    dones = dones.to(device).bool()

    if not torch.is_tensor(gamma):
        gamma = torch.full((batch_size,), float(gamma), device=device)
    else:
        gamma = gamma.to(device).float()

    delta_z = (v_max - v_min) / (n_atoms - 1)
    support = torch.linspace(v_min, v_max, n_atoms, device=device)

    # Tz = r + gamma^k * z; terminal transitions have no bootstrap term,
    # so their whole distribution collapses onto the clamped reward.
    tz = rewards.unsqueeze(1) + gamma.unsqueeze(1) * support.unsqueeze(0)
    tz[dones] = rewards[dones].unsqueeze(1)
    tz = tz.clamp(v_min, v_max)

    b = (tz - v_min) / delta_z
    b_floor = b.floor().long()
    b_ceil = b.ceil().long()

    # When b lands exactly on an atom, shift the pair so interpolation weights
    # still sum to 1 and no probability mass is dropped.
    b_floor[(b_ceil > 0) & (b_floor == b_ceil)] -= 1
    b_ceil[(b_floor < (n_atoms - 1)) & (b_floor == b_ceil)] += 1

    proj_distr = torch.zeros((batch_size, n_atoms), dtype=torch.float32, device=device)
    offset = (torch.arange(batch_size, device=device) * n_atoms).unsqueeze(1)

    proj_distr.view(-1).index_add_(
        0,
        (b_floor + offset).view(-1),
        (next_distr * (b_ceil.float() - b)).view(-1),
    )
    proj_distr.view(-1).index_add_(
        0,
        (b_ceil + offset).view(-1),
        (next_distr * (b - b_floor.float())).view(-1),
    )

    return proj_distr
