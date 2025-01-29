from typing import (Any, Callable, ClassVar, Dict, Optional, Type, TypeVar,
                    Union)

import numpy as np
import torch as th
from stable_baselines3.common.buffers import RolloutBuffer
from stable_baselines3.common.policies import (ActorCriticCnnPolicy,
                                               ActorCriticPolicy, BasePolicy,
                                               MultiInputActorCriticPolicy)
from stable_baselines3.common.type_aliases import (GymEnv, MaybeCallback,
                                                   RolloutReturn, Schedule)
from stable_baselines3.ppo.ppo import PPO

SelfGRPO = TypeVar("SelfGRPO", bound="GRPO")


class GRPO(PPO):
    """
    Generalized Policy Reward Optimization (GPRO) implementation,
    extending PPO with sub-sampling per step and a customizable
    reward scaling function.

    :param policy: The policy model to use (MlpPolicy, CnnPolicy, ...)
    :param env: The environment to learn from
    :param learning_rate: The learning rate (constant or schedule function)
    :param n_steps: The number of "macro" steps per update
    :param batch_size: Minibatch size for training
    :param n_epochs: Number of training epochs per update
    :param gamma: Discount factor for future rewards
    :param gae_lambda: Generalized Advantage Estimator parameter
    :param clip_range: Clipping range for policy updates
    :param clip_range_vf: Clipping range for value function updates
    :param normalize_advantage: Whether to normalize advantages
    :param ent_coef: Entropy coefficient (exploration regularization)
    :param vf_coef: Value function loss coefficient
    :param max_grad_norm: Max gradient norm for clipping
    :param use_sde: Whether to use generalized State-Dependent Exploration
    :param sde_sample_freq: Frequency of sampling new noise matrix for gSDE
    :param rollout_buffer_class: Rollout buffer class (default: RolloutBuffer)
    :param rollout_buffer_kwargs: Additional arguments for the rollout buffer
    :param target_kl: Maximum KL divergence threshold (None to disable)
    :param stats_window_size: Window size for logging statistics
    :param tensorboard_log: TensorBoard log directory (None to disable logging)
    :param policy_kwargs: Additional arguments for policy instantiation
    :param verbose: Verbosity level (0: no output, 1: info, 2: debug)
    :param seed: Random seed for reproducibility
    :param device: Device for computation ('cpu', 'cuda', or 'auto')
    :param _init_setup_model: Whether to build the network on instantiation
    :param samples_per_time_step: Number of sub-samples per macro step
    :param reward_scaling_fn: Custom reward scaling function (default is tanh)
    """

    policy_aliases: ClassVar[Dict[str, Type[BasePolicy]]] = {
        "MlpPolicy": ActorCriticPolicy,
        "CnnPolicy": ActorCriticCnnPolicy,
        "MultiInputPolicy": MultiInputActorCriticPolicy,
    }

    def __init__(
        self,
        policy: Union[str, Type[ActorCriticPolicy]],
        env: Union[GymEnv, str],
        learning_rate: Union[float, Schedule] = 3e-4,
        n_steps: int = 2048,
        batch_size: int = 64,
        n_epochs: int = 10,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_range: Union[float, Schedule] = 0.2,
        clip_range_vf: Optional[Union[float, Schedule]] = None,
        normalize_advantage: bool = True,
        ent_coef: float = 0.0,
        vf_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        use_sde: bool = False,
        sde_sample_freq: int = -1,
        rollout_buffer_class: Optional[Type[RolloutBuffer]] = None,
        rollout_buffer_kwargs: Optional[Dict[str, Any]] = None,
        target_kl: Optional[float] = None,
        stats_window_size: int = 100,
        tensorboard_log: Optional[str] = None,
        policy_kwargs: Optional[Dict[str, Any]] = None,
        verbose: int = 0,
        seed: Optional[int] = None,
        device: Union[th.device, str] = "auto",
        _init_setup_model: bool = True,
        samples_per_time_step: int = 5,
        reward_scaling_fn: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    ):
        if rollout_buffer_kwargs is None:
            rollout_buffer_kwargs = {}
        if rollout_buffer_class is None:
            rollout_buffer_class = RolloutBuffer

        super().__init__(
            policy=policy,
            env=env,
            learning_rate=learning_rate,
            n_steps=n_steps,
            batch_size=batch_size,
            n_epochs=n_epochs,
            gamma=gamma,
            gae_lambda=gae_lambda,
            clip_range=clip_range,
            clip_range_vf=clip_range_vf,
            normalize_advantage=normalize_advantage,
            ent_coef=ent_coef,
            vf_coef=vf_coef,
            max_grad_norm=max_grad_norm,
            use_sde=use_sde,
            sde_sample_freq=sde_sample_freq,
            rollout_buffer_class=rollout_buffer_class,
            rollout_buffer_kwargs=rollout_buffer_kwargs,
            target_kl=target_kl,
            stats_window_size=stats_window_size,
            tensorboard_log=tensorboard_log,
            policy_kwargs=policy_kwargs,
            verbose=verbose,
            seed=seed,
            device=device,
            _init_setup_model=False,
        )

        self.samples_per_time_step = samples_per_time_step
        self.reward_scaling_fn = reward_scaling_fn or self._grpo_scale_rewards

        if _init_setup_model:
            self._setup_model()

    def collect_rollouts(
        self,
        env,
        callback: MaybeCallback,
        rollout_buffer: RolloutBuffer,
        n_rollout_steps: int,
    ) -> RolloutReturn:
        """
        Collect experiences over `n_rollout_steps`,
        performing multiple sub-samples per state.

        Each environment step is sampled `self.samples_per_time_step`
        times before advancing.

        :param env: The training environment
        :param callback: Callback function for logging and stopping conditions
        :param rollout_buffer: Buffer to store collected rollouts
        :param n_rollout_steps: Number of macro steps to collect
        :return: Rollout return object containing rewards and episode states
        """
        self.policy.set_training_mode(False)
        obs = self._last_obs
        rollout_buffer.reset()

        for step in range(n_rollout_steps):
            self.num_timesteps += env.num_envs

            obs_tensor = th.as_tensor(obs, device=self.device, dtype=th.float32)
            sub_actions, sub_values, sub_log_probs = [], [], []

            for _ in range(self.samples_per_time_step):
                with th.no_grad():
                    actions, values, log_probs = self.policy.forward(obs_tensor)

                sub_actions.append(actions.cpu().numpy())
                sub_values.append(values)
                sub_log_probs.append(log_probs)

            final_action = sub_actions[-1]
            new_obs, rewards, dones, infos = env.step(final_action)

            repeated_rewards = np.tile(rewards, (self.samples_per_time_step, 1))

            for i in range(self.samples_per_time_step):
                rollout_buffer.add(obs, sub_actions[i], repeated_rewards[i], dones, sub_values[i], sub_log_probs[i])

            obs = new_obs

            if callback.on_step() is False:
                break

        rollout_buffer.rewards[:] = self.reward_scaling_fn(rollout_buffer.rewards)

        return RolloutReturn(rollout_buffer.rewards, rollout_buffer.episode_starts, np.array([False]))

    def _grpo_scale_rewards(self, rewards: np.ndarray) -> np.ndarray:
        """Normalize rewards using tanh-based scaling."""
        r_mean = rewards.mean()
        r_std = rewards.std() + 1e-8
        return np.tanh((rewards - r_mean) / r_std)
