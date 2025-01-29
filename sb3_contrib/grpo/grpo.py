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
from stable_baselines3.common.utils import get_schedule_fn
from stable_baselines3.ppo.ppo import PPO

SelfGRPO = TypeVar("SelfGRPO", bound="GRPO")


class GRPO(PPO):
    """
    A custom implementation of PPO (Proximal Policy Optimization) that integrates
    GRPO-like sampling and reward scaling.

    :param policy: The policy model to use (MlpPolicy, CnnPolicy, ...)
    :param env: The environment to learn from (if registered in Gym, can be str)
    :param learning_rate: The learning rate, it can be a function of the current progress remaining (from 1 to 0)
    :param n_steps: The number of "macro" steps to run for each environment per update
    :param batch_size: Minibatch size
    :param n_epochs: Number of epochs when optimizing the surrogate loss
    :param gamma: Discount factor
    :param gae_lambda: Factor for trade-off of bias vs variance for Generalized Advantage Estimator
    :param clip_range: Clipping parameter, it can be a function of the current progress remaining (from 1 to 0)
    :param clip_range_vf: Clipping parameter for the value function, can be a function or constant
    :param normalize_advantage: Whether to normalize the advantage
    :param ent_coef: Entropy coefficient for the loss calculation
    :param vf_coef: Value function coefficient for the loss calculation
    :param max_grad_norm: The maximum value for the gradient clipping
    :param use_sde: Whether to use generalized State Dependent Exploration (gSDE) instead of action noise exploration
    :param sde_sample_freq: Frequency of sampling new noise matrix when using gSDE
    :param rollout_buffer_class: Rollout buffer class to use. If ``None``, it will be automatically selected.
    :param rollout_buffer_kwargs: Keyword arguments to pass to the rollout buffer on creation
    :param target_kl: Limit the KL divergence between updates
    :param stats_window_size: Window size for rollout logging
    :param tensorboard_log: The log location for TensorBoard (if None, no logging)
    :param policy_kwargs: Additional arguments to be passed to the policy on creation
    :param verbose: Verbosity level (0 for no output, 1 for info messages, 2 for debug messages)
    :param seed: Seed for the pseudo-random generators
    :param device: Device to run on ('cpu', 'cuda', or 'auto')
    :param _init_setup_model: Whether to build the network at the creation of the instance
    :param samples_per_time_step: Number of sub-steps (samples) per macro step
    :param reward_scaling_fn: A callable that accepts a NumPy array of rewards and
        returns a NumPy array of scaled rewards. If ``None``, the default scaling is used.
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
        clip_range_vf: Union[None, float, Schedule] = None,
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
            rollout_buffer_kwargs={},  # Pass an empty dict to avoid conflicts
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
        self.my_rollout_buffer_kwargs = rollout_buffer_kwargs

        # If no scaling function is provided, use the default
        if reward_scaling_fn is None:
            self.reward_scaling_fn = self._default_reward_scaling_fn
        else:
            self.reward_scaling_fn = reward_scaling_fn

        if _init_setup_model:
            self._setup_model()

    def _setup_model(self) -> None:
        """
        Initializes the model components, including:
        - Learning rate schedule
        - Random seed configuration
        - Policy instantiation
        - Clipping range schedule setup
        - Rollout buffer creation

        This method ensures that all essential model elements are properly configured
        before training begins.
        """
        self._setup_lr_schedule()
        self.set_random_seed(self.seed)
        self.policy = self.policy_class(
            self.observation_space, self.action_space, self.lr_schedule, use_sde=self.use_sde, **self.policy_kwargs
        ).to(self.device)

        # Initialize schedules for clipping ranges
        self.clip_range = get_schedule_fn(self.clip_range)
        if self.clip_range_vf is not None:
            self.clip_range_vf = get_schedule_fn(self.clip_range_vf)

        # Create a rollout buffer with a size accounting for samples per step
        n_envs = self.env.num_envs
        buffer_size = self.n_steps * self.samples_per_time_step * n_envs
        self.rollout_buffer = self.rollout_buffer_class(
            buffer_size,
            self.observation_space,
            self.action_space,
            device=self.device,
            gae_lambda=self.gae_lambda,
            gamma=self.gamma,
            n_envs=n_envs,
            **self.my_rollout_buffer_kwargs,
        )

    def collect_rollouts(
        self,
        env,
        callback: MaybeCallback,
        rollout_buffer: RolloutBuffer,
        n_rollout_steps: int,
    ) -> RolloutReturn:
        """
        Collect experiences for `n_rollout_steps`, applying multiple policy samples
        per state before executing an action.

        :param env: The environment instance.
        :param callback: Callback function for logging and monitoring.
        :param rollout_buffer: Buffer for storing experience rollouts.
        :param n_rollout_steps: Number of macro steps to collect per iteration.
        :return: Rollout return object containing processed rewards and episode states.
        """
        self.policy.set_training_mode(False)
        obs = self._last_obs
        rollout_buffer.reset()

        for step in range(n_rollout_steps):
            self.num_timesteps += env.num_envs

            sub_actions = []
            sub_values = []
            sub_log_probs = []

            obs_tensor = th.as_tensor(obs, device=self.device, dtype=th.float32)
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
                rollout_buffer.add(
                    obs,
                    sub_actions[i],
                    repeated_rewards[i],
                    dones,
                    sub_values[i],
                    sub_log_probs[i],
                )

            obs = new_obs

            if callback.on_step() is False:
                break

        scaled_rewards = self.reward_scaling_fn(rollout_buffer.rewards)
        rollout_buffer.rewards[:] = scaled_rewards

        obs_tensor = th.as_tensor(obs, device=self.device, dtype=th.float32)
        with th.no_grad():
            _, last_values, _ = self.policy.forward(obs_tensor)

        rollout_buffer.compute_returns_and_advantage(last_values.flatten(), dones)
        self._last_obs = obs

        return RolloutReturn(rollout_buffer.rewards, rollout_buffer.episode_starts, np.array([False]))

    def _default_reward_scaling_fn(self, rewards: np.ndarray) -> np.ndarray:
        """
        The default reward-scaling method. This is used if no custom function is passed at init.
        Scales rewards by standardizing them, then squashing via tanh.
        """
        r_mean = rewards.mean()
        r_std = rewards.std() + 1e-8
        scaled_rewards = np.tanh((rewards - r_mean) / r_std)
        return scaled_rewards
