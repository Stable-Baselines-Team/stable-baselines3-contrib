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

# Fixes to make - env reset being called twice
# Fix Step Counting - Implement fix into env

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
    :param reward_function: The reward function that is required to recompute sampled rewards (env.reward_func)
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
        reward_function: Optional[Callable[[np.ndarray, np.ndarray], float]] = None,
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
        self.reward_function = reward_function 
        self.my_rollout_buffer_kwargs = rollout_buffer_kwargs

        # If no scaling function is provided, use the default
        if reward_scaling_fn is None:
            self.reward_scaling_fn = self._default_reward_scaling_fn
        else:
            self.reward_scaling_fn = reward_scaling_fn

        # ---- NEW: Wrap the reward function to correct the step count ----
        if self.reward_function is not None:
            original_reward_function = self.reward_function
            def wrapped_reward_function(obs, actions):
                # Temporarily adjust the environment's step count by halving it.
                env_instance = self.env
                if hasattr(env_instance, 'current_step'):
                    orig_step = env_instance.current_step
                    env_instance.current_step = orig_step // 2
                    rew = original_reward_function(obs, actions)
                    env_instance.current_step = orig_step
                else:
                    rew = original_reward_function(obs, actions)
                return rew
            self.reward_function = wrapped_reward_function

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

        # Compute effective steps (macro steps) as n_steps divided by samples_per_time_step.
        n_envs = self.env.num_envs
        effective_steps = self.n_steps // self.samples_per_time_step
        buffer_size = effective_steps * n_envs
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
            Collect experiences for n_rollout_steps (macro steps) using multiple action samples per macro step.
            For each macro step, several samples are generated to compute a transformed reward and advantage.
            Only one representative transition (using, e.g., the last sample) is stored in the rollout buffer.
            """
            # Helper function to safely call env.reset() and ensure it returns (obs, info)
            def safe_reset(env):
                result = env.reset()
                if isinstance(result, tuple):
                    return result
                else:
                    return result, {}
            self.policy.set_training_mode(False)
            obs = self._last_obs
            rollout_buffer.reset()

            last_transition = None
            # Iterate for exactly n_rollout_steps macro-steps
            macro_steps = self.n_steps // self.samples_per_time_step
            for step in range(macro_steps):
                self.num_timesteps += env.num_envs

                sample_actions = []
                sample_values = []
                sample_log_probs = []
                sample_rewards = []

                obs_tensor = th.as_tensor(obs, device=self.device, dtype=th.float32)
                # Collect multiple samples at the current state for advantage estimation
                for _ in range(self.samples_per_time_step):
                    with th.no_grad():
                        actions, values, log_probs = self.policy.forward(obs_tensor)
                    actions_np = actions.cpu().numpy()
                    sample_actions.append(actions_np)
                    # Keep values and log_probs as tensors (detach to avoid gradients)
                    sample_values.append(values.detach())
                    sample_log_probs.append(log_probs.detach())
                    if self.reward_function is not None:
                        sample_r = self.reward_function(obs, actions_np)
                    else:
                        raise TypeError("Your reward function must be passed for recomputing rewards")
                    sample_rewards.append(sample_r)

                # Step the environment using one chosen sample (e.g. the last sample)
                final_action = sample_actions[-1]
                reset_done = False
                try:
                    new_obs, _, dones, truncated, infos = env.step(final_action)
                except ValueError:
                    new_obs, _, dones, infos = env.step(final_action)
                    truncated = np.zeros_like(dones)
                if (np.any(dones) or np.any(truncated)) and (not reset_done):
                    new_obs, _ = safe_reset(env)
                    reset_done = True

                # Compute the bootstrapped value for the next state
                obs_tensor_next = th.as_tensor(new_obs, device=self.device, dtype=th.float32)
                with th.no_grad():
                    _, last_values, _ = self.policy.forward(obs_tensor_next)
                V_next = last_values.cpu().numpy().flatten()

                # Compute V(s) for the current state using the last sample's value estimate,
                # converting the tensor to numpy for arithmetic.
                V_current = sample_values[-1].flatten().cpu().numpy()

                # Compute advantages for each sample
                sample_advantages = []
                for r in sample_rewards:
                    transformed_r = np.tanh(r)  # Reward transformation (e.g., tanh)
                    A_t = transformed_r + self.gamma * V_next - V_current
                    sample_advantages.append(A_t)

                # Adjust advantages locally: subtract the mean over samples
                adv_mean = np.mean(sample_advantages, axis=0)
                adjusted_advantages = [A_t - adv_mean for A_t in sample_advantages]

                # Choose a representative sample transition (here, we pick the final sample)
                rep_action = sample_actions[-1]
                rep_value = sample_values[-1]
                rep_log_prob = sample_log_probs[-1]
                rep_advantage = adjusted_advantages[-1]

                # Store this single transition in the rollout buffer
                rollout_buffer.add(
                    obs,
                    rep_action,
                    rep_advantage,
                    dones,
                    rep_value,
                    rep_log_prob,
                )
                last_transition = (obs, rep_action, rep_advantage, dones, rep_value, rep_log_prob)

                obs = new_obs

            # If the rollout buffer is not full, fill the remaining slots with the last transition
            if not rollout_buffer.full and last_transition is not None:
                while not rollout_buffer.full:
                    rollout_buffer.add(*last_transition)
                    
            # Force the buffer to be full if needed
            if rollout_buffer.pos < rollout_buffer.buffer_size:
                rollout_buffer.full = True
            
            # Apply reward scaling if provided
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
