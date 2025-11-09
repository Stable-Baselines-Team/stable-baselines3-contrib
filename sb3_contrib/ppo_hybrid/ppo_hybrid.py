from typing import Any, ClassVar, Optional, TypeVar, Union
import warnings
import torch as th
from torch.nn import functional as F
import numpy as np
from gymnasium import spaces
from sb3_contrib.common.hybrid.policies import HybridActorCriticPolicy, HybridActorCriticCnnPolicy, HybridMultiInputActorCriticPolicy
from stable_baselines3.common.on_policy_algorithm import OnPolicyAlgorithm
from stable_baselines3.common.policies import BasePolicy
from stable_baselines3.common.vec_env import VecEnv
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.type_aliases import MaybeCallback, GymEnv, Schedule
from stable_baselines3.common.utils import obs_as_tensor
from sb3_contrib.ppo_hybrid.buffers import HybridActionsRolloutBuffer
from stable_baselines3.common.utils import explained_variance
from stable_baselines3.common.utils import FloatSchedule

SelfHybridPPO = TypeVar("SelfHybridPPO", bound="HybridPPO")


class HybridPPO(OnPolicyAlgorithm):
    policy_aliases: ClassVar[dict[str, type[BasePolicy]]] = {
        "MlpPolicy": HybridActorCriticPolicy,
        "CnnPolicy": HybridActorCriticCnnPolicy,
        "MultiInputPolicy": HybridMultiInputActorCriticPolicy,
    }
    
    rollout_buffer: HybridActionsRolloutBuffer
    policy: HybridActorCriticPolicy
    
    def __init__(
        self,
        policy: Union[str, type[HybridActorCriticPolicy]],
        env: Union[GymEnv, str],    # TODO: check if custom env needed to accept multiple actions
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
        rollout_buffer_class: Optional[type[HybridActionsRolloutBuffer]] = None,     # TODO: check if custom class needed to accept multiple actions
        rollout_buffer_kwargs: Optional[dict[str, Any]] = None,
        target_kl: Optional[float] = None,
        stats_window_size: int = 100,
        tensorboard_log: Optional[str] = None,
        policy_kwargs: Optional[dict[str, Any]] = None,
        verbose: int = 0,
        seed: Optional[int] = None,
        device: Union[th.device, str] = "auto",
        _init_setup_model: bool = True,
    ):
        super().__init__(
            policy=policy,
            env=env,
            learning_rate=learning_rate,
            n_steps=n_steps,
            gamma=gamma,
            gae_lambda=gae_lambda,
            ent_coef=ent_coef,
            vf_coef=vf_coef,
            max_grad_norm=max_grad_norm,
            use_sde=use_sde,
            sde_sample_freq=sde_sample_freq,
            rollout_buffer_class=rollout_buffer_class,
            rollout_buffer_kwargs=rollout_buffer_kwargs,
            stats_window_size=stats_window_size,
            tensorboard_log=tensorboard_log,
            policy_kwargs=policy_kwargs,
            verbose=verbose,
            seed=seed,
            device=device,
            _init_setup_model=_init_setup_model,
            supported_action_spaces=(spaces.Tuple,),
        )

        # Sanity check, otherwise it will lead to noisy gradient and NaN
        # because of the advantage normalization
        if normalize_advantage:
            assert (
                batch_size > 1
            ), "`batch_size` must be greater than 1. See https://github.com/DLR-RM/stable-baselines3/issues/440"
        
        if self.env is not None:
            # Check that `n_steps * n_envs > 1` to avoid NaN
            # when doing advantage normalization
            buffer_size = self.env.num_envs * self.n_steps
            assert buffer_size > 1 or (
                not normalize_advantage
            ), f"`n_steps * n_envs` must be greater than 1. Currently n_steps={self.n_steps} and n_envs={self.env.num_envs}"
            # Check that the rollout buffer size is a multiple of the mini-batch size
            untruncated_batches = buffer_size // batch_size
            if buffer_size % batch_size > 0:
                warnings.warn(
                    f"You have specified a mini-batch size of {batch_size},"
                    f" but because the `RolloutBuffer` is of size `n_steps * n_envs = {buffer_size}`,"
                    f" after every {untruncated_batches} untruncated mini-batches,"
                    f" there will be a truncated mini-batch of size {buffer_size % batch_size}\n"
                    f"We recommend using a `batch_size` that is a factor of `n_steps * n_envs`.\n"
                    f"Info: (n_steps={self.n_steps} and n_envs={self.env.num_envs})"
                )
        
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.clip_range = clip_range
        self.clip_range_vf = clip_range_vf
        self.normalize_advantage = normalize_advantage
        self.target_kl = target_kl

        if _init_setup_model:
            self._setup_model()
    
    def _setup_model(self) -> None:
        self._setup_lr_schedule()
        self.set_random_seed(self.seed)

        if self.rollout_buffer_class is None:
            # TODO: mauybe extend if buffers for Dict obs is implemented
            self.rollout_buffer_class = HybridActionsRolloutBuffer
        
        self.rollout_buffer = self.rollout_buffer_class(
            self.n_steps,
            self.observation_space,
            self.action_space,
            self.device,
            gamma=self.gamma,
            gae_lambda=self.gae_lambda,
            n_envs=self.n_envs,
            **self.rollout_buffer_kwargs,
        )

        self.policy = self.policy_class(
            self.observation_space,
            self.action_space,
            self.lr_schedule,
            **self.policy_kwargs,
        )
        self.policy = self.policy.to(self.device)

        if not isinstance(self.policy, HybridActorCriticPolicy):
            raise ValueError("Policy must subclass HybridActorCriticPolicy")

        # Initialize schedules for policy/value clipping
        self.clip_range = FloatSchedule(self.clip_range)
        if self.clip_range_vf is not None:
            if isinstance(self.clip_range_vf, (float, int)):
                assert self.clip_range_vf > 0, "`clip_range_vf` must be positive, " "pass `None` to deactivate vf clipping"

            self.clip_range_vf = FloatSchedule(self.clip_range_vf)

    def collect_rollouts(
        self,
        env: VecEnv,
        callback: BaseCallback,
        rollout_buffer: HybridActionsRolloutBuffer,
        n_rollout_steps: int,
    ) -> bool:
        """
        Collect experiences using the current policy and fill a ``HybridActionsRolloutBuffer``.
        The term rollout here refers to the model-free notion and should not
        be used with the concept of rollout used in model-based RL or planning.

        :param env: The training environment
        :param callback: Callback that will be called at each step
            (and at the beginning and end of the rollout)
        :param rollout_buffer: Buffer to fill with rollouts
        :param n_rollout_steps: Number of experiences to collect per environment
        :return: True if function returned with at least `n_rollout_steps`
            collected, False if callback terminated rollout prematurely.
        """
        assert self._last_obs is not None, "No previous observation was provided"
        # Switch to eval mode (this affects batch norm / dropout)
        self.policy.set_training_mode(False)

        n_steps = 0
        rollout_buffer.reset()
        # Sample new weights for the state dependent exploration
        if self.use_sde:
            self.policy.reset_noise(env.num_envs)

        callback.on_rollout_start()

        while n_steps < n_rollout_steps:
            if self.use_sde and self.sde_sample_freq > 0 and n_steps % self.sde_sample_freq == 0:
                # Sample a new noise matrix
                self.policy.reset_noise(env.num_envs)
            
            with th.no_grad():
                # Convert to pytorch tensor or to TensorDict
                obs_tensor = obs_as_tensor(self._last_obs, self.device)
                actions_d, actions_c, values, log_probs_d, log_probs_c = self.policy(obs_tensor)
            actions_d = actions_d.cpu().numpy()
            actions_c = actions_c.cpu().numpy()

            # Rescale and perform action
            clipped_actions_c = actions_c

            # action_space is spaces.Tuple[spaces.MultiDiscrete, spaces.Box]
            if self.policy.squash_output:
                # Unscale the actions to match env bounds
                # if they were previously squashed (scaled in [-1, 1])
                clipped_actions_c = self.policy.unscale_action(clipped_actions_c)
            else:
                # Otherwise, clip the actions to avoid out of bound error
                # as we are sampling from an unbounded Gaussian distribution
                clipped_actions_c = np.clip(actions_c, self.action_space[1].low, self.action_space[1].high)
            
            new_obs, rewards, dones, infos = env.step(actions_d, clipped_actions_c)

            self.num_timesteps += env.num_envs

            # Give access to local variables
            callback.update_locals(locals())
            if not callback.on_step():
                return False

            self._update_info_buffer(infos, dones)
            n_steps += 1

            # Reshape in case of discrete action
            if isinstance(self.action_space[0], spaces.Discrete):
                # Reshape in case of discrete action
                actions_d = actions_d.reshape(-1, 1)
            elif isinstance(self.action_space[0], spaces.MultiDiscrete):
                actions_d = actions_d.reshape(-1, self.action_space[0].nvec.shape[0])

            # Handle timeout by bootstrapping with value function
            # see GitHub issue #633
            for idx, done in enumerate(dones):
                if (
                    done
                    and infos[idx].get("terminal_observation") is not None
                    and infos[idx].get("TimeLimit.truncated", False)
                ):
                    terminal_obs = self.policy.obs_to_tensor(infos[idx]["terminal_observation"])[0]
                    with th.no_grad():
                        terminal_value = self.policy.predict_values(terminal_obs)[0]  # type: ignore[arg-type]
                    rewards[idx] += self.gamma * terminal_value
            
            rollout_buffer.add(
                self._last_obs,  # type: ignore[arg-type]
                actions_d,
                clipped_actions_c,
                rewards,
                self._last_episode_starts,  # type: ignore[arg-type]
                values,
                log_probs_d,
                log_probs_c
            )
            self._last_obs = new_obs  # type: ignore[assignment]
            self._last_episode_starts = dones

        with th.no_grad():
            # Compute value for the last timestep
            values = self.policy.predict_values(obs_as_tensor(new_obs, self.device))  # type: ignore[arg-type]

        rollout_buffer.compute_returns_and_advantage(last_values=values, dones=dones)

        callback.update_locals(locals())

        callback.on_rollout_end()

        return True

    def train(self) -> None:
        """
        Update policy using the currently gathered rollout buffer.
        """
        # Switch to train mode (this affects batch norm / dropout)
        self.policy.set_training_mode(True)
        # Update optimizer learning rate
        self._update_learning_rate(self.policy.optimizer)
        # Compute current clip range
        clip_range = self.clip_range(self._current_progress_remaining)  # type: ignore[operator]
        # Optional: clip range for the value function
        if self.clip_range_vf is not None:
            clip_range_vf = self.clip_range_vf(self._current_progress_remaining)  # type: ignore[operator]
        
        entropy_losses = []
        pg_losses_d, pg_losses_c, value_losses = [], [], []
        clip_fractions_d, clip_fractions_c = [], []

        continue_training = True
        # train for n_epochs epochs
        for epoch in range(self.n_epochs):
            approx_kl_divs = []
            # Do a complete pass on the rollout buffer
            for rollout_data in self.rollout_buffer.get(self.batch_size):
                actions_d = rollout_data.actions_d
                actions_c = rollout_data.actions_c
                if isinstance(self.action_space[0], spaces.Discrete):
                    # Reshape in case of discrete action
                    actions_d = actions_d.reshape(-1, 1)
                elif isinstance(self.action_space[0], spaces.MultiDiscrete):
                    actions_d = actions_d.reshape(-1, self.action_space[0].nvec.shape[0])
                
                values, log_probs, entropy = self.policy.evaluate_actions(rollout_data.observations, actions_d, actions_c)
                log_prob_d, log_prob_c = log_probs
                entropy_d, entropy_c = entropy
                values = values.flatten()

                # Normalize advantage
                advantages = rollout_data.advantages
                # Normalization does not make sense if mini batchsize == 1, see GH issue #325
                if self.normalize_advantage and len(advantages) > 1:
                    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
                
                # ratio between old and new policy, for discrete and continuous actions.
                # Should be one at the first iteration
                ratio_d = th.exp(log_prob_d - rollout_data.old_log_prob_d)
                ratio_c = th.exp(log_prob_c - rollout_data.old_log_prob_c)

                # clipped surrogate loss for discrete actions
                policy_loss_d_1 = advantages * ratio_d
                policy_loss_d_2 = advantages * th.clamp(ratio_d, 1 - clip_range, 1 + clip_range)
                policy_loss_d = -th.min(policy_loss_d_1, policy_loss_d_2).mean()

                # clipped surrogate loss for continuous actions
                policy_loss_c_1 = advantages * ratio_c
                policy_loss_c_2 = advantages * th.clamp(ratio_c, 1 - clip_range, 1 + clip_range)
                policy_loss_c = -th.min(policy_loss_c_1, policy_loss_c_2).mean()

                # Logging
                pg_losses_d.append(policy_loss_d.item())
                clip_fraction_d = th.mean((th.abs(ratio_d - 1) > clip_range).float()).item()
                clip_fractions_d.append(clip_fraction_d)
                pg_losses_c.append(policy_loss_c.item())
                clip_fraction_c = th.mean((th.abs(ratio_c - 1) > clip_range).float()).item()
                clip_fractions_c.append(clip_fraction_c)

                # Value loss
                if self.clip_range_vf is None:
                    # No clipping
                    values_pred = values
                else:
                    # Clip the difference between old and new value
                    # NOTE: this depends on the reward scaling
                    values_pred = rollout_data.old_values + th.clamp(
                        values - rollout_data.old_values, -clip_range_vf, clip_range_vf
                    )
                # Value loss using the TD(gae_lambda) target
                value_loss = F.mse_loss(rollout_data.returns, values_pred)
                value_losses.append(value_loss.item())

                # Entropy loss favor exploration
                if entropy is None:
                    # Approximate entropy when no analytical form
                    entropy_loss_d = -th.mean(-log_prob_d)
                    entropy_loss_c = -th.mean(-log_prob_c)
                    entropy_loss = entropy_loss_d + entropy_loss_c
                else:
                    entropy_loss = -th.mean(entropy_d) - th.mean(entropy_c)
                entropy_losses.append(entropy_loss.item())

                # total loss function
                loss = 0.5 * (policy_loss_d + policy_loss_c) + self.ent_coef * entropy_loss + self.vf_coef * value_loss

                # Calculate approximate form of reverse KL Divergence for early stopping
                # see issue #417: https://github.com/DLR-RM/stable-baselines3/issues/417
                # and discussion in PR #419: https://github.com/DLR-RM/stable-baselines3/pull/419
                # and Schulman blog: http://joschu.net/blog/kl-approx.html
                # Note: using the max KL divergence between discrete and continuous actions to stay conservative
                with th.no_grad():
                    log_ratio_d = log_prob_d - rollout_data.old_log_prob_d
                    log_ratio_c = log_prob_c - rollout_data.old_log_prob_c
                    approx_kl_div_d = th.mean((th.exp(log_ratio_d) - 1) - log_ratio_d).cpu().numpy()
                    approx_kl_div_c = th.mean((th.exp(log_ratio_c) - 1) - log_ratio_c).cpu().numpy()
                    approx_kl_div = max(approx_kl_div_d, approx_kl_div_c)
                    approx_kl_divs.append(approx_kl_div)

                if self.target_kl is not None and approx_kl_div > 1.5 * self.target_kl:
                    continue_training = False
                    if self.verbose >= 1:
                        print(f"Early stopping at step {epoch} due to reaching max kl: {approx_kl_div:.2f}")
                    break

                # Optimization step
                self.policy.optimizer.zero_grad()
                loss.backward()
                # Clip grad norm
                th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.policy.optimizer.step()

            self._n_updates += 1
            if not continue_training:
                break
        
        explained_var = explained_variance(self.rollout_buffer.values.flatten(), self.rollout_buffer.returns.flatten())

        # Logs
        self.logger.record("train/entropy_loss", np.mean(entropy_losses))
        self.logger.record("train/policy_gradient_discrete_loss", np.mean(pg_losses_d))
        self.logger.record("train/policy_gradient_continuous_loss", np.mean(pg_losses_c))
        self.logger.record("train/value_loss", np.mean(value_losses))
        self.logger.record("train/approx_kl", np.mean(approx_kl_divs))
        self.logger.record("train/clip_fraction_discrete", np.mean(clip_fractions_d))
        self.logger.record("train/clip_fraction_continuous", np.mean(clip_fractions_c))
        self.logger.record("train/loss", loss.item())
        self.logger.record("train/explained_variance", explained_var)
        if hasattr(self.policy, "log_std"):
            self.logger.record("train/std", th.exp(self.policy.log_std).mean().item())
        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/clip_range", clip_range)
        if self.clip_range_vf is not None:
            self.logger.record("train/clip_range_vf", clip_range_vf)

    def learn(
        self: SelfHybridPPO,
        total_timesteps: int,
        callback: MaybeCallback = None,
        log_interval: int = 1,
        tb_log_name: str = "Hybrid PPO",
        reset_num_timesteps: bool = True,
        progress_bar: bool = False,
    ) -> SelfHybridPPO:
        return super().learn(
            total_timesteps=total_timesteps,
            callback=callback,
            log_interval=log_interval,
            tb_log_name=tb_log_name,
            reset_num_timesteps=reset_num_timesteps,
            progress_bar=progress_bar,
        )
