import copy
import warnings
from typing import Any, Dict, Optional, Type, Union

import numpy as np
import torch
import torch as th
from gym import spaces
from torch.distributions import kl_divergence

from sb3_contrib.common.policies import ActorCriticPolicy
from sb3_contrib.common.utils import flat_grad, cg_solver
from stable_baselines3.common.on_policy_algorithm import OnPolicyAlgorithm
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, Schedule
from stable_baselines3.common.utils import explained_variance


class TRPO(OnPolicyAlgorithm):
    """
    Trust Region Policy Optimization (TRPO)

    Paper: https://arxiv.org/abs/1502.05477
    Code: This implementation borrows code from OpenAI Spinning Up (https://github.com/openai/spinningup/)
        and Stable Baselines (TRPO from https://github.com/hill-a/stable-baselines)

    Introduction to TRPO: https://spinningup.openai.com/en/latest/algorithms/trpo.html

    :param policy: The policy model to use (MlpPolicy, CnnPolicy, ...)
    :param env: The environment to learn from (if registered in Gym, can be str)
    :param learning_rate: The learning rate, it can be a function
        of the current progress remaining (from 1 to 0)
    :param n_steps: The number of steps to run for each environment per update
        (i.e. rollout buffer size is n_steps * n_envs where n_envs is number of environment copies running in parallel)
        NOTE: n_steps * n_envs must be greater than 1 (because of the advantage normalization)
        See https://github.com/pytorch/pytorch/issues/29372
    :param batch_size: Minibatch size
    :param n_epochs: Number of epoch when optimizing the surrogate loss
    :param gamma: Discount factor
    :param cg_max_steps: maximum number of steps in the Conjugate Gradient algoritgm
        for computing the Hessian vector product
    :param cg_damping: damping in the Hessian vector product computation
    :param ls_alpha: step-size reduction factor for the line-search (i.e. theta_new = theta + alpha^i * step)
    :param ls_steps: maximum number of steps in the line-search
    :param gae_lambda: Factor for trade-off of bias vs variance for Generalized Advantage Estimator
    :param ent_coef: Entropy coefficient for the loss calculation
    :param vf_coef: Value function coefficient for the loss calculation
    :param max_grad_norm: The maximum value for the gradient clipping
    :param use_sde: Whether to use generalized State Dependent Exploration (gSDE)
        instead of action noise exploration (default: False)
    :param sde_sample_freq: Sample a new noise matrix every n steps when using gSDE
        Default: -1 (only sample at the beginning of the rollout)
    :param target_kl: Limit the KL divergence between updates,
        because the clipping is not enough to prevent large update
        see issue #213 (cf https://github.com/hill-a/stable-baselines/issues/213)
        By default, there is no limit on the kl div.
    :param tensorboard_log: the log location for tensorboard (if None, no logging)
    :param create_eval_env: Whether to create a second environment that will be
        used for evaluating the agent periodically. (Only available when passing string for the environment)
    :param policy_kwargs: additional arguments to be passed to the policy on creation
    :param verbose: the verbosity level: 0 no output, 1 info, 2 debug
    :param seed: Seed for the pseudo random generators
    :param device: Device (cpu, cuda, ...) on which the code should be run.
        Setting it to auto, the code will be run on the GPU if possible.
    :param _init_setup_model: Whether or not to build the network at the creation of the instance
    """

    def __init__(
        self,
        policy: Union[str, Type[ActorCriticPolicy]],
        env: Union[GymEnv, str],
        learning_rate: Union[float, Schedule] = 3e-4,
        n_steps: int = 2048,
        batch_size: Optional[int] = 64,
        n_epochs: int = 10,
        gamma: float = 0.99,
        cg_max_steps: int = 10,
        cg_damping: float = 0.1,
        ls_alpha: float = 0.99,
        ls_steps: int = 10,
        gae_lambda: float = 0.95,
        ent_coef: float = 0.0,
        vf_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        use_sde: bool = False,
        sde_sample_freq: int = -1,
        target_kl: float = 0.01,
        tensorboard_log: Optional[str] = None,
        create_eval_env: bool = False,
        policy_kwargs: Optional[Dict[str, Any]] = None,
        verbose: int = 0,
        seed: Optional[int] = None,
        device: Union[th.device, str] = "auto",
        _init_setup_model: bool = True,
    ):

        super(TRPO, self).__init__(
            policy,
            env,
            learning_rate=learning_rate,
            n_steps=n_steps,
            gamma=gamma,
            gae_lambda=gae_lambda,
            ent_coef=ent_coef,
            vf_coef=vf_coef,
            max_grad_norm=max_grad_norm,
            use_sde=use_sde,
            sde_sample_freq=sde_sample_freq,
            tensorboard_log=tensorboard_log,
            policy_kwargs=policy_kwargs,
            verbose=verbose,
            device=device,
            create_eval_env=create_eval_env,
            seed=seed,
            _init_setup_model=False,
            supported_action_spaces=(
                spaces.Box,
                spaces.Discrete,
                spaces.MultiDiscrete,
                spaces.MultiBinary,
            ),
        )

        # Sanity check, otherwise it will lead to noisy gradient and NaN
        # because of the advantage normalization
        assert (
            batch_size > 1
        ), "`batch_size` must be greater than 1. See https://github.com/DLR-RM/stable-baselines3/issues/440"

        if self.env is not None:
            # Check that `n_steps * n_envs > 1` to avoid NaN
            # when doing advantage normalization
            buffer_size = self.env.num_envs * self.n_steps
            assert (
                buffer_size > 1
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
        self.cg_max_steps = cg_max_steps
        self.cg_damping = cg_damping
        self.ls_alpha = ls_alpha
        self.ls_steps = ls_steps
        self.target_kl = target_kl

        if _init_setup_model:
            self._setup_model()

    def train(self) -> None:
        """
        Update policy using the currently gathered rollout buffer.
        """
        # Update optimizer learning rate
        self._update_learning_rate(self.policy.optimizer)

        po_values = []
        kl_divergences = []
        line_search_results = []

        continue_training = True

        # train for n_epochs epochs
        for epoch in range(self.n_epochs):
            # Do a complete pass on the rollout buffer
            for rollout_data in self.rollout_buffer.get(self.batch_size):
                actions = rollout_data.actions
                if isinstance(self.action_space, spaces.Discrete):
                    # Convert discrete action from float to long
                    actions = rollout_data.actions.long().flatten()

                # Re-sample the noise matrix because the log_std has changed
                # TODO: investigate why there is no issue with the gradient
                # if that line is commented (as in SAC)
                if self.use_sde:
                    self.policy.reset_noise(self.batch_size)

                with torch.no_grad():
                    _ = self.policy.evaluate_actions(rollout_data.observations, actions)
                    old_distribution = copy.copy(self.policy.get_distribution())

                values, log_prob, _ = self.policy.evaluate_actions(
                    rollout_data.observations, actions
                )
                distribution = self.policy.get_distribution()

                advantages = rollout_data.advantages
                advantages = (advantages - advantages.mean()) / (
                    rollout_data.advantages.std() + 1e-8
                )

                # ratio between old and new policy, should be one at the first iteration
                ratio = th.exp(log_prob - rollout_data.old_log_prob)

                # surrogate policy objective
                policy_obj = (advantages * ratio).mean()

                # KL divergence
                kl_div = kl_divergence(
                    distribution.distribution, old_distribution.distribution
                ).mean()

                # Surrogate & KL gradient
                self.policy.optimizer.zero_grad()

                # This is necessary because not all the parameters in the policy have gradients w.r.t. the KL divergence
                g = []
                grad_kl = []
                grad_shape = []
                params = []
                for param in self.policy.parameters():
                    kl_param_grad, *_ = torch.autograd.grad(
                        kl_div,
                        param,
                        create_graph=True,
                        retain_graph=True,
                        allow_unused=True,
                        only_inputs=True,
                    )
                    if kl_param_grad is not None:
                        g_grad, *_ = torch.autograd.grad(
                            policy_obj, param, retain_graph=True, only_inputs=True
                        )

                        grad_shape.append(kl_param_grad.shape)
                        grad_kl.append(kl_param_grad.view(-1))
                        g.append(g_grad.view(-1))
                        params.append(param)

                g = torch.cat(g)
                grad_kl = torch.cat(grad_kl)

                def Hpv(v, retain_graph=True):
                    jvp = (grad_kl * v).sum()
                    return flat_grad(jvp, params, retain_graph=retain_graph) + self.cg_damping * v

                s = cg_solver(Hpv, g, max_iter=self.cg_max_steps)

                beta = 2 * self.target_kl
                beta /= torch.matmul(s, Hpv(s, retain_graph=False))
                beta = torch.sqrt(beta)

                alpha = self.ls_alpha
                orig_params = [param.detach().clone() for param in params]

                line_search_success = False
                with torch.no_grad():
                    for i in range(self.ls_steps):

                        j = 0
                        for param, shape in zip(params, grad_shape):
                            k = param.numel()
                            param.data += alpha * beta * s[j : (j + k)].view(shape)
                            j += k

                        values, log_prob, _ = self.policy.evaluate_actions(
                            rollout_data.observations, actions
                        )

                        ratio = th.exp(log_prob - rollout_data.old_log_prob)
                        new_policy_obj = (advantages * ratio).mean()

                        kl_div = kl_divergence(
                            distribution.distribution, old_distribution.distribution
                        ).mean()

                        if (kl_div < self.target_kl) and (new_policy_obj > policy_obj):
                            line_search_success = True
                            break

                        for param, orig_param in zip(params, orig_params):
                            param.data = orig_param.data.clone()

                        alpha *= alpha

                    line_search_results.append(line_search_success)

                    if not line_search_success:
                        for param, orig_param in zip(params, orig_params):
                            param.data = orig_param.data.clone()

                        po_values.append(policy_obj.item())
                        kl_divergences.append(0)
                    else:
                        po_values.append(new_policy_obj.item())
                        kl_divergences.append(kl_div.item())

                # TODO: Critic training?

            if not continue_training:
                break

        self._n_updates += self.n_epochs
        explained_var = explained_variance(
            self.rollout_buffer.values.flatten(), self.rollout_buffer.returns.flatten()
        )

        # Logs
        self.logger.record("train/policy_objective_value", np.mean(po_values))
        self.logger.record("train/kl_divergence_loss", np.mean(kl_divergences))
        self.logger.record("train/explained_variance", explained_var)
        self.logger.record("train/line_search_success", np.mean(line_search_results))
        if hasattr(self.policy, "log_std"):
            self.logger.record("train/std", th.exp(self.policy.log_std).mean().item())

        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")

    def learn(
        self,
        total_timesteps: int,
        callback: MaybeCallback = None,
        log_interval: int = 1,
        eval_env: Optional[GymEnv] = None,
        eval_freq: int = -1,
        n_eval_episodes: int = 5,
        tb_log_name: str = "TRPO",
        eval_log_path: Optional[str] = None,
        reset_num_timesteps: bool = True,
    ) -> OnPolicyAlgorithm:

        return super(TRPO, self).learn(
            total_timesteps=total_timesteps,
            callback=callback,
            log_interval=log_interval,
            eval_env=eval_env,
            eval_freq=eval_freq,
            n_eval_episodes=n_eval_episodes,
            tb_log_name=tb_log_name,
            eval_log_path=eval_log_path,
            reset_num_timesteps=reset_num_timesteps,
        )
