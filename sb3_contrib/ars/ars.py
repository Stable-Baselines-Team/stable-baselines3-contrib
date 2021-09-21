import copy
import time
import warnings
from typing import Any, Dict, Optional, Type, Union
import torch as th
import numpy as np
import gym
import torch.nn.utils

from sb3_contrib.ars.policies import ARSPolicy
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.policies import BasePolicy
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, Schedule
from stable_baselines3.common.utils import get_schedule_fn, safe_mean

from torch.multiprocessing import Queue, Process


class ARS(BaseAlgorithm):
    """
    Augmented Random Search: https://arxiv.org/abs/1803.07055

    :param policy: The policy to train, can be an instance of ARSPolicy, or a string
    :param env: The environment to train on, may be a string if registred with gym
    :param n_delta: How many random pertubations of the policy to try at each update step. For performance reasons should be a multiple of n_envs in the vec env... TODO
    :param n_top: How many of the top delta to use in each update step. Default is n_delta
    :param alpha: Float or schedule for the step size
    :param sigma: Float or schedule for the exploration noise
    :param policy_kwargs: Keyword arguments to pass to the policy on creation
    :param policy_base: Base class to use for the policy TODO I still don't really know why this is here
    :param tensorboard_log: String with the directory to put tensorboard logs:
    :param seed: Random seed for the training
    :param verbose: Verbosity level: 0 no output, 1 info, 2 debug
    :param device: Torch device to use for training, defaults to "cpu"
    :param _init_setup_model: Whether or not to build the network at the creation of the instance TODO will this always be called?
    :param alive_bonus_offset: Constant added to the reward at each step, a value of -1 is used in the original paper
    """

    def __init__(
            self,
            policy: Union[str, Type[ARSPolicy]],
            env:  Union[GymEnv, str],
            n_delta: int = 64,
            n_top: Optional[int] = None,
            alpha: Union[float, Schedule] = 0.05,
            sigma: Union[float, Schedule] = 0.05,
            policy_kwargs: Optional[Dict[str, Any]] = None,
            policy_base: Type[BasePolicy] = ARSPolicy,
            tensorboard_log: Optional[str] = None,
            seed: Optional[int] = None,
            verbose: int = 0,
            device: Union[th.device, str] = "cpu",
            _init_setup_model: bool = True,
            alive_bonus_offset: float = 0,
            zero_policy: bool = True,
    ):

        super().__init__(
            policy,
            env,
            learning_rate=0.0,
            tensorboard_log=tensorboard_log,
            policy_base=policy_base,
            policy_kwargs=policy_kwargs,
            verbose=verbose,
            device=device,
            supported_action_spaces=(gym.spaces.Box,),
            support_multi_env=True
        )

        self.n_delta = n_delta
        self.alpha = get_schedule_fn(alpha)
        self.sigma = get_schedule_fn(sigma)

        if n_top is None:
           n_top = n_delta
        self.n_top = n_top

        self.n_workers = None # We check this at training time ... I guess??

        if policy_kwargs is None:
            policy_kwargs = {}
        self.policy_kwargs = policy_kwargs

        self.seed = seed
        self.alive_bonus_offset = alive_bonus_offset

        self.zero_policy = zero_policy
        self.theta = None # Need to call init model to initialize theta

        if _init_setup_model:  # TODO ... what do I do if this is false? am i guaranteed that someone will call this before training?
            self._setup_model()

    def _setup_model(self) -> None:
        self.set_random_seed(self.seed)
        self.rng = np.random.default_rng(self.seed)

        self.policy = self.policy_class(self.observation_space, self.action_space, **self.policy_kwargs)
        self.dtype = self.policy.parameters().__next__().dtype # This seems sort of like a hack
        self.theta = th.nn.utils.parameters_to_vector(self.policy.parameters()).detach().numpy()

        if self.zero_policy:
            self.theta = np.zeros_like(self.theta)
            theta_tensor = th.tensor(self.theta, requires_grad=False, dtype=self.dtype)
            th.nn.utils.vector_to_parameters(theta_tensor, self.policy.parameters())

        self.n_params = len(self.theta)
        self.policy = self.policy.to(self.device)

    def _collect_rollouts(self, policy_deltas):
        with th.no_grad():
            batch_steps = 0
            theta_idx = 0

            # Generate 2*n_delta candidate policies by adding noise to the current theta
            candidate_thetas = np.concatenate([self.theta + policy_deltas, self.theta - policy_deltas])
            candidate_returns = np.zeros(candidate_thetas.shape[0]) # returns == sum of rewards
            self.ep_info_buffer = []

            self.callback.on_rollout_start()
            while theta_idx < candidate_returns.shape[0]:
                policy_list = []

                # We are using a vecenv with n_envs==n_workers. We batch our candidate theta evaluations into vectors
                # of length n_workers
                for _ in range(self.n_workers):
                    theta_tensor = th.tensor(candidate_thetas[theta_idx], dtype=self.dtype)
                    th.nn.utils.vector_to_parameters(theta_tensor, self.policy.parameters())
                    policy_list.append(copy.deepcopy(self.policy))
                    theta_idx += 1

                # We only want one rollout per policy, but vec envs will automatically reset when a done is sent.
                # To get around this we use vec_done_once to keep track of which environments have already finished,
                # and we ignore rewards once we see the first done.
                vec_obs = self.env.reset()
                vec_acts = np.zeros((self.n_workers, self.action_space.shape[0]))
                vec_returns = np.zeros(self.n_workers)
                vec_done_once = np.zeros(self.n_workers, dtype=bool)  # Tracks if each env has returned done at least once
                vec_steps_taken = np.zeros(self.n_workers) # used to update logs

                while not np.all(vec_done_once):
                    for i in range(self.n_workers):
                        vec_acts[i, :] = policy_list[i].predict(vec_obs[i, :])[0]
                    vec_obs, vec_rews, vec_done, vec_info = self.env.step(vec_acts)
                    self.callback.on_step()

                    vec_rews += self.alive_bonus_offset # Alive offset from the original paper, defaults to zero

                    # The order of episodes in the ep_info_buffer won't match up with the order of theta, which feels gross,
                    # But we are only using ep_info for logging so I don't think that should natter
                    for i, (done_this_time, done_before) in enumerate(zip(vec_done, vec_done_once)):
                        if done_this_time and not done_before:
                            maybe_ep_info = vec_info[i].get('episode')
                            if maybe_ep_info is not None:
                                self.ep_info_buffer.append(maybe_ep_info)

                    vec_steps_taken += np.invert(vec_done_once)
                    vec_returns += vec_rews*np.invert(vec_done_once)
                    vec_done_once = np.bitwise_or(vec_done, vec_done_once)

                candidate_returns[theta_idx-self.n_workers:theta_idx] = vec_returns
                batch_steps += sum(vec_steps_taken)  # only count steps used in the update step towards our total_steps

            self.callback.on_rollout_end()

        return candidate_returns, batch_steps

    # Make sure our hyper parameters are valid and auto correct them if they are not
    def _validate_hypers(self):
        if self.n_top > self.n_delta:
            warnings.warn(f"n_top = {self.n_top} > n_delta = {self.n_top}, setting n_top = n_delta")
            self.n_top = self.n_delta
        if self.n_delta % self.n_workers:
            new_n_delta = self.n_delta + (self.n_workers - self.n_delta % self.n_workers)
            warnings.warn(f"n_delta = {self.n_delta} should be a multiple of the number of workers = {self.n_workers}"
                          f" automatically bumping n_delta to {new_n_delta}")

            print(f"n_delta = {self.n_delta} should be a multiple of the number of workers = {self.n_workers}"
                  f" automatically bumping n_delta to {new_n_delta}")

            self.n_delta = new_n_delta

        return

    def _log_and_dump(self):
        fps = int(self.num_timesteps / (time.time() - self.start_time))
        self.logger.record("time/iterations", self._n_updates, exclude="tensorboard")
        if len(self.ep_info_buffer) > 0 and len(self.ep_info_buffer[0]) > 0:
            self.logger.record("rollout/ep_rew_mean",
                               safe_mean([ep_info["r"] for ep_info in self.ep_info_buffer]))
            self.logger.record("rollout/ep_len_mean",
                               safe_mean([ep_info["l"] for ep_info in self.ep_info_buffer]))
        self.logger.record("time/fps", fps)
        self.logger.record("time/time_elapsed", int(time.time() - self.start_time), exclude="tensorboard")
        self.logger.record("time/total_timesteps", self.num_timesteps, exclude="tensorboard")
        self.logger.dump(step=self.num_timesteps)

    def _do_one_update(self):
        sigma_this_step = self.sigma(self._current_progress_remaining)
        alpha_this_step = self.alpha(self._current_progress_remaining)

        deltas = self.rng.standard_normal((self.n_delta, self.n_params))

        candidate_returns, batch_steps = self._collect_rollouts(deltas * sigma_this_step)

        plus_returns = candidate_returns[:self.n_delta]
        minus_returns = candidate_returns[self.n_delta:]

        top_returns = np.zeros_like(plus_returns)
        for i in range(len(top_returns)):
            top_returns[i] = np.max([plus_returns[i], minus_returns[i]])

        top_idx = np.argsort(top_returns)[::-1][:self.n_top]
        plus_returns = plus_returns[top_idx]
        minus_returns = minus_returns[top_idx]
        deltas = deltas[top_idx]

        update_return_std = np.concatenate([plus_returns, minus_returns]).std()
        step_size = alpha_this_step / (self.n_top * update_return_std + 1e-6)
        self.theta = self.theta + step_size * (plus_returns - minus_returns) @ deltas

        self._n_updates += 1
        self.num_timesteps += batch_steps

    def learn(
            self,
            total_timesteps: int,
            callback: MaybeCallback = None,
            log_interval: int = 100,
            tb_log_name: str = "ARS",
            eval_env: Optional[GymEnv] = None,
            eval_freq: int = -1,
            n_eval_episodes: int = 5,
            eval_log_path: Optional[str] = None,
            reset_num_timesteps: bool = True
    ) -> "ARS":

        total_steps = total_timesteps
        total_steps, callback = self._setup_learn(
            total_steps, eval_env, callback, eval_freq, n_eval_episodes, eval_log_path, reset_num_timesteps, tb_log_name
        )

        self.callback = callback

        self.n_workers = self.env.num_envs # TODO can I do this somewhere earlier?? when is the env loaded ... ?
        self._validate_hypers()
        self.callback.on_training_start(locals(), globals())

        while self.num_timesteps < total_steps:
            self._update_current_progress_remaining(self.num_timesteps, total_timesteps)

            self._do_one_update()

            if log_interval is not None and self._n_updates % log_interval == 0:
                self._log_and_dump()

        theta_tensor = th.tensor(self.theta, dtype=self.dtype, device=self.device) # leave requires_grad how we found it
        torch.nn.utils.vector_to_parameters(theta_tensor, self.policy.parameters())

        self.callback.on_training_end()

        return self


if __name__ == "__main__":
    # Some more args

    from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize
    from stable_baselines3.common.env_util import make_vec_env

    env_name = "Swimmer-v2"

    venv = make_vec_env(env_name, 16, vec_env_cls=SubprocVecEnv)
    venv = VecNormalize(venv, norm_reward=False)

    model = ARS("LinearPolicy", venv, verbose=2, n_delta=32, alpha=0.02, sigma=0.03)

    model.learn(5e6, log_interval=1)
    from stable_baselines3.common.evaluation import evaluate_policy

    evaluate_policy(model, venv)

    # from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize
    # from stable_baselines3.common.env_util import make_vec_env
    # env_name = "Swimmer-v2"
    #
    # venv = make_vec_env(env_name, 16, vec_env_cls=SubprocVecEnv)
    # venv = VecNormalize(venv, norm_reward=False)
    #
    # model = ARS("LinearPolicy", venv, verbose=1, n_delta=32, alpha=0.02, sigma=0.03)
    #
    # model.learn(5e6, log_interval=10)
    # from stable_baselines3.common.evaluation import evaluate_policy
    # evaluate_policy(model, venv)



    # model.save("test.pkl")
    #
    # model.load("test.pkl")
    # print(model.verbose)



    # Basic example
    # from sb3_contrib import ARS
    #
    # model = ARS("LinearPolicy", "Pendulum-v0", verbose=1)
    # model.learn(total_timesteps=10000, log_interval=4)
    # model.save("ars_pendulum")

    # Vec Env
    # env_name = "Hopper-v2"
    # from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize
    # from stable_baselines3.common.env_util import make_vec_env
    #
    # venv = make_vec_env(env_name, 4, vec_env_cls=SubprocVecEnv)
    # venv = VecNormalize(venv, norm_reward=False)
    #
    # agent = ARS("LinearPolicy", venv, alpha=0.05, sigma=0.05, n_delta=8, n_top=4, alive_bonus_offset=-1, verbose=1)
    #
    # agent.learn(2e6, log_interval=10)
