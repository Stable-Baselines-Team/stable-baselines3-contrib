import copy
import time
from typing import Any, Dict, Optional, Type, Union
import torch as th
import numpy as np
import gym
import torch.nn.utils

from sb3_contrib.ars.policies import ARSPolicy
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.policies import BasePolicy
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, Schedule
from stable_baselines3.common.utils import get_schedule_fn


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
    :param verbose: Verbosity level TODO (probably 0,1, or 2??)
    :param device: Torch device to use for training, defaults to "cpu"
    :param _init_setip_model: TODO
    :param alive_bonus_offset: Constant added to the reward at each step, a value of -1 is used in the original paper
    """

    def __init__(
            self,
            policy: Union[str, Type[ARSPolicy]],
            env:  Union[GymEnv, str],
            n_delta: int = 64,
            n_top: Optional[int] = None,
            alpha: Union[float, Schedule] = 0.02,
            sigma: Union[float, Schedule] = 0.025,
            policy_kwargs: Optional[Dict[str, Any]] = None,
            policy_base: Type[BasePolicy] = ARSPolicy,
            tensorboard_log: Optional[str] = None,
            seed: Optional[int] = None,
            verbose: int = 0,
            device: Union[th.device, str] = "cpu",
            _init_setup_model: bool = True,
            alive_bonus_offset: float = 0,
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

        self.n_workers = env.num_envs

        if policy_kwargs is None:
            policy_kwargs = {}
        self.policy_kwargs = policy_kwargs

        self.seed = seed
        self.alive_bonus_offset = alive_bonus_offset

        if _init_setup_model: # TODO ... what do I do if this is false? am i guarunteed that someone will call this before training?
            self._setup_model()

    def _setup_model(self) -> None:
        self.set_random_seed(self.seed)
        self.rng = np.random.default_rng(self.seed)

        self.policy = self.policy_class(self.observation_space, self.action_space, **self.policy_kwargs)
        self.policy = self.policy.to(self.device)
        self.dtype = self.policy.parameters().__next__().dtype # This seems sort of like a hack
        self.theta = th.nn.utils.parameters_to_vector(self.policy.parameters()).detach().numpy()
        self.n_params = len(self.theta)

    def train(self, total_steps: int, callback: MaybeCallback):
        cur_steps = 0

        theta_tensor = th.tensor(np.zeros_like(self.theta), requires_grad=False, dtype=self.dtype)
        th.nn.utils.vector_to_parameters(theta_tensor, self.policy.parameters())
        self.theta = th.nn.utils.parameters_to_vector(self.policy.parameters()).detach().numpy()


        while cur_steps < total_steps:
            print("cur_step", cur_steps)
            deltas = self.rng.standard_normal((self.n_delta, self.n_params)) * self.sigma(cur_steps)  # TODO check if this is working the way I think it is
            candidate_returns, batch_steps = self._collect_rollouts(deltas)
            cur_steps += batch_steps

            plus_returns = candidate_returns[:self.n_delta]
            minus_returns = candidate_returns[self.n_delta:]
            top_returns = np.zeros_like(plus_returns)

            for i, ret in enumerate(zip(plus_returns, minus_returns)):
                top_returns[i] = np.max([plus_returns[i], minus_returns[i]])

            top_idx = np.argsort(top_returns)[::-1][:self.n_top]

            print(f"Top return: {top_returns[top_idx].mean()}")
            plus_returns = plus_returns[top_idx]
            minus_returns = minus_returns[top_idx]

            update_return_std = np.sqrt((plus_returns.var() + minus_returns.var()).mean())
            step_size = self.alpha(cur_steps) / (self.n_top * update_return_std + 1e-6)
            self.theta = self.theta + step_size*np.sum((plus_returns - minus_returns)*deltas[top_idx].T, axis=1)

        theta_tensor = th.tensor(self.theta, requires_grad=False, dtype=self.dtype)
        torch.nn.utils.vector_to_parameters(theta_tensor, self.policy.parameters())
        return self

    def _collect_rollouts(self, deltas):
        batch_steps = 0
        theta_idx = 0

        candidate_thetas = np.concatenate([self.theta + deltas, self.theta - deltas])
        candidate_returns = np.zeros(candidate_thetas.shape[0])

        while theta_idx < candidate_returns.shape[0]:
            policies = []
            for _ in range(self.n_workers):
                theta_tensor = th.tensor(candidate_thetas[theta_idx], requires_grad=False, dtype=self.dtype)
                th.nn.utils.vector_to_parameters(theta_tensor, self.policy.parameters())
                policies.append(copy.deepcopy(self.policy))
                theta_idx += 1

            vec_obs = self.env.reset()
            vec_acts = np.zeros((self.n_workers, self.action_space.shape[0]))
            vec_returns = np.zeros(self.n_workers)
            vec_done_once = np.zeros(self.n_workers, dtype=bool)  # Tracks if each env has returned done at least once
            vec_steps_taken = np.zeros(self.n_workers)

            while not np.all(vec_done_once):  # Keep going until every workers has finished at least once
                for i in range(self.n_workers):
                    vec_acts[i, :] = policies[i].predict(vec_obs[i, :])[0]
                vec_obs, vec_rews, vec_done, _ = self.env.step(vec_acts)

                #print(vec_acts)
                vec_rews += self.alive_bonus_offset

                vec_done_once = np.bitwise_or(vec_done, vec_done_once)
                vec_returns += vec_rews*np.invert(vec_done_once)
                vec_steps_taken += np.invert(vec_done_once)

            candidate_returns[theta_idx-self.n_workers:theta_idx] = vec_returns
            batch_steps += sum(vec_steps_taken)

        return candidate_returns, batch_steps

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

        callback.on_training_start(locals(), globals())
        self.train(total_steps, callback=callback)
        callback.on_training_end()
        return self


import pybullet_envs

if __name__ == "__main__":
    env_name = "HalfCheetah-v2"
    from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize
    from stable_baselines3.common.env_util import make_vec_env

    venv = make_vec_env(env_name, 16, vec_env_cls=SubprocVecEnv)
    venv = VecNormalize(venv, norm_reward=False)

    agent = ARS("LinearPolicy", venv, n_delta=32, n_top=4, alive_bonus_offset=0)
    agent.learn(2e6)
