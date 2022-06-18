import multiprocessing
import multiprocessing as mp
from collections import defaultdict
from typing import Callable, List, Optional, Tuple, Union

import numpy as np
import torch as th
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.policies import BasePolicy
from stable_baselines3.common.running_mean_std import RunningMeanStd
from stable_baselines3.common.vec_env import VecEnv, unwrap_vec_normalize
from stable_baselines3.common.vec_env.base_vec_env import CloudpickleWrapper


def _worker(
    remote: mp.connection.Connection,
    parent_remote: mp.connection.Connection,
    worker_env_wrapper: CloudpickleWrapper,
    train_policy_wrapper: CloudpickleWrapper,
    n_eval_episodes: int = 1,
) -> None:
    """
    Function that will be run in each process.
    It is in charge of creating environments, evaluating candidates
    and communicating with the main process.

    :param remote: Pipe to communicate with the parent process.
    :param parent_remote:
    :param worker_env_wrapper: Callable used to create the environment inside the process.
    :param train_policy_wrapper: Callable used to create the policy inside the process.
    :param n_eval_episodes: Number of evaluation episodes per candidate.
    """
    parent_remote.close()
    env = worker_env_wrapper.var()
    train_policy = train_policy_wrapper.var
    vec_normalize = unwrap_vec_normalize(env)
    if vec_normalize is not None:
        obs_rms = vec_normalize.obs_rms
    else:
        obs_rms = None
    while True:
        try:
            cmd, data = remote.recv()
            if cmd == "eval":
                results = []
                # Evaluate each candidate and save results
                for weights_idx, candidate_weights in data:
                    train_policy.load_from_vector(candidate_weights.cpu())
                    episode_rewards, episode_lengths = evaluate_policy(
                        train_policy,
                        env,
                        n_eval_episodes=n_eval_episodes,
                        return_episode_rewards=True,
                        warn=False,
                    )
                    results.append((weights_idx, (episode_rewards, episode_lengths)))
                remote.send(results)
            elif cmd == "seed":
                remote.send(env.seed(data))
            elif cmd == "get_obs_rms":
                remote.send(obs_rms)
            elif cmd == "sync_obs_rms":
                vec_normalize.obs_rms = data
                obs_rms = data
            elif cmd == "close":
                env.close()
                remote.close()
                break
            else:
                raise NotImplementedError(f"`{cmd}` is not implemented in the worker")
        except EOFError:
            break


class AsyncEval:
    """
    Helper class to do asynchronous evaluation of different policies with multiple processes.
    It is useful when implementing population based methods like Evolution Strategies (ES),
    Cross Entropy Method (CEM) or Augmented Random Search (ARS).

    .. warning::

        Only 'forkserver' and 'spawn' start methods are thread-safe,
        which is important to avoid race conditions.
        However, compared to
        'fork' they incur a small start-up cost and have restrictions on
        global variables. With those methods, users must wrap the code in an
        ``if __name__ == "__main__":`` block.
        For more information, see the multiprocessing documentation.

    :param envs_fn: Vectorized environments to run in subprocesses (callable)
    :param train_policy: The policy object that will load the different candidate
        weights.
    :param start_method: method used to start the subprocesses.
           Must be one of the methods returned by ``multiprocessing.get_all_start_methods()``.
           Defaults to 'forkserver' on available platforms, and 'spawn' otherwise.
    :param n_eval_episodes: The number of episodes to test each agent
    """

    def __init__(
        self,
        envs_fn: List[Callable[[], VecEnv]],
        train_policy: BasePolicy,
        start_method: Optional[str] = None,
        n_eval_episodes: int = 1,
    ):
        self.waiting = False
        self.closed = False
        n_envs = len(envs_fn)

        if start_method is None:
            # Fork is not a thread safe method (see issue #217)
            # but is more user friendly (does not require to wrap the code in
            # a `if __name__ == "__main__":`)
            forkserver_available = "forkserver" in mp.get_all_start_methods()
            start_method = "forkserver" if forkserver_available else "spawn"
        ctx = mp.get_context(start_method)

        self.remotes, self.work_remotes = zip(*[ctx.Pipe() for _ in range(n_envs)])
        self.processes = []
        for work_remote, remote, worker_env in zip(self.work_remotes, self.remotes, envs_fn):
            args = (
                work_remote,
                remote,
                CloudpickleWrapper(worker_env),
                CloudpickleWrapper(train_policy),
                n_eval_episodes,
            )
            # daemon=True: if the main process crashes, we should not cause things to hang
            process = ctx.Process(target=_worker, args=args, daemon=True)  # pytype:disable=attribute-error
            process.start()
            self.processes.append(process)
            work_remote.close()

    def send_jobs(self, candidate_weights: th.Tensor, pop_size: int) -> None:
        """
        Send jobs to the workers to evaluate new candidates.

        :param candidate_weights: The weights to be evaluated.
        :pop_size: The number of candidate (size of the population)
        """
        jobs_per_worker = defaultdict(list)
        for weights_idx in range(pop_size):
            jobs_per_worker[weights_idx % len(self.remotes)].append((weights_idx, candidate_weights[weights_idx]))

        for remote_idx, remote in enumerate(self.remotes):
            remote.send(("eval", jobs_per_worker[remote_idx]))
        self.waiting = True

    def seed(self, seed: Optional[int] = None) -> List[Union[None, int]]:
        """
        Seed the environments.

        :param seed: The seed for the pseudo-random generators.
        :return:
        """
        for idx, remote in enumerate(self.remotes):
            remote.send(("seed", seed + idx))
        return [remote.recv() for remote in self.remotes]

    def get_results(self) -> List[Tuple[int, Tuple[np.ndarray, np.ndarray]]]:
        """
        Retreive episode rewards and lengths from each worker
        for all candidates (there might be multiple candidates per worker)

        :return: A list of tuples containing each candidate index and its
            result (episodic reward and episode length)
        """
        results = [remote.recv() for remote in self.remotes]
        flat_results = [result for worker_results in results for result in worker_results]
        self.waiting = False
        return flat_results

    def get_obs_rms(self) -> List[RunningMeanStd]:
        """
        Retrieve the observation filters (observation running mean std)
        of each process, they will be combined in the main process.
        Synchronisation is done afterward using ``sync_obs_rms()``.
        :return: A list of ``RunningMeanStd`` objects (one per process)
        """
        for remote in self.remotes:
            remote.send(("get_obs_rms", None))
        return [remote.recv() for remote in self.remotes]

    def sync_obs_rms(self, obs_rms: RunningMeanStd) -> None:
        """
        Synchronise (and update) the observation filters
        (observation running mean std)
        :param obs_rms: The updated ``RunningMeanStd`` to be used
            by workers for normalizing observations.
        """
        for remote in self.remotes:
            remote.send(("sync_obs_rms", obs_rms))

    def close(self) -> None:
        """
        Close the processes.
        """
        if self.closed:
            return
        if self.waiting:
            for remote in self.remotes:
                remote.recv()
        for remote in self.remotes:
            remote.send(("close", None))
        for process in self.processes:
            process.join()
        self.closed = True
