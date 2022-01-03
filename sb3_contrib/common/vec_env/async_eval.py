import multiprocessing
import multiprocessing as mp
from collections import defaultdict
from typing import List, Optional, Union

from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import unwrap_vec_normalize
from stable_baselines3.common.vec_env.base_vec_env import CloudpickleWrapper


def _worker(
    remote: mp.connection.Connection,
    parent_remote: mp.connection.Connection,
    worker_env_wrapper: CloudpickleWrapper,
    train_policy_wrapper: CloudpickleWrapper,
    n_eval_episodes: int = 1,
) -> None:
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
                for weights_idx, candidate_weights in data:
                    train_policy.load_from_vector(candidate_weights)
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


class AsyncEval(object):
    def __init__(self, envs_fn, train_policy, start_method: Optional[str] = None):
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
            )
            # daemon=True: if the main process crashes, we should not cause things to hang
            process = ctx.Process(target=_worker, args=args, daemon=True)  # pytype:disable=attribute-error
            process.start()
            self.processes.append(process)
            work_remote.close()

    def send_jobs(self, candidate_weights, pop_size):
        jobs_per_worker = defaultdict(list)
        for weights_idx in range(pop_size):
            jobs_per_worker[weights_idx % len(self.remotes)].append((weights_idx, candidate_weights[weights_idx]))

        for remote_idx, remote in enumerate(self.remotes):
            remote.send(("eval", jobs_per_worker[remote_idx]))
        self.waiting = True

    def seed(self, seed: Optional[int] = None) -> List[Union[None, int]]:
        for idx, remote in enumerate(self.remotes):
            remote.send(("seed", seed + idx))
        return [remote.recv() for remote in self.remotes]

    def get_results(self):
        results = [remote.recv() for remote in self.remotes]
        flat_results = [result for worker_results in results for result in worker_results]
        self.waiting = False
        return flat_results

    def get_obs_rms(self):
        for remote in self.remotes:
            remote.send(("get_obs_rms", None))
        # TODO: set waiting flag?
        return [remote.recv() for remote in self.remotes]

    def sync_obs_rms(self, obs_rms):
        for remote in self.remotes:
            remote.send(("sync_obs_rms", obs_rms))

    def close(self) -> None:
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
