import multiprocessing
from copy import deepcopy
from multiprocessing import Queue

from stable_baselines3.common.evaluation import evaluate_policy


class EnvEvalProcess(multiprocessing.Process):
    def __init__(self, env, train_policy, job_queue: Queue, result_queue: Queue, exit_process=False, n_eval_episodes: int = 1):
        super().__init__(daemon=True)
        self.env = env
        # TODO: replace with pipes
        # and use CloudpickleWrapper?
        self.job_queue = job_queue
        self.result_queue = result_queue
        self.n_eval_episodes = n_eval_episodes
        self.exit_process = exit_process
        self.train_policy = deepcopy(train_policy)

    def run(self):
        # while not self.exit_process.set():
        while True:
            weights_idx, candidate_weights = self.job_queue.get()
            if candidate_weights is None:
                break
            self.train_policy.load_from_vector(candidate_weights)
            episode_rewards, episode_lengths = evaluate_policy(
                self.train_policy,
                self.env,
                n_eval_episodes=self.n_eval_episodes,
                return_episode_rewards=True,
                # Note: callback will be called by every process,
                # there will be a race condition...
                # callback=lambda _locals, _globals: callback.on_step(),
                warn=False,
            )
            self.result_queue.put(((weights_idx, self.env), (episode_rewards, episode_lengths)))
