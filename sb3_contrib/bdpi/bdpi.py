import random
from typing import Any, Dict, List, Optional, Tuple, Type, Union

import gym
import torch as th
import torch.multiprocessing as mp
from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.off_policy_algorithm import OffPolicyAlgorithm
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, Schedule, TensorDict

from sb3_contrib.bdpi.policies import BDPIPolicy

# Because BDPI uses many critics per agent, and each critic has 2 Q-Networks, sharing them with file descriptors
# exhausts the maximum number of open file descriptors on many Linux distributions. The file_system sharing method
# creates many small files in /dev/shm, that are then shared by file-name. This avoids reaching the maximum number
# of open file descriptors.
mp.set_sharing_strategy("file_system")


def train_model(
    model: th.nn.Module,
    inp: Union[th.Tensor, TensorDict],
    outp: th.Tensor,
    gradient_steps: int,
) -> float:
    """Train a PyTorch module on inputs and outputs, minimizing the MSE loss for gradient_steps steps.

    :param model: PyTorch module to be trained. It must have a ".optimizer" attribute with an instance of Optimizer in it.
    :param inp: Input tensor (or dictionary of tensors if model is a MultiInput model)
    :param outp: Expected outputs tensor
    :param gradient_steps: Number of gradient steps to execute when minimizing the MSE.
    :return: MSE loss (with the 'sum' reduction) after the last gradient step, as a float.
    """
    mse_loss = th.nn.MSELoss(reduction="sum")

    for i in range(gradient_steps):
        predicted = model(inp)
        loss = mse_loss(predicted, outp)

        model.optimizer.zero_grad()
        loss.backward()
        model.optimizer.step()

    return float(loss.item())


class BDPI(OffPolicyAlgorithm):
    """
    Bootstrapped Dual Policy Iteration

    Sample-efficient discrete-action RL algorithm, built on one actor trained
    to imitate the greedy policy of several Q-Learning critics.

    Paper: https://arxiv.org/abs/1903.04193

    :param policy: The policy model to use (MlpPolicy, CnnPolicy, ...)
    :param env: The environment to learn from (if registered in Gym, can be str)
    :param learning_rate: learning rate for adam optimizer,
        the same learning rate will be used for all networks (Q-Values and Actor)
        it can be a function of the current progress remaining (from 1 to 0)
    :param actor_lr: Conservative Policy Iteration learning rate for the actor (used in a formula, not for Adam gradient steps)
    :param critic_lr: Q-Learning "alpha" learning rate for the critics
    :param buffer_size: size of the replay buffer
    :param learning_starts: how many steps of the model to collect transitions for before learning starts
    :param batch_size: Minibatch size for each gradient update
    :param gamma: the discount factor
    :param train_freq: Update the model every ``train_freq`` steps. Alternatively pass a tuple of frequency and unit
        like ``(5, "step")`` or ``(2, "episode")``.
    :param gradient_steps: How many gradient steps to do after each rollout (see ``train_freq``)
    :param threads: Number of threads to use to train the actor and critics in parallel
    :param replay_buffer_class: Replay buffer class to use (for instance ``HerReplayBuffer``).
        If ``None``, it will be automatically selected.
    :param replay_buffer_kwargs: Keyword arguments to pass to the replay buffer on creation.
    :param optimize_memory_usage: Enable a memory efficient variant of the replay buffer
        at a cost of more complexity.
        See https://github.com/DLR-RM/stable-baselines3/issues/37#issuecomment-637501195
    :param create_eval_env: Whether to create a second environment that will be
        used for evaluating the agent periodically. (Only available when passing string for the environment)
    :param policy_kwargs: additional arguments to be passed to the policy on creation
    :param verbose: the verbosity level: 0 no output, 1 info, 2 debug
    :param seed: Seed for the pseudo random generators
    :param device: Device (cpu, cuda, ...) on which the code should be run.
        Setting it to auto, the code will be run on the GPU if possible.
    :param _init_setup_model: Whether or not to build the network at the creation of the instance.
    """

    def __init__(
        self,
        policy: Union[str, Type[BDPIPolicy]],
        env: Union[GymEnv, str],
        learning_rate: Union[float, Schedule] = 1e-4,
        actor_lr: float = 0.05,
        critic_lr: float = 0.2,
        buffer_size: int = 1000000,  # 1e6
        learning_starts: int = 256,
        batch_size: int = 256,
        gamma: float = 0.99,
        train_freq: Union[int, Tuple[int, str]] = 1,
        gradient_steps: int = 20,
        threads: int = 1,
        replay_buffer_class: Optional[ReplayBuffer] = None,
        replay_buffer_kwargs: Optional[Dict[str, Any]] = None,
        optimize_memory_usage: bool = False,
        tensorboard_log: Optional[str] = None,
        create_eval_env: bool = False,
        policy_kwargs: Dict[str, Any] = None,
        verbose: int = 0,
        seed: Optional[int] = None,
        device: Union[th.device, str] = "auto",
        _init_setup_model: bool = True,
    ):
        super(BDPI, self).__init__(
            policy,
            env,
            BDPIPolicy,
            learning_rate,
            buffer_size,
            learning_starts,
            batch_size,
            0.0,
            gamma,
            train_freq,
            gradient_steps,
            None,
            replay_buffer_class=replay_buffer_class,
            replay_buffer_kwargs=replay_buffer_kwargs,
            policy_kwargs=policy_kwargs,
            tensorboard_log=tensorboard_log,
            verbose=verbose,
            device=device,
            create_eval_env=create_eval_env,
            seed=seed,
            use_sde=False,
            sde_sample_freq=1,
            use_sde_at_warmup=False,
            optimize_memory_usage=optimize_memory_usage,
            supported_action_spaces=(gym.spaces.Discrete),
            sde_support=False,
        )

        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
        self.threads = threads
        self.pool = mp.get_context("spawn").Pool(threads)

        if _init_setup_model:
            self._setup_model()

    def _setup_model(self) -> None:
        """Create the BDPI actor and critics, and make their memory shared across processes."""
        super(BDPI, self)._setup_model()

        self.actor = self.policy.actor
        self.criticsA = self.policy.criticsA
        self.criticsB = self.policy.criticsB

        self.actor.share_memory()

        for cA, cB in zip(self.criticsA, self.criticsB):
            cA.share_memory()
            cB.share_memory()

    def _excluded_save_params(self) -> List[str]:
        """Process pools cannot be pickled, so exclude "self.pool" from the saved parameters of BDPI."""
        return super()._excluded_save_params() + ["pool"]

    def train(self, gradient_steps: int, batch_size: int = 64) -> None:
        """BDPI Training procedure.

        This method is called every time-step (if train_freq=1, as in the original paper).
        Every time this method is called, the following steps are performed:

        - Every critic, in random order, gets updated with the Clipped DQN equation on its own batch of experiences
        - Every critic, just after being updated, computes its greedy policy and updates the actor towards it
        - After every critic has been updated, their QA and QB networks are swapped.

        This method implements some basic multi-processing:

        - Every critic and the actor are PyTorch modules with share_memory() called on them
        - A process pool is used to perform the neural network training operations (gradient descent steps)

        This approach only has a minimal impact on code, but does not scale very well:

        - On the plus side, the actor is trained concurrently by several workers, ala HOGWILD
        - However, the predictions (getting Q(next state), computing updated Q-Values and the greedy policy)
          all happen sequentially in the main process. With self.threads>8, the bottleneck therefore becomes
          the main process, that has to perform all the updates and predictions. The worker processes only
          fit neural networks.
        """

        # Update optimizers learning rate
        optimizers = [self.actor.optimizer] + [c.optimizer for c in self.criticsA] + [c.optimizer for c in self.criticsB]
        self._update_learning_rate(optimizers)

        # Update every critic (and the actor after each critic), in a random order
        critic_losses = []
        actor_losses = []
        critics = list(zip(self.criticsA, self.criticsB))

        random.shuffle(critics)

        for criticA, criticB in critics:
            # Sample replay buffer
            with th.no_grad():
                replay_data = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)

            # Update the critic (code taken from DQN)
            with th.no_grad():
                qvA = criticA(replay_data.next_observations)
                qvB = criticB(replay_data.next_observations)
                qv = th.min(qvA, qvB)

                QN = th.arange(replay_data.rewards.shape[0])
                next_q_values = qv[QN, qvA.argmax(1)].reshape(-1, 1)

                # 1-step TD target
                target_values = replay_data.rewards + (1 - replay_data.dones) * self.gamma * next_q_values

                # Make real supervised learning target Q-Values (even for non-taken actions)
                target_q_values = criticA(replay_data.observations)
                actions = replay_data.actions.long().flatten()
                target_q_values[QN, actions] += self.critic_lr * (target_values.flatten() - target_q_values[QN, actions])

            critic_losses.append(
                self.pool.apply_async(train_model, (criticA, replay_data.observations, target_q_values, gradient_steps))
            )

            self.logger.record("train/avg_q", float(target_q_values.mean()))

            # Update the actor
            with th.no_grad():
                greedy_actions = target_q_values.argmax(1)

                train_probas = th.zeros_like(target_q_values)
                train_probas[QN, greedy_actions] = 1.0

                # Normalize the direction to be pursued
                train_probas /= 1e-6 + train_probas.sum(1)[:, None]
                actor_probas = self.actor(replay_data.observations)

                # Imitation learning (or distillation, or reward-penalty Pursuit, all these are the same thing)
                alr = self.actor_lr
                train_probas = (1.0 - alr) * actor_probas + alr * train_probas
                train_probas /= train_probas.sum(-1, keepdim=True)

            actor_losses.append(
                self.pool.apply_async(train_model, (self.actor, replay_data.observations, train_probas, gradient_steps))
            )

        # Log losses
        for aloss, closs in zip(actor_losses, critic_losses):
            self.logger.record("train/critic_loss", closs.get())
            self.logger.record("train/actor_loss", aloss.get())

        # Swap QA and QB
        self.criticsA, self.criticsB = self.criticsB, self.criticsA

    def learn(
        self,
        total_timesteps: int,
        callback: MaybeCallback = None,
        log_interval: int = 4,
        eval_env: Optional[GymEnv] = None,
        eval_freq: int = -1,
        n_eval_episodes: int = 5,
        tb_log_name: str = "BDPI",
        eval_log_path: Optional[str] = None,
        reset_num_timesteps: bool = True,
    ) -> OffPolicyAlgorithm:

        return super(BDPI, self).learn(
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
