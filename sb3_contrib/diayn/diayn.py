from typing import Any, Dict, List, Optional, Tuple, Type, Union

import gym
import numpy as np
import torch as th
import torch.nn.functional as F
from gym import spaces
from stable_baselines3 import SAC
from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.noise import ActionNoise
from stable_baselines3.common.surgeon import Surgeon
from stable_baselines3.common.type_aliases import GymEnv, Schedule
from stable_baselines3.common.utils import get_device, obs_as_tensor
from torch import nn
from torch.distributions import Categorical


class Discriminator(nn.Module):
    """
    The dicriminator, estimate log p(z | s)

    :param obs_dim: Observation dimension
    :param net_arch: The specification of the network
    :param nb_skills: The number of skills
    :param activation_fn: The activation function to use for the networks
    :param device:
    """

    def __init__(
        self,
        obs_dim: int,
        net_arch: List[int],
        nb_skills: int,
        activation_fn: Type[nn.Module] = nn.ReLU,
        device: Union[th.device, str] = "auto",
    ) -> None:
        super().__init__()
        self.device = get_device(device)
        layers = []
        previous_layer_dim = obs_dim
        # Iterate through the shared layers and build the shared parts of the network
        for layer_dim in net_arch:
            layers.append(nn.Linear(previous_layer_dim, layer_dim))  # add linear of size layer
            layers.append(activation_fn())
            previous_layer_dim = layer_dim
        layers.append(nn.Linear(previous_layer_dim, nb_skills))
        layers.append(nn.Softmax(dim=-1))

        # Create networks
        self.net = nn.Sequential(*layers).to(self.device)

    def forward(self, obs: th.Tensor) -> th.Tensor:
        probs = self.net(obs)
        return th.argmax(probs)

    def probs(self, obs: th.Tensor) -> th.Tensor:
        return self.net(obs)


class SkillWrapper(gym.GoalEnv, gym.Wrapper):
    """
    Make an envrionment skill-conditionned.

    The observation return by step is a dict composed of the ``observation`` and the ``skill``.
    ``reward`` is :math:`r = \log q(z | s_{t+1}) - log p(z)` where :math:`q` is the discriminator,
    :math:`z` the skill, and :math:`p` the categorical skill distribution.

    :param env: The environment
    :param skill_distribution: The categorical skill distribution
    :param discriminator: The discrminator, mapping obs to skill
    """

    def __init__(
        self,
        env: gym.Env,
        skill_distribution: Categorical,
        discriminator: Discriminator,
    ) -> None:
        self.env = env
        self.action_space = self.env.action_space
        self.skill_distribution = skill_distribution
        self.nb_skills = self.skill_distribution.param_shape[0]
        self.discriminator = discriminator
        observation_space = self.env.observation_space
        self.observation_space = spaces.Dict(
            {
                "observation": spaces.Box(observation_space.low, observation_space.high, observation_space.shape),
                "skill": spaces.Box(-10, 10, (self.nb_skills,)),
            }
        )

    def step(self, action: np.ndarray) -> Tuple[Dict[str, np.ndarray], float, bool, Dict[str, Any]]:
        obs, reward, done, info = self.env.step(action)
        obs = obs.astype(np.float32)  # ensure that observation type is float32
        reward = self._compute_reward(obs)
        return self._get_dict_obs(obs), reward, done, info

    def reset(self) -> Dict[str, np.ndarray]:
        self.skill = self.skill_distribution.sample()  # sample a skill
        obs = self.env.reset().astype(np.float32)  # reset the env and ensure that observation type is float32
        return self._get_dict_obs(obs)

    def _get_dict_obs(self, obs: np.ndarray) -> Dict[str, np.ndarray]:
        # Returns the dict observation composed of the wrapped observation and the skill.
        skill = F.one_hot(self.skill, num_classes=self.nb_skills)
        return {
            "observation": obs,
            "skill": skill,
        }

    def _compute_reward(self, obs: np.ndarray) -> Union[float, np.ndarray]:
        """
        Returns the reward as :math:`r = \log q_\varphi(z | s_{t+1}) - log p(z)` where :math:`q_\varphi` is
        the discriminator, :math:`z` the latent representation, and :math:`p` the skill distribution.

        :param obs: Curent observation
        :return: Reward or array of rewards, depending on the input
        """
        obs_th = obs_as_tensor(obs, self.discriminator.device)
        probs = self.discriminator.probs(obs_th)  # q_\varphi(⸱ | s_{t+1})
        prob = probs[self.skill]  # q_\varphi(z | s_{t+1})
        skill_log_prob = self.skill_distribution.log_prob(self.skill)  # log p(z)
        reward = th.log(prob) - skill_log_prob
        return reward.detach().cpu().item()


class DIAYN(SAC):
    def __init__(
        self,
        env: Union[GymEnv, str],
        nb_skills: int,
        net_arch: Optional[List] = None,
        learning_rate: Union[float, Schedule] = 0.0003,
        buffer_size: int = 1000000,
        learning_starts: int = 100,
        batch_size: int = 256,
        tau: float = 0.005,
        gamma: float = 0.99,
        train_freq: Union[int, Tuple[int, str]] = 1,
        gradient_steps: int = 1,
        action_noise: Optional[ActionNoise] = None,
        replay_buffer_class: Optional[ReplayBuffer] = None,
        replay_buffer_kwargs: Optional[Dict[str, Any]] = None,
        optimize_memory_usage: bool = False,
        ent_coef: Union[str, float] = "auto",
        target_update_interval: int = 1,
        target_entropy: Union[str, float] = "auto",
        use_sde: bool = False,
        sde_sample_freq: int = -1,
        use_sde_at_warmup: bool = False,
        tensorboard_log: Optional[str] = None,
        create_eval_env: bool = False,
        policy_kwargs: Optional[Dict[str, Any]] = None,
        surgeon: Optional[Surgeon] = None,
        verbose: int = 0,
        seed: Optional[int] = None,
        device: Union[th.device, str] = "auto",
        _init_setup_model: bool = True,
    ):

        obs_dim = env.observation_space.shape[0]
        if net_arch is None:
            net_arch = [256, 256]
        skill_distribution = Categorical(logits=th.ones(nb_skills))
        self.discriminator = Discriminator(obs_dim, net_arch=net_arch, nb_skills=nb_skills)
        env = SkillWrapper(env, skill_distribution, self.discriminator)
        self.discriminator_optimizer = th.optim.SGD(self.discriminator.parameters(), learning_rate)

        super().__init__(
            "MultiInputPolicy",
            env,
            learning_rate,
            buffer_size,
            learning_starts,
            batch_size,
            tau,
            gamma,
            train_freq,
            gradient_steps,
            action_noise,
            replay_buffer_class,
            replay_buffer_kwargs,
            optimize_memory_usage,
            ent_coef,
            target_update_interval,
            target_entropy,
            use_sde,
            sde_sample_freq,
            use_sde_at_warmup,
            tensorboard_log,
            create_eval_env,
            policy_kwargs,
            surgeon,
            verbose,
            seed,
            device,
            _init_setup_model,
        )

    def train(self, gradient_steps: int, batch_size: int = 64) -> None:
        super().train(gradient_steps, batch_size)

        discriminator_losses = []
        for gradient_step in range(gradient_steps):
            # Sample replay buffer
            replay_data = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)
            observations = replay_data.next_observations["observation"]
            input = self.discriminator.probs(observations)
            target = replay_data.observations["skill"]
            discriminator_loss = F.cross_entropy(input, target)

            # Optimize the discriminator
            self.discriminator_optimizer.zero_grad()
            discriminator_loss.backward()
            self.discriminator_optimizer.step()

            discriminator_losses.append(discriminator_loss.item())

        self.logger.record("train/discriminator_loss", np.mean(discriminator_losses))
