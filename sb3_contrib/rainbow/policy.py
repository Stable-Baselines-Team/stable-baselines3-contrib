import math
from math import sqrt
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from gymnasium import spaces
from stable_baselines3.common.policies import BasePolicy
from stable_baselines3.common.type_aliases import PyTorchObs, Schedule
from torch import Tensor, nn, optim
from torch.nn import init as torch_init


class RainbowPolicy(BasePolicy):
    def __init__(
        self,
        observation_space: spaces.Box,
        action_space: spaces.Discrete,
        lr_schedule: Schedule,
        linear_size: int = 512,
        **kwargs: Any,
    ) -> None:
        """
        Initialise the Rainbow policy.

        :param observation_space: Observation space of the environment.
        :param action_space: Discrete act**n space of the environment.
        :param lr_schedule: Learning rate schedule.
        :param linear_size: Number of units in the noisy linear hidden layers.
        :param kwargs: Additional keyword arguments for police configuration.
        """
        super().__init__(observation_space, action_space)

        if observation_space.shape is None:
            raise ValueError("Rainbow requires an observation space with a defined shape.")

        if len(observation_space.shape) != 3:
            raise ValueError("Rainbow requires three-dimensional image observations.")

        obs_shape = observation_space.shape
        n_actions = int(action_space.n)

        self.linear_size = linear_size

        self.q_net = NatureC51(
            in_depth=obs_shape[0],
            actions=n_actions,
            device=self.device,
            image_width=obs_shape[1],
            image_height=obs_shape[2],
            linear_size=linear_size,
        )

        self.q_net_target = NatureC51(
            in_depth=obs_shape[0],
            actions=n_actions,
            device=self.device,
            image_width=obs_shape[1],
            image_height=obs_shape[2],
            linear_size=linear_size,
        )

        self.optimizer = optim.Adam(self.q_net.parameters(), lr=lr_schedule(1), eps=1.5e-4)

    def forward(self, obs: Tensor) -> Tensor:
        """
        Compute Q-values for the given observations.

        :param obs: Batch of observations.
        :return: Q-values for each observation and action.
        """
        return self.q_net.qvals(obs)

    def _predict(
        self,
        observation: PyTorchObs,
        deterministic: bool = False,
    ) -> Tensor:
        """
        Predict actions for the given observations.

        :param observation: Batch of observations.
        :param deterministic: Whether to disable NoisyNet noise for prediction.
        :return: Greedy action for each observation.
        """
        if not isinstance(observation, Tensor):
            raise TypeError("Rainbow only supports tensor observations.")
        if deterministic:
            self.disable_noise()
        else:
            self.reset_noise()
        qvals = self.forward(observation)
        return qvals.argmax(dim=1)

    @torch.no_grad()
    def disable_noise(self) -> None:
        """
        Disable noise in all noisy linear layers of the online network.
        """
        for module in self.q_net.modules():
            if isinstance(module, FactorizedNoisyLinear):
                module.disable_noise()

    @torch.no_grad()
    def reset_noise(self) -> None:
        """
        Resample the noise in all noisy linear layers of the online network.
        """
        for module in self.q_net.modules():
            if isinstance(module, FactorizedNoisyLinear):
                module.reset_noise()


class FactorizedNoisyLinear(nn.Module):
    """
    Linear layer with factorised Gaussian noise for NoisyNet exploration.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        sigma_0: float = 0.5,
        self_norm: bool = False,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.sigma_0 = sigma_0

        # weight: w = \mu^w + \sigma^w . \epsilon^w
        self.weight_mu = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_epsilon: Tensor
        self.register_buffer("weight_epsilon", torch.empty(out_features, in_features))

        # bias: b = \mu^b + \sigma^b . \epsilon^b
        self.bias_mu = nn.Parameter(torch.empty(out_features))
        self.bias_sigma = nn.Parameter(torch.empty(out_features))
        self.bias_epsilon: Tensor
        self.register_buffer("bias_epsilon", torch.empty(out_features))

        if self_norm:
            self.reset_parameters_self_norm()
        else:
            self.reset_parameters()
        self.reset_noise()

        self.disable_noise()

    @torch.no_grad()
    def reset_parameters(self) -> None:
        """
        Initialise the learnable parameters of the noisy linear layer.
        """
        scale = 1 / sqrt(self.in_features)

        torch_init.uniform_(self.weight_mu, -scale, scale)
        torch_init.uniform_(self.bias_mu, -scale, scale)

        torch_init.constant_(self.weight_sigma, self.sigma_0 * scale)
        torch_init.constant_(self.bias_sigma, self.sigma_0 * scale)

    @torch.no_grad()
    def reset_parameters_self_norm(self) -> None:
        """
        Initialise the layer parameters using self-normalising initialisation.
        """
        nn.init.normal_(self.weight_mu, std=1 / math.sqrt(self.out_features))

        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight_mu)
        bound = 1 / math.sqrt(fan_in)
        nn.init.uniform_(self.bias_mu, -bound, bound)

        scale = 1 / sqrt(self.in_features)
        torch_init.constant_(self.weight_sigma, self.sigma_0 * scale)
        torch_init.constant_(self.bias_sigma, self.sigma_0 * scale)

    @torch.no_grad()
    def _get_noise(self, size: int) -> Tensor:
        """
        Generate factorised Gaussian noise.

        :param size: Number of noise values to generate.
        :return: Transformed Gaussian noise tensor.
        """
        noise = torch.randn(size, device=self.weight_mu.device)
        return noise.sign().mul_(noise.abs().sqrt_())

    @torch.no_grad()
    def reset_noise(self) -> None:
        """
        Resample the factorised Gaussian noise.

        Independent noise vectors are generated for the input and output features
        and combined to update the weight and bias noise buffers.
        """
        epsilon_in = self._get_noise(self.in_features)
        epsilon_out = self._get_noise(self.out_features)
        self.weight_epsilon.copy_(epsilon_out.outer(epsilon_in))
        self.bias_epsilon.copy_(epsilon_out)

    @torch.no_grad()
    def disable_noise(self) -> None:
        """
        Disable factorised Gaussian noise in the layer.
        """
        self.weight_epsilon[:] = 0
        self.bias_epsilon[:] = 0

    def forward(self, input: Tensor) -> Tensor:
        """
        Apply the noisy linear transformation.

        :param input: Input tensor.
        :return: Output tensor after applying the noisy weights and biases.
        """
        return F.linear(
            input, self.weight_mu + self.weight_sigma * self.weight_epsilon, self.bias_mu + self.bias_sigma * self.bias_epsilon
        )


class NatureC51(nn.Module):
    """
    Nature CNN with dueling categorical heads for Rainbow DQN.
    """

    def __init__(
        self,
        in_depth: int,
        actions: int,
        device: torch.device | str,
        image_width: int = 84,
        image_height: int = 84,
        atoms: int = 51,
        v_min: float = -10.0,
        v_max: float = 10.0,
        linear_size: int = 512,
    ) -> None:
        """
        Initialise the categorical dueling Q-network.

        :param in_depth: Number of input channels.
        :param actions: Number of discrete actions.
        :param image_width: Observation image width.
        :param image_height: Observation image height.
        :param atoms: Number of atoms in the categorical value distribution.
        :param v_min: Minimum value of the categorical support.
        :param v_max: Maximum value of the categorical support.
        :param device: PyTorch device on which to place the network.
        :param linear_size: Number of units in the noisy linear hidden layers.
        """
        super().__init__()

        self.actions = actions
        self.atoms = atoms
        self.device = device
        self.linear_size = linear_size

        delta_z = (v_max - v_min) / (atoms - 1)

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=in_depth, out_channels=32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1),
            nn.ReLU(),
        )

        conv_out_size = self._get_conv_out((in_depth, image_width, image_height))

        # Noisy Linear Layers, with both value and advantage functions for dueling DQN
        self.fc1V = FactorizedNoisyLinear(conv_out_size, self.linear_size)
        self.fc1A = FactorizedNoisyLinear(conv_out_size, self.linear_size)
        self.fcV2 = FactorizedNoisyLinear(self.linear_size, self.atoms)
        self.fcA2 = FactorizedNoisyLinear(self.linear_size, actions * self.atoms)

        self.supports: Tensor
        self.register_buffer("supports", torch.arange(v_min, v_max + delta_z, delta_z))
        self.softmax = nn.Softmax(dim=1)

        self.to(device)

    @torch.no_grad()
    def reset_noise(self) -> None:
        """
        Resample the noise in all noisy linear layers.
        """
        for module in self.modules():
            if isinstance(module, FactorizedNoisyLinear):
                module.reset_noise()

    def _get_conv_out(self, shape: tuple[int, ...]) -> int:
        """
        Compute the flattened output size of the convolutional network.

        :param shape: Shape of a single input observation.
        :return: Flattened size of the convolutional output.
        """
        o = self.conv(torch.zeros(1, *shape))
        return int(np.prod(o.size()))

    def fc_val(self, x: Tensor) -> Tensor:
        """
        Compute the categorical value-stream output.

        :param x: Flattened convolutional features.
        :return: Logits produced by the value stream.
        """
        x = F.relu(self.fc1V(x))
        x = self.fcV2(x)

        return x

    def fc_adv(self, x: Tensor) -> Tensor:
        """
        Compute the categorical advantage-stream output.

        :param x: Flattened convolutional features.
        :return: Logits produced by the advantage stream.
        """
        x = F.relu(self.fc1A(x))
        x = self.fcA2(x)

        return x

    def forward(self, x: Tensor) -> Tensor:
        """
        Compute the categorical action-value logits.

        :param x: Batch of observations.
        :return: Categorical logits for each action and atom.
        """
        batch_size = x.size()[0]
        device = next(self.parameters()).device
        fx = x.to(device).float() / 255
        conv_out = self.conv(fx)

        conv_out = conv_out.view(batch_size, -1)

        val_out = self.fc_val(conv_out).view(batch_size, 1, self.atoms)
        adv_out = self.fc_adv(conv_out).view(batch_size, -1, self.atoms)
        adv_mean = adv_out.mean(dim=1, keepdim=True)
        return val_out + (adv_out - adv_mean)

    def both(self, x: Tensor) -> tuple[Tensor, Tensor]:
        """
        Compute the categorical logits and expected Q-values.

        :param x: Batch of observations.
        :return: Categorical logits and expected Q-values.
        """
        cat_out = self(x)
        probs = self.apply_softmax(cat_out)
        weights = probs * self.supports
        res = weights.sum(dim=2)
        return cat_out, res

    def qvals(self, x: Tensor) -> Tensor:
        """
        Compute expected Q-values for each action.

        :param x: Batch of observations.
        :return: Expected Q-values for each action.
        """
        return self.both(x)[1]

    def apply_softmax(self, t: Tensor) -> Tensor:
        """
        Apply softmax over the categorical atoms.

        :param t: Categorical logits.
        :return: Probability distribution over atoms.
        """
        return self.softmax(t.view(-1, self.atoms)).view(t.size())

    def save_checkpoint(self, name: str) -> None:
        """
        Save the network parameters to a checkpoint.

        :param name: Base name of the checkpoint file.
        """
        torch.save(self.state_dict(), name + ".model")

    def load_checkpoint(self, name: str) -> None:
        """
        Load network parameters from a checkpoint.

        :param name: Name of the checkpoint file.
        """
        self.load_state_dict(torch.load(name))
