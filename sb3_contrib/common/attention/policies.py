from typing import Any, Dict, List, Optional, Tuple, Type, Union

import numpy as np
import torch as th
from gym import spaces
from stable_baselines3.common.distributions import Distribution
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import (
    BaseFeaturesExtractor,
    CombinedExtractor,
    FlattenExtractor,
    MlpExtractor,
    NatureCNN,
)
from stable_baselines3.common.type_aliases import Schedule
from stable_baselines3.common.utils import zip_strict
from torch import nn

from sb3_contrib.common.attention.type_aliases import AttnMemory
from sb3_contrib.ppo_attention.architecture import GTrXLNet


class AttentionActorCriticPolicy(ActorCriticPolicy):
    """
    Attention policy class for actor-critic algorithms (has both policy and value prediction).
    To be used with A2C, PPO and the likes.
    It assumes that both the actor and the critic GTrXL
    have the same model.

    :param observation_space: Observation space
    :param action_space: Action space
    :param lr_schedule: Learning rate schedule (could be constant)
    :param net_arch: The specification of the policy and value networks.
    :param activation_fn: Activation function
    :param ortho_init: Whether to use or not orthogonal initialization
    :param use_sde: Whether to use State Dependent Exploration or not
    :param log_std_init: Initial value for the log standard deviation
    :param full_std: Whether to use (n_features x n_actions) parameters
        for the std instead of only (n_features,) when using gSDE
    :param use_expln: Use ``expln()`` function instead of ``exp()`` to ensure
        a positive standard deviation (cf paper). It allows to keep variance
        above zero and prevent it from growing too fast. In practice, ``exp()`` is usually enough.
    :param squash_output: Whether to squash the output using a tanh function,
        this allows to ensure boundaries when using gSDE.
    :param features_extractor_class: Features extractor to use.
    :param features_extractor_kwargs: Keyword arguments
        to pass to the features extractor.
    :param share_features_extractor: If True, the features extractor is shared between the policy and value networks.
    :param normalize_images: Whether to normalize images or not,
         dividing by 255.0 (True by default)
    :param optimizer_class: The optimizer to use,
        ``th.optim.Adam`` by default
    :param optimizer_kwargs: Additional keyword arguments,
        excluding the learning rate, to pass to the optimizer
    :param n_layers: Number of layers (MHA + Position-wise MLP)
    :param attention_dim: Dimension of the attention latent space
    :param num_heads: Number of heads of the MHA
    :param memory_inference: (not used)
    :param memory_training: (not used)
    :param head_dim: Heads dimension of the MHA
    :param position_wise_mlp_dim: Dimension of the Position-wise MLP
    :param init_gru_gate_bias: Bias initialization of the GRU gates
    :param device: PyTorch device.
    """

    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        lr_schedule: Schedule,
        net_arch: Optional[Union[List[int], Dict[str, List[int]]]] = None,
        activation_fn: Type[nn.Module] = nn.Tanh,
        ortho_init: bool = True,
        use_sde: bool = False,
        log_std_init: float = 0.0,
        full_std: bool = True,
        use_expln: bool = False,
        squash_output: bool = False,
        features_extractor_class: Type[BaseFeaturesExtractor] = FlattenExtractor,
        features_extractor_kwargs: Optional[Dict[str, Any]] = None,
        share_features_extractor: bool = True,
        normalize_images: bool = True,
        optimizer_class: Type[th.optim.Optimizer] = th.optim.Adam,
        optimizer_kwargs: Optional[Dict[str, Any]] = None,

        n_layers: int = 1,
        attention_dim: int = 64,
        n_heads: int = 2,
        memory_inference: int = 50,
        memory_training: int = 50,
        head_dim: int = 32,
        position_wise_mlp_dim: int = 32,
        init_gru_gate_bias: float = 2.0,
    ):
        self.attention_dim = attention_dim
        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            net_arch,
            activation_fn,
            ortho_init,
            use_sde,
            log_std_init,
            full_std,
            use_expln,
            squash_output,
            features_extractor_class,
            features_extractor_kwargs,
            share_features_extractor,
            normalize_images,
            optimizer_class,
            optimizer_kwargs,
        )

        self.n_layers = n_layers
        self.n_heads = n_heads
        # Same model for actor and critic
        self.model = GTrXLNet(
            feature_dim=self.features_dim,
            n_layers=n_layers,
            attention_dim=attention_dim,
            num_heads=n_heads,
            memory_inference=memory_inference,
            memory_training=memory_training,
            head_dim=head_dim,
            position_wise_mlp_dim=position_wise_mlp_dim,
            init_gru_gate_bias=init_gru_gate_bias,
            device=self.device,
        )
        # For the predict() method, to initialize attention memory
        # (n_layers, batch_size, attention_dim)
        self.memory_shape = (n_layers, 1, attention_dim)

        # Setup optimizer with initial learning rate
        self.optimizer = self.optimizer_class(self.parameters(), lr=lr_schedule(1), **self.optimizer_kwargs)

    def _build_mlp_extractor(self) -> None:
        """
        Create the policy and value networks.
        Part of the layers can be shared.
        """
        self.mlp_extractor = MlpExtractor(
            self.attention_dim,
            net_arch=self.net_arch,
            activation_fn=self.activation_fn,
            device=self.device,
        )

    @staticmethod
    def _process_sequence(
        features: th.Tensor,
        attn_memory: th.Tensor,#Tuple[th.Tensor, th.Tensor],
        episode_starts: th.Tensor,
        model: GTrXLNet,
    ) -> Tuple[th.Tensor, th.Tensor]:
        """
        Do a forward pass in the GTrXL network.

        :param features: Input tensor
        :param attn_memory: previous attention memory of the GTrXL
        :param episode_starts: Indicates when a new episode starts,
            in that case, we need to reset the attention.
        :param model: GTrXL network object.
        :return: GTrXL output and updated GTrXL memory.
        """
        # (sequence length, batch size, features dim)
        # (batch size = n_envs for data collection or n_seq when doing gradient update)
        n_seq = attn_memory.shape[1]
        # Batch to sequence
        # (padded batch size, features_dim) -> (n_seq, max length, features_dim) -> (max length, n_seq, features_dim)
        # note: max length (max sequence length) is always 1 during data collection
        features_sequence = features.reshape((n_seq, -1, model.input_size)).swapaxes(0, 1)
        episode_starts = episode_starts.reshape((n_seq, -1)).swapaxes(0, 1)

        # If we don't have to reset the memory in the middle of a sequence
        # we can avoid the for loop, which speeds up things
        if th.all(episode_starts == 0.0):
            attn_output, attn_memory = model(features_sequence, attn_memory)
            # attn_output = th.flatten(attn_output.transpose(0, 1), start_dim=0, end_dim=1)
            return attn_output, attn_memory

        outputs = []
        # Iterate over the sequence
        # print('features_sequence', features_sequence.size())
        for features, episode_start in zip_strict(features_sequence, episode_starts):
            out, attn_memory = model(
                features.unsqueeze(dim=0),
                (
                    # Reset the memory at the beginning of a new episode
                    (1.0 - episode_start).view(1, -1, 1) * attn_memory
                ),
            )
            outputs += [out]
        # Sequence to batch
        # (sequence length, n_seq, out_dim) -> (batch_size, out_dim)
        outputs = th.cat(outputs)
        #attn_output = th.flatten(th.cat(attn_output).transpose(0, 1), start_dim=0, end_dim=1)
        return outputs, attn_memory

    def forward(
        self,
        obs: th.Tensor,
        attn_memory: AttnMemory,
        episode_starts: th.Tensor,
        deterministic: bool = False,
    ) -> Tuple[th.Tensor, th.Tensor, th.Tensor, AttnMemory]:
        """
        Forward pass in all the networks (actor and critic)

        :param obs: Observation. Observation
        :param attn_memory: The last attention memory for the GTrXL model.
        :param episode_starts: Whether the observations correspond to new episodes
            or not (we reset the attention memory in that case).
        :param deterministic: Whether to sample or use deterministic actions
        :return: action, value and log probability of the action
        """
        # Preprocess the observation if needed
        features = self.extract_features(obs)
        # if self.share_features_extractor:
        #     pi_features = vf_features = features  # alis
        # else:
        #     pi_features, vf_features = features
        pi_features = vf_features = features  # alis
        # latent_pi, latent_vf = self.mlp_extractor(features)
        latent_pi, attn_memory_pi = self._process_sequence(pi_features, attn_memory.pi, episode_starts, self.model)
        # if self.model_critic is not None:
        #     latent_vf, attn_memory_vf = self._process_sequence(vf_features, attn_memory.vf, episode_starts, self.model_critic)
        # elif self.shared_model:
        #     # Re-use GTrXL features but do not backpropagate
        #     latent_vf = latent_pi.detach()
        #     attn_memory_vf = (attn_memory_pi[0].detach(), attn_memory_pi[1].detach())
        # else:
        #     # Critic only has a feedforward network
        #     latent_vf = self.critic(vf_features)
        #     attn_memory_vf = attn_memory_pi

        #print('out:', latent_pi.size(), 'attn_memory:', attn_memory_pi.size())
        # attn_memory_vf = attn_memory_pi.detach()
        latent_vf = latent_pi.detach()

        latent_pi = self.mlp_extractor.forward_actor(latent_pi)
        latent_vf = self.mlp_extractor.forward_critic(latent_vf)

        # Evaluate the values for the given observations
        values = self.value_net(latent_pi) # latent_vf
        distribution = self._get_action_dist_from_latent(latent_pi)
        actions = distribution.get_actions(deterministic=deterministic)
        log_prob = distribution.log_prob(actions)
        return actions, values, log_prob, AttnMemory(attn_memory_pi)

    def get_distribution(
        self,
        obs: th.Tensor,
        attn_memory: th.Tensor,#Tuple[th.Tensor, th.Tensor],
        episode_starts: th.Tensor,
    ) -> Tuple[Distribution, Tuple[th.Tensor, ...]]:
        """
        Get the current policy distribution given the observations.

        :param obs: Observation.
        :param attn_memory: The last attention memory for the GTrXL model.
        :param episode_starts: Whether the observations correspond to new episodes
            or not (we reset the attention memory in that case).
        :return: the action distribution and new memory.
        """
        # Call the method from the parent of the parent class
        features = super(ActorCriticPolicy, self).extract_features(obs, self.pi_features_extractor)
        latent_pi, attn_memory = self._process_sequence(features, attn_memory, episode_starts, self.model)
        latent_pi = self.mlp_extractor.forward_actor(latent_pi)
        return self._get_action_dist_from_latent(latent_pi), attn_memory

    def predict_values(
        self,
        obs: th.Tensor,
        attn_memory: th.Tensor,#Tuple[th.Tensor, th.Tensor],
        episode_starts: th.Tensor,
    ) -> th.Tensor:
        """
        Get the estimated values according to the current policy given the observations.

        :param obs: Observation.
        :param attn_memory: The attention memory for the GTrXL model.
        :param episode_starts: Whether the observations correspond to new episodes
            or not (we reset the attention memory in that case).
        :return: the estimated values.
        """
        # Call the method from the parent of the parent class
        features = super(ActorCriticPolicy, self).extract_features(obs, self.vf_features_extractor)

        # if self.model_critic is not None:
        #     latent_vf, attn_memory = self._process_sequence(features, attn_memory, episode_starts, self.model_critic)
        # elif self.shared_model:
        #     # Use GTrXL from the actor
        #     latent_pi, _ = self._process_sequence(features, attn_memory, episode_starts, self.model_actor)
        #     latent_vf = latent_pi.detach()
        # else:
        #     latent_vf = self.critic(features)
        latent_pi, _ = self._process_sequence(features, attn_memory, episode_starts, self.model)
        latent_vf = latent_pi.detach()

        latent_vf = self.mlp_extractor.forward_critic(latent_vf)
        return self.value_net(latent_vf)

    def evaluate_actions(
        self, obs: th.Tensor, actions: th.Tensor, attn_memory: AttnMemory, episode_starts: th.Tensor
    ) -> Tuple[th.Tensor, th.Tensor, th.Tensor]:
        """
        Evaluate actions according to the current policy,
        given the observations.

        :param obs: Observation.
        :param actions:
        :param attn_memory: The last attention memory for the GTrXL model.
        :param episode_starts: Whether the observations correspond to new episodes
            or not (we reset the attention memory in that case).
        :return: estimated value, log likelihood of taking those actions
            and entropy of the action distribution.
        """
        #print('OBS', obs.size())
        # Preprocess the observation if needed
        features = self.extract_features(obs)
        #print('FEATURES', features.size())
        # if self.share_features_extractor:
        #     pi_features = vf_features = features  # alias
        # else:
        #     pi_features, vf_features = features
        pi_features = vf_features = features  # alias
        #attn_memory = th.tensor(attn_memory.pi, dtype=th.float32, device=self.device)
        latent_pi, _ = self._process_sequence(pi_features, attn_memory.pi, episode_starts, self.model)
        # if self.model_critic is not None:
        #     latent_vf, _ = self._process_sequence(vf_features, attn_memory.vf, episode_starts, self.model_critic)
        # elif self.shared_model:
        #     latent_vf = latent_pi.detach()
        # else:
        #     latent_vf = self.critic(vf_features)
        latent_vf = latent_pi.detach()

        latent_pi = self.mlp_extractor.forward_actor(latent_pi)
        #latent_vf = self.mlp_extractor.forward_critic(latent_vf)
        
        distribution = self._get_action_dist_from_latent(latent_pi)
        log_prob = distribution.log_prob(actions)
        values = self.value_net(latent_pi) #values = self.value_net(latent_vf)
        return values, log_prob, distribution.entropy()

    def _predict(
        self,
        observation: th.Tensor,
        attn_memory: th.Tensor, #Tuple[th.Tensor, th.Tensor],
        episode_starts: th.Tensor,
        deterministic: bool = False,
    ) -> Tuple[th.Tensor, Tuple[th.Tensor, ...]]:
        """
        Get the action according to the policy for a given observation.

        :param observation:
        :param attn_memory: The last attention memory for the GTrXL model.
        :param episode_starts: Whether the observations correspond to new episodes
            or not (we reset the attention memory in that case).
        :param deterministic: Whether to use stochastic or deterministic actions
        :return: Taken action according to the policy and memory of the Attention network
        """
        distribution, attn_memory = self.get_distribution(observation, attn_memory, episode_starts)
        return distribution.get_actions(deterministic=deterministic), attn_memory

    def predict(
        self,
        observation: Union[np.ndarray, Dict[str, np.ndarray]],
        memory: Optional[np.ndarray] = None,
        episode_start: Optional[np.ndarray] = None,
        deterministic: bool = False,
    ) -> Tuple[np.ndarray, Optional[Tuple[np.ndarray, ...]]]:
        """
        Get the policy action from an observation (and optional attention memory).
        Includes sugar-coating to handle different observations (e.g. normalizing images).

        :param observation: the input observation
        :param memory: The last attention memory for the GTrXL.
        :param episode_starts: Whether the observations correspond to new episodes
            or not (we reset the attention memory in that case).
        :param deterministic: Whether or not to return deterministic actions.
        :return: the model's action and the next memory
            (used in attention policies)
        """
        # Switch to eval mode (this affects batch norm / dropout)
        self.set_training_mode(False)

        observation, vectorized_env = self.obs_to_tensor(observation)

        if isinstance(observation, dict):
            n_envs = observation[list(observation.keys())[0]].shape[0]
        else:
            n_envs = observation.shape[0]
        # memory : (n_layers, n_envs, dim)
        if memory is None:
            # Initialize memory to zeros
            memory = np.concatenate([np.zeros(self.memory_shape) for _ in range(n_envs)], axis=1)
            #memory = (memory, memory)

        if episode_start is None:
            episode_start = np.array([False for _ in range(n_envs)])

        with th.no_grad():
            # Convert to PyTorch tensors
            #memory = th.tensor(memory[0], dtype=th.float32, device=self.device), th.tensor(
            #    memory[1], dtype=th.float32, device=self.device
            #)
            episode_starts = th.tensor(episode_start, dtype=th.float32, device=self.device)
            actions, memory = self._predict(
                observation, attn_memory=memory, episode_starts=episode_starts, deterministic=deterministic
            )

        # Convert to numpy
        actions = actions.cpu().numpy()

        if isinstance(self.action_space, spaces.Box):
            if self.squash_output:
                # Rescale to proper domain when using squashing
                actions = self.unscale_action(actions)
            else:
                # Actions could be on arbitrary scale, so clip the actions to avoid
                # out of bound error (e.g. if sampling from a Gaussian distribution)
                actions = np.clip(actions, self.action_space.low, self.action_space.high)

        # Remove batch dimension if needed
        if not vectorized_env:
            actions = actions.squeeze(axis=0)

        return actions, memory


class AttentionActorCriticCnnPolicy(AttentionActorCriticPolicy):
    """
    CNN attention policy class for actor-critic algorithms (has both policy and value prediction).
    Used by A2C, PPO and the likes.

    :param observation_space: Observation space
    :param action_space: Action space
    :param lr_schedule: Learning rate schedule (could be constant)
    :param net_arch: The specification of the policy and value networks.
    :param activation_fn: Activation function
    :param ortho_init: Whether to use or not orthogonal initialization
    :param use_sde: Whether to use State Dependent Exploration or not
    :param log_std_init: Initial value for the log standard deviation
    :param full_std: Whether to use (n_features x n_actions) parameters
        for the std instead of only (n_features,) when using gSDE
    :param use_expln: Use ``expln()`` function instead of ``exp()`` to ensure
        a positive standard deviation (cf paper). It allows to keep variance
        above zero and prevent it from growing too fast. In practice, ``exp()`` is usually enough.
    :param squash_output: Whether to squash the output using a tanh function,
        this allows to ensure boundaries when using gSDE.
    :param features_extractor_class: Features extractor to use.
    :param features_extractor_kwargs: Keyword arguments
        to pass to the features extractor.
    :param share_features_extractor: If True, the features extractor is shared between the policy and value networks.
    :param normalize_images: Whether to normalize images or not,
         dividing by 255.0 (True by default)
    :param optimizer_class: The optimizer to use,
        ``th.optim.Adam`` by default
    :param optimizer_kwargs: Additional keyword arguments,
        excluding the learning rate, to pass to the optimizer
    :param n_layers: Number of layers (MHA + Position-wise MLP)
    :param attention_dim: Dimension of the attention latent space
    :param num_heads: Number of heads of the MHA
    :param memory_inference: (not used)
    :param memory_training: (not used)
    :param head_dim: Heads dimension of the MHA
    :param position_wise_mlp_dim: Dimension of the Position-wise MLP
    :param init_gru_gate_bias: Bias initialization of the GRU gates
    :param device: PyTorch device.
    """

    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        lr_schedule: Schedule,
        net_arch: Optional[Union[List[int], Dict[str, List[int]]]] = None,
        activation_fn: Type[nn.Module] = nn.Tanh,
        ortho_init: bool = True,
        use_sde: bool = False,
        log_std_init: float = 0.0,
        full_std: bool = True,
        use_expln: bool = False,
        squash_output: bool = False,
        features_extractor_class: Type[BaseFeaturesExtractor] = NatureCNN,
        features_extractor_kwargs: Optional[Dict[str, Any]] = None,
        share_features_extractor: bool = True,
        normalize_images: bool = True,
        optimizer_class: Type[th.optim.Optimizer] = th.optim.Adam,
        optimizer_kwargs: Optional[Dict[str, Any]] = None,

        n_layers: int = 1,
        attention_dim: int = 64,
        n_heads: int = 2,
        memory_inference: int = 50,
        memory_training: int = 50,
        head_dim: int = 32,
        position_wise_mlp_dim: int = 32,
        init_gru_gate_bias: float = 2.0,
    ):
        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            net_arch,
            activation_fn,
            ortho_init,
            use_sde,
            log_std_init,
            full_std,
            use_expln,
            squash_output,
            features_extractor_class,
            features_extractor_kwargs,
            share_features_extractor,
            normalize_images,
            optimizer_class,
            optimizer_kwargs,
            n_layers,
            attention_dim,
            n_heads,
            memory_inference,
            memory_training,
            head_dim,
            position_wise_mlp_dim,
            init_gru_gate_bias,
        )


class AttentionMultiInputActorCriticPolicy(AttentionActorCriticPolicy):
    """
    MultiInputActorClass policy class for actor-critic algorithms (has both policy and value prediction).
    Used by A2C, PPO and the likes.

    :param observation_space: Observation space
    :param action_space: Action space
    :param lr_schedule: Learning rate schedule (could be constant)
    :param net_arch: The specification of the policy and value networks.
    :param activation_fn: Activation function
    :param ortho_init: Whether to use or not orthogonal initialization
    :param use_sde: Whether to use State Dependent Exploration or not
    :param log_std_init: Initial value for the log standard deviation
    :param full_std: Whether to use (n_features x n_actions) parameters
        for the std instead of only (n_features,) when using gSDE
    :param use_expln: Use ``expln()`` function instead of ``exp()`` to ensure
        a positive standard deviation (cf paper). It allows to keep variance
        above zero and prevent it from growing too fast. In practice, ``exp()`` is usually enough.
    :param squash_output: Whether to squash the output using a tanh function,
        this allows to ensure boundaries when using gSDE.
    :param features_extractor_class: Features extractor to use.
    :param features_extractor_kwargs: Keyword arguments
        to pass to the features extractor.
    :param share_features_extractor: If True, the features extractor is shared between the policy and value networks.
    :param normalize_images: Whether to normalize images or not,
         dividing by 255.0 (True by default)
    :param optimizer_class: The optimizer to use,
        ``th.optim.Adam`` by default
    :param optimizer_kwargs: Additional keyword arguments,
        excluding the learning rate, to pass to the optimizer
    :param n_layers: Number of layers (MHA + Position-wise MLP)
    :param attention_dim: Dimension of the attention latent space
    :param num_heads: Number of heads of the MHA
    :param memory_inference: (not used)
    :param memory_training: (not used)
    :param head_dim: Heads dimension of the MHA
    :param position_wise_mlp_dim: Dimension of the Position-wise MLP
    :param init_gru_gate_bias: Bias initialization of the GRU gates
    :param device: PyTorch device.
    """

    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        lr_schedule: Schedule,
        net_arch: Optional[Union[List[int], Dict[str, List[int]]]] = None,
        activation_fn: Type[nn.Module] = nn.Tanh,
        ortho_init: bool = True,
        use_sde: bool = False,
        log_std_init: float = 0.0,
        full_std: bool = True,
        use_expln: bool = False,
        squash_output: bool = False,
        features_extractor_class: Type[BaseFeaturesExtractor] = CombinedExtractor,
        features_extractor_kwargs: Optional[Dict[str, Any]] = None,
        share_features_extractor: bool = True,
        normalize_images: bool = True,
        optimizer_class: Type[th.optim.Optimizer] = th.optim.Adam,
        optimizer_kwargs: Optional[Dict[str, Any]] = None,

        n_layers: int = 1,
        attention_dim: int = 64,
        n_heads: int = 2,
        memory_inference: int = 50,
        memory_training: int = 50,
        head_dim: int = 32,
        position_wise_mlp_dim: int = 32,
        init_gru_gate_bias: float = 2.0,
    ):
        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            net_arch,
            activation_fn,
            ortho_init,
            use_sde,
            log_std_init,
            full_std,
            use_expln,
            squash_output,
            features_extractor_class,
            features_extractor_kwargs,
            share_features_extractor,
            normalize_images,
            optimizer_class,
            optimizer_kwargs,
            n_layers,
            attention_dim,
            n_heads,
            memory_inference,
            memory_training,
            head_dim,
            position_wise_mlp_dim,
            init_gru_gate_bias,
        )
