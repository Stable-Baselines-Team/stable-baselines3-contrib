import gym.spaces
import torch as th
from stable_baselines3.common.preprocessing import get_flattened_obs_dim, is_image_space
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor, NatureCNN
from stable_baselines3.common.type_aliases import TensorDict
from torch import nn


class GoExploreExtractor(BaseFeaturesExtractor):
    """
    Feature extraction for GoExplore. Dict observation spaces containing keys "observation" and "goal".

    The output is the concatenation of:
     - the output of a feature extractor on the "observation" (CNN or MLP, depending on input shape).
     - the output of a feature extractor on the "goal" (CNN or MLP, depending on input shape).

                 •---------•
         obs --> | Encoder | ------>|
                 •---------•        |---> features
                            goal -->|

    :param observation_space: The Dict observation space
    :param cnn_output_dim: Number of features to output from each CNN submodule(s). Defaults to
        256 to avoid exploding network sizes.
    """

    def __init__(self, observation_space: gym.spaces.Dict, cell_dim: int, cnn_output_dim: int = 256):
        super(GoExploreExtractor, self).__init__(observation_space, features_dim=1)

        if is_image_space(observation_space["observation"]):
            self.observation_extractor = NatureCNN(observation_space["observation"], features_dim=cnn_output_dim)
            obs_features_dim = cnn_output_dim
        else:
            # The observation key is a vector, flatten it if needed
            obs_features_dim = get_flattened_obs_dim(observation_space["observation"])
            self.observation_extractor = nn.Sequential(nn.Flatten(), nn.Linear(obs_features_dim, obs_features_dim))

        self._features_dim = obs_features_dim + cell_dim

    def forward(self, observations: TensorDict) -> th.Tensor:
        features = self.observation_extractor(observations["observation"])
        features = th.cat([features, observations["goal"]], dim=1)
        return features
