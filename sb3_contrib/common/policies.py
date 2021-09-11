"""Policies: abstract base class and concrete implementations."""

from stable_baselines3.common.distributions import Distribution
from stable_baselines3.common.policies import ActorCriticPolicy as _ActorCriticPolicy


class ActorCriticPolicy(_ActorCriticPolicy):
    """
    Policy class for actor-critic algorithms (has both policy and value prediction).
    Used by A2C, PPO and the likes.
    """

    def get_distribution(self) -> Distribution:
        """
        Get the current action distribution
        :return: Action distribution
        """
        return self.action_dist


# This is just to propagate get_distribution
class ActorCriticCnnPolicy(ActorCriticPolicy):
    pass


# This is just to propagate get_distribution
class MultiInputActorCriticPolicy(ActorCriticPolicy):
    pass
