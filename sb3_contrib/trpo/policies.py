# This file is here just to define MlpPolicy/CnnPolicy
# that work for TRPO
from sb3_contrib.common.policies import ActorCriticPolicy, ActorCriticCnnPolicy, MultiInputActorCriticPolicy
from stable_baselines3.common.policies import register_policy


MlpPolicy = ActorCriticPolicy
CnnPolicy = ActorCriticCnnPolicy
MultiInputPolicy = MultiInputActorCriticPolicy

register_policy("MlpPolicy", ActorCriticPolicy)
register_policy("CnnPolicy", ActorCriticCnnPolicy)
register_policy("MultiInputPolicy", MultiInputPolicy)
