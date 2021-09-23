from stable_baselines3.common.policies import register_policy

from sb3_contrib.common.maskable.policies import (  # MaskableMultiInputActorCriticPolicy,
    MaskableActorCriticCnnPolicy,
    MaskableActorCriticPolicy,
)

MlpPolicy = MaskableActorCriticPolicy
CnnPolicy = MaskableActorCriticCnnPolicy
# MultiInputPolicy = MaskableMultiInputActorCriticPolicy

register_policy("MlpPolicy", MaskableActorCriticPolicy)
register_policy("CnnPolicy", MaskableActorCriticCnnPolicy)
# Currently not supported
# register_policy("MultiInputPolicy", MaskableMultiInputActorCriticPolicy)
