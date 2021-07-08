from stable_baselines3.common.policies import register_policy

from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy

MlpPolicy = MaskableActorCriticPolicy

# NOTE: register does not work properly as MaskableActorCriticPolicy
# is not a direct subclass of BasePolicy, so we need to ask for a MaskablePolicy
# this won't work if we have more than one algorithm that supports masking
register_policy("MlpPolicy", MaskableActorCriticPolicy)
