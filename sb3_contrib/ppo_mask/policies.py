from stable_baselines3.common.policies import register_policy

from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy

MlpPolicy = MaskableActorCriticPolicy

register_policy("MlpPolicy", MlpPolicy)
