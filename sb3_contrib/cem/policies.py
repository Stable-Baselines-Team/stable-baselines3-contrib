from stable_baselines3.common.policies import register_policy

from sb3_contrib.ars.policies import ARSLinearPolicy, ARSPolicy

MlpPolicy = ARSPolicy
LinearPolicy = ARSLinearPolicy


register_policy("LinearPolicy", LinearPolicy)
register_policy("MlpPolicy", MlpPolicy)
