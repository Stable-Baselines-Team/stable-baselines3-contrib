from stable_baselines3.common.policies import register_policy

from sb3_contrib.common.policies import ESLinearPolicy, ESPolicy

MlpPolicy = ESPolicy
LinearPolicy = ESLinearPolicy


register_policy("LinearPolicy", LinearPolicy)
register_policy("MlpPolicy", MlpPolicy)
