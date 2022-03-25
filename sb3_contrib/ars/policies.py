from stable_baselines3.common.policies import register_policy

from sb3_contrib.common.policies import ESLinearPolicy, ESPolicy

# Backward compat
ARSLinearPolicy = ESLinearPolicy
ARSPolicy = ESPolicy
# Aliases
MlpPolicy = ARSPolicy
LinearPolicy = ARSLinearPolicy


register_policy("LinearPolicy", LinearPolicy)
register_policy("MlpPolicy", MlpPolicy)
