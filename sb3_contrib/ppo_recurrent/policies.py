from stable_baselines3.common.policies import register_policy

from sb3_contrib.common.recurrent.policies import (
    RecurrentActorCriticCnnPolicy,
    RecurrentActorCriticPolicy,
    RecurrentMultiInputActorCriticPolicy,
)

MlpLstmPolicy = RecurrentActorCriticPolicy
CnnLstmPolicy = RecurrentActorCriticCnnPolicy
MultiInputLstmPolicy = RecurrentMultiInputActorCriticPolicy

register_policy("MlpLstmPolicy", RecurrentActorCriticPolicy)
register_policy("CnnLstmPolicy", RecurrentActorCriticCnnPolicy)
register_policy("MultiInputLstmPolicy", RecurrentMultiInputActorCriticPolicy)
