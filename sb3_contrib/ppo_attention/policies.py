from sb3_contrib.common.attention.policies import (
    AttentionActorCriticCnnPolicy,
    AttentionActorCriticPolicy,
    AttentionMultiInputActorCriticPolicy,
)

MlpAttnPolicy = AttentionActorCriticPolicy
CnnAttnPolicy = AttentionActorCriticCnnPolicy
MultiInputAttnPolicy = AttentionMultiInputActorCriticPolicy
