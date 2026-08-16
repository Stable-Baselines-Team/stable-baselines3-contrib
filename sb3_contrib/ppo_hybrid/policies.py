from sb3_contrib.common.hybrid.policies import (
    HybridActorCriticPolicy,
    HybridActorCriticCnnPolicy,
    HybridMultiInputActorCriticPolicy,
)

MlpPolicy = HybridActorCriticPolicy
CnnPolicy = HybridActorCriticCnnPolicy
MultiInputPolicy = HybridMultiInputActorCriticPolicy
