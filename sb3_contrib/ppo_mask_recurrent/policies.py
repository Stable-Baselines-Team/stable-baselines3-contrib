from sb3_contrib.common.maskable_recurrent.policies import (
    MaskableRecurrentActorCriticPolicy,
    MaskableRecurrentActorCriticCnnPolicy,
    MaskableRecurrentMultiInputActorCriticPolicy,
)

MlpLstmPolicy = MaskableRecurrentActorCriticPolicy
CnnLstmPolicy = MaskableRecurrentActorCriticCnnPolicy
MultiInputLstmPolicy = MaskableRecurrentMultiInputActorCriticPolicy
