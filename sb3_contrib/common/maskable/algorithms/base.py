from typing import Optional, Tuple

import numpy as np
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.callbacks import BaseCallback, CallbackList, ConvertCallback
from stable_baselines3.common.type_aliases import MaybeCallback
from stable_baselines3.common.vec_env import VecEnv, is_vecenv_wrapped

from sb3_contrib.common.maskable.callbacks import MaskableEvalCallback
from sb3_contrib.common.maskable.policies import MaskablePolicy
from sb3_contrib.common.vec_env.wrappers import VecActionMasker
from sb3_contrib.common.wrappers import ActionMasker


class MaskableAlgorithm(BaseAlgorithm):
    @staticmethod
    def _wrap_env(*args, **kwargs) -> VecEnv:
        env = super(MaskableAlgorithm, MaskableAlgorithm)._wrap_env(*args, **kwargs)

        # If env is wrapped with ActionMasker, wrap the VecEnv as well for convenience
        if not is_vecenv_wrapped(env, VecActionMasker) and all(env.env_is_wrapped(ActionMasker)):
            env = VecActionMasker(env)

        return env

    def _init_callback(
        self,
        callback: MaybeCallback,
        eval_env: Optional[VecEnv] = None,
        eval_freq: int = 10000,
        n_eval_episodes: int = 5,
        log_path: Optional[str] = None,
    ) -> BaseCallback:
        """
        :param callback: Callback(s) called at every step with state of the algorithm.
        :param eval_freq: How many steps between evaluations; if None, do not evaluate.
        :param n_eval_episodes: How many episodes to play per evaluation
        :param n_eval_episodes: Number of episodes to rollout during evaluation.
        :param log_path: Path to a folder where the evaluations will be saved
        :return: A hybrid callback calling `callback` and performing evaluation.
        """
        # Convert a list of callbacks into a callback
        if isinstance(callback, list):
            callback = CallbackList(callback)

        # Convert functional callback to object
        if not isinstance(callback, BaseCallback):
            callback = ConvertCallback(callback)

        # Create eval callback in charge of the evaluation
        if eval_env is not None:
            eval_callback = MaskableEvalCallback(
                eval_env,
                best_model_save_path=log_path,
                log_path=log_path,
                eval_freq=eval_freq,
                n_eval_episodes=n_eval_episodes,
            )
            callback = CallbackList([callback, eval_callback])

        callback.init_callback(self)
        return callback

    def predict(
        self,
        observation: np.ndarray,
        state: Optional[np.ndarray] = None,
        mask: Optional[np.ndarray] = None,
        deterministic: bool = False,
        action_masks: np.ndarray = None,
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Get the model's action(s) from an observation
        :param observation: the input observation
        :param state: The last states (can be None, used in recurrent policies)
        :param mask: The last masks (can be None, used in recurrent policies)
        :param deterministic: Whether or not to return deterministic actions.
        :param action_masks: Action masks to apply to the action distribution.
        :return: the model's action and the next state
            (used in recurrent policies)
        """

        if isinstance(self.policy, MaskablePolicy):
            return self.policy.predict(observation, state, mask, deterministic, action_masks=action_masks)
        else:
            return self.policy.predict(observation, state, mask, deterministic)
