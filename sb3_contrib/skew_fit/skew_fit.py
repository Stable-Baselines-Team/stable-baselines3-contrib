from typing import Any, Dict, Optional, Tuple, Union

import torch as th
from stable_baselines3 import SAC
from stable_baselines3.common.buffers import DictReplayBuffer
from stable_baselines3.common.noise import ActionNoise
from stable_baselines3.common.surgeon import Surgeon
from stable_baselines3.common.type_aliases import GymEnv, Schedule

from sb3_contrib.skew_fit.goalify import Goalify


class SkewFit(SAC):
    def __init__(
        self,
        env: Union[GymEnv, str],
        nb_models: int = 100,
        power: float = -1.0,
        num_presampled_goals: int = 2048,
        learning_rate: Union[float, Schedule] = 0.0003,
        buffer_size: int = 1000000,
        learning_starts: int = 100,
        batch_size: int = 256,
        tau: float = 0.005,
        gamma: float = 0.99,
        train_freq: Union[int, Tuple[int, str]] = 1,
        gradient_steps: int = 1,
        action_noise: Optional[ActionNoise] = None,
        replay_buffer_class: Optional[DictReplayBuffer] = None,
        replay_buffer_kwargs: Optional[Dict[str, Any]] = None,
        optimize_memory_usage: bool = False,
        ent_coef: Union[str, float] = "auto",
        target_update_interval: int = 1,
        target_entropy: Union[str, float] = "auto",
        use_sde: bool = False,
        sde_sample_freq: int = -1,
        use_sde_at_warmup: bool = False,
        tensorboard_log: Optional[str] = None,
        create_eval_env: bool = False,
        policy_kwargs: Optional[Dict[str, Any]] = None,
        surgeon: Optional[Surgeon] = None,
        verbose: int = 0,
        seed: Optional[int] = None,
        device: Union[th.device, str] = "auto",
        _init_setup_model: bool = True,
    ):
        # Wrap the env
        env = Goalify(
            env,
            nb_models=nb_models,
            gradient_steps=gradient_steps,
            batch_size=batch_size,
            power=power,
            num_presampled_goals=num_presampled_goals,
        )
        super().__init__(
            "MultiInputPolicy",
            env,
            learning_rate,
            buffer_size,
            learning_starts,
            batch_size,
            tau,
            gamma,
            train_freq,
            gradient_steps,
            action_noise,
            replay_buffer_class,
            replay_buffer_kwargs,
            optimize_memory_usage,
            ent_coef,
            target_update_interval,
            target_entropy,
            use_sde,
            sde_sample_freq,
            use_sde_at_warmup,
            tensorboard_log,
            create_eval_env,
            policy_kwargs,
            surgeon,
            verbose,
            seed,
            device,
            _init_setup_model,
        )
        env.set_buffer(self.replay_buffer)
