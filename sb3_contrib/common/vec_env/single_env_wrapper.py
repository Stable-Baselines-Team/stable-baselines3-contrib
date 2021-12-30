import numpy as np
from stable_baselines3.common.vec_env.base_vec_env import VecEnv, VecEnvObs, VecEnvStepReturn, VecEnvWrapper


class VecSingleEnvWrapper(VecEnvWrapper):
    """
    A vectorized wrapper for extracting single env.

    :param venv: The vectorized environment
    :param index: The index of the env
    """

    def __init__(self, venv: VecEnv, index: int = 0):
        self.index = index
        assert 0 <= index <= venv.num_envs, f"Invalid env index: {index} (n_envs={venv.num_envs})"
        super().__init__(venv=venv)
        self.num_envs = 1

    def reset(self) -> VecEnvObs:
        return self.venv.env_method("reset", indices=self.index)[0].reshape(1, -1)

    def step(self, action: np.ndarray) -> VecEnvStepReturn:
        obs, reward, done, infos = self.venv.env_method("step", action=action[0], indices=self.index)[0]
        if done:
            infos["terminal_observation"] = obs
            obs = self.reset()
        return obs.reshape(1, -1), np.array([reward]), np.array([done]), [infos]

    def step_wait(self) -> VecEnvStepReturn:
        return self.venv.step_wait()
