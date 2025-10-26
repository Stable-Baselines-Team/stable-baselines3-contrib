from typing import Optional, Tuple, Dict

import gymnasium as gym
from gymnasium import spaces
import numpy as np


class CatchingPointEnv(gym.Env):
    """
    Enviornment for Hybrid PPO for the 'Catching Point' task of the paper
    'Hybrid Actor-Critic Reinforcement Learning in Parameterized Action Space', Fan et al.
    (https://arxiv.org/pdf/1903.01344)
    """

    def __init__(
        self,
        arena_size: float = 1.0,
        move_dist: float = 0.05,
        catch_radius: float = 0.05,
        max_catches: float = 10,
        max_steps: float = 500
    ):
        super().__init__()
        self.max_steps = max_steps
        self.max_catches = max_catches
        self.arena_size = arena_size
        self.move_dist = move_dist
        self.catch_radius = catch_radius

        # action space
        self.action_space = spaces.Tuple(
            spaces=(
                spaces.MultiDiscrete([2]),  # MOVE=0, CATCH=1
                spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)  # direction
            )
        )

        # observation: [agent_x, agent_y, target_x, target_y, catches_left, step_norm]
        obs_low = np.array([-arena_size, -arena_size, -arena_size, -arena_size, 0.0, 0.0], dtype=np.float32)
        obs_high= np.array([ arena_size,  arena_size,  arena_size,  arena_size, float(max_catches), 1.0], dtype=np.float32)
        self.observation_space = spaces.Box(obs_low, obs_high, dtype=np.float32)
    
    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None) -> tuple[np.ndarray, dict]:
        """
        Reset the environment to an initial state and return the initial observation.

        :param seed: The seed for random number generation
        :param options: Additional options for environment reset, e.g., {'difficulty': 'hard'}
        :return: Tuple of (observation, info)
        """
        super().reset(seed=seed)

        # Handle options (none used in this environment currently, but following the gymnasium API)
        if options is not None:
            # Example of how to handle options if needed:
            # if 'difficulty' in options:
            #     self.move_dist = 0.03 if options['difficulty'] == 'hard' else 0.05
            pass

        self.agent_pos = self.np_random.uniform(-self.arena_size, self.arena_size, size=2).astype(np.float32)
        self.target_pos = self.np_random.uniform(-self.arena_size, self.arena_size, size=2).astype(np.float32)
        self.catches_used = 0
        self.step_count = 0
        
        obs = self._get_obs()
        info = {}  # Additional info dict, empty for now but could include initialization info
        return obs, info

    def step(self, action: Tuple[np.ndarray, np.ndarray]) -> Tuple[np.ndarray, float, bool, bool, Dict[str, bool]]:
        """
        Take a step in the environment using the provided action.
        
        :param action: A tuple containing the discrete action and continuous parameters.
        :return: Tuple of (observation, reward, terminated, truncated, info)
        """
        action_d = int(action[0][0])
        dir_vec = action[1]
        reward = 0.0
        terminated = False
        truncated = False

        # step penalty
        reward = -0.01

        # MOVE
        if action_d == 0:
            norm = np.linalg.norm(dir_vec)
            dir_u = dir_vec / norm
            self.agent_pos = (self.agent_pos + dir_u * self.move_dist).astype(np.float32)
            # clamp to arena
            self.agent_pos = np.clip(self.agent_pos, -self.arena_size, self.arena_size)
        
        # CATCH
        else:
            self.catches_used += 1
            dist = np.linalg.norm(self.agent_pos - self.target_pos)
            if dist <= self.catch_radius:
                reward = 1.0    # caught the target
                terminated = True  # Natural termination
            else:
                if self.catches_used >= self.max_catches:
                    terminated = True  # Natural termination

        self.step_count += 1
        if self.step_count >= self.max_steps:
            truncated = True  # Episode truncated due to time limit

        obs = self._get_obs()
        info = {"caught": (reward > 0)}
        return obs, float(reward), terminated, truncated, info

    def _get_obs(self) -> np.ndarray:
        """
        Get the current observation.
        """
        step_norm = self.step_count / self.max_steps
        catches_left = self.max_catches - self.catches_used
        obs = np.concatenate((
            self.agent_pos,
            self.target_pos,
            np.array([catches_left, step_norm], dtype=np.float32)
        ))
        return obs
