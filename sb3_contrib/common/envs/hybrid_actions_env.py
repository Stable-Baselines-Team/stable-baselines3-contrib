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
        max_steps: float = 100
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
    
    def reset(self, **kwargs) -> tuple[np.ndarray, dict]:
        """
        Reset the environment to an initial state and return the initial observation.
        """
        self.agent_pos = self.np_random.uniform(-self.arena_size, self.arena_size, size=2).astype(np.float32)
        self.target_pos = self.np_random.uniform(-self.arena_size, self.arena_size, size=2).astype(np.float32)
        self.catches_used = 0
        self.step_count = 0
        return self._get_obs(), {}

    def step(self, concat_actions: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict]:
        """
        Step the environment with the given actions.
        Compatible with VecEnv interface.
        
        :param concat_actions: A concatenated array containing both discrete and continuous actions.
        :return: observation, reward, terminated, truncated, info
        """
        # Unstack the concatenated actions back into discrete and continuous
        n_discrete_actions = self.action_space[0].nvec.size if isinstance(self.action_space[0], spaces.MultiDiscrete) else 1
        actions_d = concat_actions[:n_discrete_actions].astype(int)
        actions_c = concat_actions[n_discrete_actions:]
        
        # actual step logic
        return self._step(actions_d, actions_c)

    def _step(self, actions_d: np.ndarray, actions_c: np.ndarray) -> tuple[np.ndarray, float, bool, dict]:
        """
        Take a step in the environment using the provided actions.
        
        :param actions_d: Discrete action
        :param actions_c: Continuous action
        :return: observation, reward, done, info
        """
        action_d = actions_d.item()  # only 1 discrete action -> extract scalar
        dir_vec = actions_c
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
            reward -= 0.05    # catch attempt penalty
            self.catches_used += 1
            dist = np.linalg.norm(self.agent_pos - self.target_pos)
            if dist <= self.catch_radius:
                reward = 1.0    # caught the target
                terminated = True   # target caught: natural termination
            else:
                if self.catches_used >= self.max_catches:
                    reward = -1.0   # failed to catch within max catches
                    terminated = True   # max catches reached: natural termination

        self.step_count += 1
        if self.step_count >= self.max_steps:
            reward = -1.0   # failed to catch within max steps
            truncated = True    # max steps reached: truncation

        obs = self._get_obs()
        info = {"caught": (reward > 0)}
        return obs, float(reward), terminated, truncated, info

    def _get_obs(self) -> np.ndarray:
        """
        Get the current observation.
        """
        steps_left_norm = self.step_count / self.max_steps
        catches_left_norm = (self.max_catches - self.catches_used) / self.max_catches
        obs = np.concatenate((
            self.agent_pos,
            self.target_pos,
            np.array([catches_left_norm, steps_left_norm])
        ), dtype=np.float32)
        return obs
