import gym
import numpy as np
from gym.spaces import Box, Dict
from stable_baselines3.common.env_util import make_vec_env

from sb3_contrib.go_explore.archive import ArchiveBuffer
from sb3_contrib.go_explore.cells import cell_is_obs
from sb3_contrib.go_explore.go_explore import Goalify


class MyEnv(gym.Env):
    observation_space = Box(-1, 1, (1,))
    action_space = Box(-1, 1, (1,))

    def reset(self):
        self.pos = 0
        return np.array([self.pos])

    def step(self, action):
        self.pos += action
        return np.array([self.pos]), 0, False, {}


def test_sample_trajectory():
    archive = ArchiveBuffer(
        buffer_size=100,
        observation_space=Dict({"observation": Box(-1, 1, (2,)), "goal": Box(-1, 1, (2,))}),
        action_space=Box(-1, 1, (2,)),
        env=gym.Env(),
        cell_factory=cell_is_obs,
    )
    for i in range(10):
        archive.add(
            obs={"observation": np.array([[i % 4, 0]]), "goal": np.array([[0, 0]])},
            next_obs={"observation": np.array([[i % 4 + 1, 0]]), "goal": np.array([[0, 0]])},
            action=np.array([[0, 0]]),
            reward=np.array([0]),
            done=np.array([False]),
            infos=[{}],
        )

    archive.recompute_cells()
    sampled_trajectories = [archive.sample_trajectory().tolist() for _ in range(10)]
    possible_trajectories = [
        [[1.0, 0.0]],
        [[1.0, 0.0], [2.0, 0.0]],
        [[1.0, 0.0], [2.0, 0.0], [3.0, 0.0]],
        [[1.0, 0.0], [2.0, 0.0], [3.0, 0.0], [4.0, 0.0]],
    ]
    assert np.all(
        [trajectory in possible_trajectories for trajectory in sampled_trajectories]
    ), "Sampled unpossible trajctories."
    assert np.all(
        [trajectory in sampled_trajectories for trajectory in possible_trajectories]
    ), "All possible trajectories are not sampled."


def test_goalify():
    env = make_vec_env(MyEnv, wrapper_class=Goalify, wrapper_kwargs=dict(cell_factory=cell_is_obs))
    archive = ArchiveBuffer(
        buffer_size=100,
        observation_space=env.observation_space,
        action_space=env.action_space,
        env=env,
        cell_factory=cell_is_obs,
    )
    env.get_attr("set_archive")[0](archive)
    obs = env.reset()
    action = np.zeros(1)
    next_obs, reward, done, info = env.step(action)
    archive.add(obs, next_obs, action, reward, done, info)
