import gym
import gym_continuous_maze
import numpy as np
from render_maze import render_and_save

from sb3_contrib import SkewFit

env = gym.make("ContinuousMaze-v0")
model = SkewFit(
    env,
    nb_models=32,
    power=-1.2,
    num_presampled_goals=512,
    verbose=1
)

model.learn(500000)
env = model.env.envs[0].env
goal_distribution = env.get_goal_distribution()
goals = np.array([env.sample_goal(goal_distribution, power=-1.2, num_presampled_goals=512) for _ in range(500)])
im = render_and_save(model.env.envs, "skew_fit_500k.png", goals)
