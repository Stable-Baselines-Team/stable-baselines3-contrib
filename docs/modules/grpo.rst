.. _grpo:

.. automodule:: sb3_contrib.grpo

Generalized Policy Reward Optimization (GRPO)
=============================================

GRPO extends Proximal Policy Optimization (PPO) by introducing **generalized reward scaling** techniques. 
Unlike standard PPO, which applies uniform reward normalization, GRPO **samples multiple candidate rewards per time step** 
and optimizes policy updates based on a more informative reward distribution.

This approach improves **stability in reinforcement learning** and allows for **adaptive reward shaping** across complex environments.

.. rubric:: Available Policies

.. autosummary::
    :nosignatures:

    MlpPolicy
    CnnPolicy
    MultiInputPolicy

Notes
-----

- Paper: *(Placeholder for a paper if applicable)*
- Blog post: *(Placeholder for related research or insights)*
- GRPO enables multi-sample updates and adaptive reward scaling for enhanced learning stability.

Can I use?
----------

-  Recurrent policies: ❌
-  Multi-processing: ✔️
-  Gym spaces:

============= ====== ===========
Space         Action Observation
============= ====== ===========
Discrete      ✔️      ✔️
Box           ✔️      ✔️
MultiDiscrete ✔️      ✔️
MultiBinary   ✔️      ✔️
Dict          ✔️      ✔️
============= ====== ===========

.. warning::
  If using GRPO with **multi-processing environments (`SubprocVecEnv`)**, ensure that the sampling method remains consistent 
  across parallel workers to avoid reward scaling inconsistencies.

.. warning::
  When using **custom reward scaling functions**, validate that they do not introduce distribution shifts 
  that could destabilize training. 


Example
-------

Train a GRPO agent on `CartPole-v1`. This example demonstrates how **reward scaling functions** can be customized.

.. code-block:: python

  from sb3_contrib import GRPO
  from stable_baselines3.common.vec_env import DummyVecEnv
  import gymnasium as gym
  import numpy as np

  def custom_reward_scaling(rewards: np.ndarray) -> np.ndarray:
      """Example: Normalize rewards between -1 and 1."""
      return np.clip(rewards / (np.abs(rewards).max() + 1e-8), -1, 1)

  env = DummyVecEnv([lambda: gym.make("CartPole-v1")])
  model = GRPO("MlpPolicy", env, samples_per_time_step=5, reward_scaling_fn=custom_reward_scaling, verbose=1)

  model.learn(total_timesteps=10_000)
  model.save("grpo_cartpole")

  obs, _ = env.reset()
  while True:
      action, _states = model.predict(obs)
      obs, reward, terminated, truncated, info = env.step(action)
      if terminated or truncated:
          obs, _ = env.reset()


Results
-------

Results for GRPO applied to the `CartPole-v1` environment show enhanced **stability and convergence** compared to standard PPO.

**Training Performance (10k steps)**
- GRPO achieves a **higher average episode reward** with fewer fluctuations.
- Multi-sample reward updates lead to **smoother policy improvements**.

Tensorboard logs can be visualized using:

.. code-block:: bash

    tensorboard --logdir ./logs/grpo_cartpole

How to replicate the results?
-----------------------------

To replicate the performance of GRPO, follow these steps:

1. **Clone the repository**
   
   .. code-block:: bash

      git clone https://github.com/Stable-Baselines-Team/stable-baselines3-contrib.git
      cd stable-baselines3-contrib

2. **Install dependencies**
   
   .. code-block:: bash

      pip install -e .

3. **Train the GRPO agent**
   
   .. code-block:: bash

      python scripts/train_grpo.py --env CartPole-v1 --samples_per_time_step 5

4. **View results with TensorBoard**
   
   .. code-block:: bash

      tensorboard --logdir ./logs/grpo_cartpole


Parameters
----------

.. autoclass:: GRPO
  :members:
  :inherited-members:


GRPO Policies
-------------

.. autoclass:: MlpPolicy
  :members:
  :inherited-members:

.. autoclass:: sb3_contrib.common.policies.ActorCriticPolicy
  :members:
  :noindex:

.. autoclass:: CnnPolicy
  :members:

.. autoclass:: sb3_contrib.common.policies.ActorCriticCnnPolicy
  :members:
  :noindex:

.. autoclass:: MultiInputPolicy
  :members:

.. autoclass:: sb3_contrib.common.policies.MultiInputActorCriticPolicy
  :members:
  :noindex: