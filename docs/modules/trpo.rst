.. _tqc:

.. automodule:: sb3_contrib.trpo

TRPO
====

`Trust Region Policy Optimization (TRPO) <https://arxiv.org/abs/1502.05477>`_
is an iterative approach for optimizing policies with guaranteed monotonic improvement.
.. rubric:: Available Policies

.. autosummary::
    :nosignatures:

    MlpPolicy
    CnnPolicy
    MultiInputPolicy



Notes
-----

-  Original paper:  https://arxiv.org/abs/1502.05477
-  OpenAI blog post: https://blog.openai.com/openai-baselines-ppo/


Can I use?
----------

-  Recurrent policies: ❌
-  Multi processing: ✔️
-  Gym spaces:


============= ====== ===========
Space         Action Observation
============= ====== ===========
Discrete      ✔️      ✔️
Box           ✔️      ✔️
MultiDiscrete ✔️      ✔️
MultiBinary   ✔️      ✔️
Dict          ❌      ✔️
============= ====== ===========


Example
-------

.. code-block:: python

  import gym
  import numpy as np

  from sb3_contrib import TRPO

  env = gym.make("Pendulum-v0")

  model = TRPO("MlpPolicy", env, verbose=1)
  model.learn(total_timesteps=10000, log_interval=4)
  model.save("trpo_pendulum")

  del model # remove to demonstrate saving and loading

  model = TRPO.load("trpo_pendulum")

  obs = env.reset()
  while True:
      action, _states = model.predict(obs, deterministic=True)
      obs, reward, done, info = env.step(action)
      env.render()
      if done:
        obs = env.reset()


Results
-------

Result on the PyBullet benchmark (1M steps) using 3 seeds.
The complete learning curves are available in the `associated PR <https://github.com/Stable-Baselines-Team/stable-baselines3-contrib/pull/40>`_.


===================== ============ ============
Environments          PPO          TRPO
===================== ============ ============
HalfCheetah           1976 +/- 479 0000 +/- 157
Ant                   2364 +/- 120 0000 +/- 37
Hopper                1567 +/- 339 0000 +/- 62
Walker2D              1230 +/- 147 0000 +/- 94
===================== ============ ============


How to replicate the results?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Clone RL-Zoo and checkout the branch ``feat/trpo``:

.. code-block:: bash

  git clone https://github.com/DLR-RM/rl-baselines3-zoo
  cd rl-baselines3-zoo/
  git checkout feat/trpo

Run the benchmark (replace ``$ENV_ID`` by the envs mentioned above):

.. code-block:: bash

  python train.py --algo tqc --env $ENV_ID --eval-episodes 10 --eval-freq 10000


Plot the results:

.. code-block:: bash

  python scripts/all_plots.py -a trpo -e HalfCheetah Ant Hopper Walker2D BipedalWalkerHardcore -f logs/ -o logs/trpo_results
  python scripts/plot_from_file.py -i logs/trpo_results.pkl -latex -l TRPO


Parameters
----------

.. autoclass:: TRPO
  :members:
  :inherited-members:

.. _trpo_policies:

TRPO Policies
-------------

.. autoclass:: MlpPolicy
  :members:
  :inherited-members:

.. autoclass:: stable_baselines3.common.policies.ActorCriticPolicy
  :members:
  :noindex:

.. autoclass:: CnnPolicy
  :members:

.. autoclass:: stable_baselines3.common.policies.ActorCriticCnnPolicy
  :members:
  :noindex:

.. autoclass:: MultiInputPolicy
  :members:

.. autoclass:: stable_baselines3.common.policies.MultiInputActorCriticPolicy
  :members:
  :noindex:
