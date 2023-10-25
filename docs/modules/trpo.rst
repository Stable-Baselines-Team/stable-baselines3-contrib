.. _trpo:

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

  import gymnasium as gym
  import numpy as np

  from sb3_contrib import TRPO

  env = gym.make("Pendulum-v1", render_mode="human")

  model = TRPO("MlpPolicy", env, verbose=1)
  model.learn(total_timesteps=10_000, log_interval=4)
  model.save("trpo_pendulum")

  del model # remove to demonstrate saving and loading

  model = TRPO.load("trpo_pendulum")

  obs, _ = env.reset()
  while True:
      action, _states = model.predict(obs, deterministic=True)
      obs, reward, terminated, truncated, info = env.step(action)
      env.render()
      if terminated or truncated:
        obs, _ = env.reset()


Results
-------

Result on the MuJoCo benchmark (1M steps on ``-v3`` envs with MuJoCo v2.1.0) using 3 seeds.
The complete learning curves are available in the `associated PR <https://github.com/Stable-Baselines-Team/stable-baselines3-contrib/pull/40>`_.


===================== ============
Environments          TRPO
===================== ============
HalfCheetah           1803 +/- 46
Ant                   3554 +/- 591
Hopper                3372 +/- 215
Walker2d              4502 +/- 234
Swimmer               359 +/- 2
===================== ============


How to replicate the results?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Clone RL-Zoo and checkout the branch ``feat/trpo``:

.. code-block:: bash

  git clone https://github.com/cyprienc/rl-baselines3-zoo
  cd rl-baselines3-zoo/

Run the benchmark (replace ``$ENV_ID`` by the envs mentioned above):

.. code-block:: bash

  python train.py --algo trpo --env $ENV_ID --n-eval-envs 10 --eval-episodes 20 --eval-freq 50000


Plot the results:

.. code-block:: bash

  python scripts/all_plots.py -a trpo -e HalfCheetah Ant Hopper Walker2d Swimmer -f logs/ -o logs/trpo_results
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
