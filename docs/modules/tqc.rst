.. _tqc:

.. automodule:: sb3_contrib.tqc


TQC
===

Controlling Overestimation Bias with Truncated Mixture of Continuous Distributional Quantile Critics (TQC).
Truncated Quantile Critics (TQC) builds on SAC, TD3 and QR-DQN, making use of quantile regression to predict a distribution for the value function (instead of a mean value).
It truncates the quantiles predicted by different networks (a bit as it is done in TD3).


.. rubric:: Available Policies

.. autosummary::
    :nosignatures:

    MlpPolicy
    CnnPolicy
    MultiInputPolicy


Notes
-----

- Original paper: https://arxiv.org/abs/2005.04269
- Original Implementation: https://github.com/bayesgroup/tqc_pytorch


Can I use?
----------

-  Recurrent policies: ❌
-  Multi processing: ✔️
-  Gym spaces:


============= ====== ===========
Space         Action Observation
============= ====== ===========
Discrete      ❌      ✔️
Box           ✔️       ✔️
MultiDiscrete ❌      ✔️
MultiBinary   ❌      ✔️
Dict          ❌     ✔️
============= ====== ===========


Example
-------

.. code-block:: python

  import gymnasium as gym
  import numpy as np

  from sb3_contrib import TQC

  env = gym.make("Pendulum-v1", render_mode="human")

  policy_kwargs = dict(n_critics=2, n_quantiles=25)
  model = TQC("MlpPolicy", env, top_quantiles_to_drop_per_net=2, verbose=1, policy_kwargs=policy_kwargs)
  model.learn(total_timesteps=10_000, log_interval=4)
  model.save("tqc_pendulum")

  del model # remove to demonstrate saving and loading

  model = TQC.load("tqc_pendulum")

  obs, _ = env.reset()
  while True:
      action, _states = model.predict(obs, deterministic=True)
      obs, reward, terminated, truncated, info = env.step(action)
      env.render()
      if terminated or truncated:
        obs, _ = env.reset()


Results
-------

Result on the PyBullet benchmark (1M steps) and on BipedalWalkerHardcore-v3 (2M steps)
using 3 seeds.
The complete learning curves are available in the `associated PR <https://github.com/Stable-Baselines-Team/stable-baselines3-contrib/pull/4>`_.

The main difference with SAC is on harder environments (BipedalWalkerHardcore, Walker2D).


.. note::

  Hyperparameters from the `gSDE paper <https://arxiv.org/abs/2005.05719>`_ were used (as they are tuned for SAC on PyBullet envs),
  including using gSDE for the exploration and not the unstructured Gaussian noise
  but this should not affect results in simulation.


.. note::

  We are using the open source PyBullet environments and not the MuJoCo simulator (as done in the original paper).
  You can find a complete benchmark on PyBullet envs in the `gSDE paper <https://arxiv.org/abs/2005.05719>`_
  if you want to compare TQC results to those of A2C/PPO/SAC/TD3.


===================== ============ ============
Environments          SAC          TQC
===================== ============ ============
\                     gSDE         gSDE
HalfCheetah           2984 +/- 202 3041 +/- 157
Ant                   3102 +/- 37  3700 +/- 37
Hopper                2262 +/- 1   2401 +/- 62*
Walker2D              2136 +/- 67  2535 +/- 94
BipedalWalkerHardcore 13 +/- 18    228 +/- 18
===================== ============ ============

\* with tuned hyperparameter ``top_quantiles_to_drop_per_net`` taken from the original paper


How to replicate the results?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Clone RL-Zoo and checkout the branch ``feat/tqc``:

.. code-block:: bash

  git clone https://github.com/DLR-RM/rl-baselines3-zoo
  cd rl-baselines3-zoo/
  git checkout feat/tqc

Run the benchmark (replace ``$ENV_ID`` by the envs mentioned above):

.. code-block:: bash

  python train.py --algo tqc --env $ENV_ID --eval-episodes 10 --eval-freq 10000


Plot the results:

.. code-block:: bash

  python scripts/all_plots.py -a tqc -e HalfCheetah Ant Hopper Walker2D BipedalWalkerHardcore -f logs/ -o logs/tqc_results
  python scripts/plot_from_file.py -i logs/tqc_results.pkl -latex -l TQC

Comments
--------

This implementation is based on SB3 SAC implementation and uses the code from the original TQC implementation for the quantile huber loss.


Parameters
----------

.. autoclass:: TQC
  :members:
  :inherited-members:

.. _tqc_policies:

TQC Policies
-------------

.. autoclass:: MlpPolicy
  :members:
  :inherited-members:

.. autoclass:: sb3_contrib.tqc.policies.TQCPolicy
  :members:
  :noindex:

.. autoclass:: CnnPolicy
  :members:
