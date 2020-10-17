.. _tqc:

.. automodule:: sb3_contrib.tqc


TQC
===

Controlling Overestimation Bias with Truncated Mixture of Continuous Distributional Quantile Critics.
This paper build on SAC, TD3 and QR-DQN, making use of quantile regression to predict a distribution for the value function (instead of a mean value).
It truncates the quantiles predicted by different networks (a bit as it is done in TD3).
This is for continuous actions only.


.. rubric:: Available Policies

.. autosummary::
    :nosignatures:

    MlpPolicy
    CnnPolicy


Notes
-----

- Original paper: https://arxiv.org/abs/2005.04269
- Original Implementation: https://github.com/bayesgroup/tqc_pytorch


Can I use?
----------

-  Recurrent policies: ❌
-  Multi processing: ❌
-  Gym spaces:


============= ====== ===========
Space         Action Observation
============= ====== ===========
Discrete      ❌      ✔️
Box           ✔️       ✔️
MultiDiscrete ❌      ✔️
MultiBinary   ❌      ✔️
============= ====== ===========


Example
-------

.. code-block:: python

  import gym
  import numpy as np

  from sb3_contrib import TQC

  env = gym.make("Pendulum-v0")

  policy_kwargs = dict(n_critics=2, n_quantiles=25)
  model = TQC("MlpPolicy", env, top_quantiles_to_drop_per_net=2, verbose=1, policy_kwargs=policy_kwargs)
  model.learn(total_timesteps=10000, log_interval=4)
  model.save("tqc_pendulum")

  del model # remove to demonstrate saving and loading

  model = TQC.load("tqc_pendulum")

  obs = env.reset()
  while True:
      action, _states = model.predict(obs, deterministic=True)
      obs, reward, done, info = env.step(action)
      env.render()
      if done:
        obs = env.reset()

Example
-------

A minimal example on how to use the feature (full, runnable code).

Results
-------

A description and comparison of results (e.g. how the change improved results over the non-changed algorithm), if
applicable.

Include the expected results from the work that originally proposed the method (e.g. original paper).

Include the code to replicate these results or a link to repository/branch where the code can be found.
Use `rl-baselines3-zoo <https://github.com/DLR-RM/rl-baselines3-zoo>`_ if possible, fork it, create a new branch
and share the code to replicate results there.

Comments
--------

Comments regarding the implementation, e.g. missing parts, uncertain parts, differences
to the original implementation.

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

.. .. autoclass:: CnnPolicy
..   :members:
..   :inherited-members:
