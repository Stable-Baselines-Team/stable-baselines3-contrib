.. _qrdqn:

.. automodule:: sb3_contrib.qrdqn


QR-DQN
======

`Quantile Regression DQN (QR-DQN) <https://arxiv.org/abs/1710.10044>`_ builds on `Deep Q-Network (DQN) <https://arxiv.org/abs/1312.5602>`_
and make use of quantile regression to explicitly model the `distribution over returns <https://arxiv.org/abs/1707.06887>`_,
instead of predicting the mean return (DQN).


.. rubric:: Available Policies

.. autosummary::
    :nosignatures:

    MlpPolicy
    CnnPolicy


Notes
-----

- Original paper: https://arxiv.org/abs/1710.100442
- Distributional RL (C51): https://arxiv.org/abs/1707.06887


Can I use?
----------

-  Recurrent policies: ❌
-  Multi processing: ❌
-  Gym spaces:


============= ====== ===========
Space         Action Observation
============= ====== ===========
Discrete      ✔      ✔
Box           ❌      ✔
MultiDiscrete ❌      ✔
MultiBinary   ❌      ✔
============= ====== ===========


Example
-------

.. code-block:: python

  import gym

  from sb3_contrib import QRDQN

  env = gym.make("CartPole-v1")

  policy_kwargs = dict(n_quantiles=50)
  model = QRDQN("MlpPolicy", env, policy_kwargs=policy_kwargs, verbose=1)
  model.learn(total_timesteps=10000, log_interval=4)
  model.save("qrdqn_cartpole")

  del model # remove to demonstrate saving and loading

  model = QRDQN.load("qrdqn_cartpole")

  obs = env.reset()
  while True:
      action, _states = model.predict(obs, deterministic=True)
      obs, reward, done, info = env.step(action)
      env.render()
      if done:
        obs = env.reset()


Results
-------



Parameters
----------

.. autoclass:: QRDQN
  :members:
  :inherited-members:

.. _qrdqn_policies:

QR-DQN Policies
---------------

.. autoclass:: MlpPolicy
  :members:
  :inherited-members:

.. autoclass:: sb3_contrib.qrdqn.policies.QRDQNPolicy
  :members:
  :noindex:

.. autoclass:: CnnPolicy
  :members:
