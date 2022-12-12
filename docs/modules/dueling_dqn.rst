.. _dueling_dqn:

.. automodule:: sb3_contrib.dueling_dqn


Dueling-DQN
===========

`Dueling DQN <https://arxiv.org/abs/1511.06581>`_ builds on `Deep Q-Network (DQN) <https://arxiv.org/abs/1312.5602>`_
and #TODO:


.. rubric:: Available Policies

.. autosummary::
    :nosignatures:

    MlpPolicy
    CnnPolicy
    MultiInputPolicy


Notes
-----

- Original paper: https://arxiv.org/abs/1511.06581


Can I use?
----------

-  Recurrent policies: ❌
-  Multi processing: ✔️
-  Gym spaces:


============= ====== ===========
Space         Action Observation
============= ====== ===========
Discrete      ✔️     ✔️
Box           ❌     ✔️
MultiDiscrete ❌     ✔️
MultiBinary   ❌     ✔️
Dict          ❌     ✔️
============= ====== ===========


Example
-------

.. code-block:: python

  import gym

  from sb3_contrib import DuelingDQN

  env = gym.make("CartPole-v1")

  model = DuelingDQN("MlpPolicy", env, verbose=1)
  model.learn(total_timesteps=10000, log_interval=4)
  model.save("dueling_dqn_cartpole")

  del model # remove to demonstrate saving and loading

  model = DuelingDQN.load("dueling_dqn_cartpole")

  obs = env.reset()
  while True:
      action, _states = model.predict(obs, deterministic=True)
      obs, reward, done, info = env.step(action)
      env.render()
      if done:
          obs = env.reset()


Results
-------

Result on Atari environments (10M steps, Pong and Breakout) and classic control tasks using 3 and 5 seeds.

The complete learning curves are available in the `associated PR <https://github.com/Stable-Baselines-Team/stable-baselines3-contrib/pull/126>`_. #TODO:


.. note::

    DuelingDQN implementation was validated against #TODO: valid the results


============ =========== ===========
Environments DuelingDQN     DQN
============ =========== ===========
Breakout                 ~300
Pong                     ~20
CartPole                 500 +/- 0
MountainCar              -107 +/- 4
LunarLander              195 +/- 28
Acrobot                  -74 +/- 2
============ =========== ===========

#TODO: Fill the tabular

How to replicate the results?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Clone RL-Zoo fork and checkout the branch ``feat/dueling-dqn``:

.. code-block:: bash

  git clone https://github.com/DLR-RM/rl-baselines3-zoo/
  cd rl-baselines3-zoo/
  git checkout feat/dueling-dqn #TODO: create this branch

Run the benchmark (replace ``$ENV_ID`` by the envs mentioned above):

.. code-block:: bash

  python train.py --algo dueling_dqn --env $ENV_ID --eval-episodes 10 --eval-freq 10000 #TODO: check if that command line works


Plot the results:

.. code-block:: bash

  python scripts/all_plots.py -a dueling_dqn -e Breakout Pong -f logs/ -o logs/dueling_dqn_results #TODO: check if that command line works
  python scripts/plot_from_file.py -i logs/dueling_dqn_results.pkl -latex -l Dueling DQN #TODO: check if that command line works



Parameters
----------

.. autoclass:: DuelingDQN
  :members:
  :inherited-members:

.. _dueling_dqn_policies:

Dueling DQN Policies
--------------------

.. autoclass:: MlpPolicy
  :members:
  :inherited-members:

.. autoclass:: sb3_contrib.dueling_dqn.policies.DuelingDQNPolicy
  :members:
  :noindex:

.. autoclass:: CnnPolicy
  :members:

.. autoclass:: MultiInputPolicy
  :members:
