.. _bdpi:

.. automodule:: sb3_contrib.bdpi

BDPI
====

`Bootstrapped Dual Policy Iteration <https://arxiv.org/abs/1903.04193>`_ is an actor-critic algorithm for
discrete action spaces. The distinctive components of BDPI are as follows:

- Like Bootstrapped DQN, it uses several critics, with each critic having a Qa and Qb network
  (like Clipped DQN).
- The BDPI critics, inspired from the DQN literature, are therefore off-policy. They don't know
  about the actor, and do not use any form of off-policy corrections to evaluate the actor. They
  instead directly approximate Q*, the optimal value function.
- The actor is trained with an equation inspired from Conservative Policy Iteration, instead of
  Policy Gradient (as used by A2C, PPO, SAC, DDPG, etc). This use of Conservative Policy Iteration
  is what allows the BDPI actor to be compatible with off-policy critics.

As a result, BDPI can be configured to be highly sample-efficient, at the cost of compute efficiency.
The off-policy critics can learn aggressively (many samples, many gradient steps), as they don't have
to remain close to the actor. The actor then learns from a mixture of high-quality critics, leading
to good exploration even in challenging environments (see the Table environment described in the paper
linked above).

Can I use?
----------

-  Recurrent policies: ❌
-  Multi processing: ✔️
-  Gym spaces:


============= ====== ===========
Space         Action Observation
============= ====== ===========
Discrete      ✔️      ✔️
Box           ❌     ✔️
MultiDiscrete ❌     ✔️
MultiBinary   ❌     ✔️
Dict          ❌     ✔️
============= ====== ===========

Example
-------

Train a BDPI agent on ``LunarLander-v2``, with hyper-parameters tuned by Optuna in rl-baselines3-zoo:

.. code-block:: python

  import gym

  from sb3_contrib import BDPI

  model = BDPI(
    "MlpPolicy",
    'LunarLander-v2',
    actor_lr=0.01,       # How fast the actor pursues the greedy policy of the critics
    critic_lr=0.234,     # Q-Learning learning rate of the critics
    batch_size=256,      # 256 experiences sampled from the buffer every time-step, for every critic
    buffer_size=100000,
    gradient_steps=64,   # Actor and critics fit for 64 gradient steps per time-step
    learning_rate=0.00026, # Adam optimizer learning rate
    policy_kwargs=dict(net_arch=[64, 64], n_critics=8), # 8 critics
    verbose=1,
    tensorboard_log='./tb_log'
  )

  model.learn(total_timesteps=50000)
  model.save("bdpi_lunarlander")

  del model # remove to demonstrate saving and loading

  model = BDPI.load("bdpi_lunarlander")

  obs = env.reset()
  while True:
      action, _states = model.predict(obs)
      obs, rewards, dones, info = env.step(action)
      env.render()


Results
-------

LunarLander
^^^^^^^^^^^

Results for BDPI are available in `this Github issue <https://github.com/DLR-RM/stable-baselines3/issues/499>`_.

How to replicate the results?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Clone the `rl-zoo repo <https://github.com/DLR-RM/rl-baselines3-zoo>`_:

.. code-block:: bash

  git clone https://github.com/DLR-RM/rl-baselines3-zoo
  cd rl-baselines3-zoo/


Run the benchmark (replace ``$ENV_ID`` by the envs mentioned above, and ``$N`` with
the number of CPU cores in your machine):

.. code-block:: bash

  python train.py --algo bdpi --env $ENV_ID --eval-episodes 10 --eval-freq 10000 -params threads:$N


Plot the results (here for LunarLander only):

.. code-block:: bash

  python scripts/all_plots.py -a bdpi -e LunarLander -f logs/ -o logs/bdpi_results
  python scripts/plot_from_file.py -i logs/bdpi_results.pkl -latex -l BDPI


Parameters
----------

.. autoclass:: BDPI
  :members:
  :inherited-members:


BDPI Policies
-------------

.. autoclass:: MlpPolicy
  :members:

.. autoclass:: CnnPolicy
  :members:

.. autoclass:: MultiInputPolicy
  :members:
