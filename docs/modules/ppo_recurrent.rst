.. _ppo_lstm:

.. automodule:: sb3_contrib.ppo_recurrent

Recurrent PPO
=============

Implementation of recurrent policies for the Proximal Policy Optimization (PPO)
algorithm. Other than adding support for recurrent policies (LSTM here), the behavior is the same as in SB3's core PPO algorithm.


.. rubric:: Available Policies

.. autosummary::
    :nosignatures:

    MlpLstmPolicy
    CnnLstmPolicy
    MultiInputLstmPolicy


Notes
-----

- Blog post: https://iclr-blog-track.github.io/2022/03/25/ppo-implementation-details/


Can I use?
----------

-  Recurrent policies: ✔️
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

.. note::

  It is particularly important to pass the ``lstm_states``
  and ``episode_start`` argument to the ``predict()`` method,
  so the cell and hidden states of the LSTM are correctly updated.


.. code-block:: python

  import numpy as np

  from sb3_contrib import RecurrentPPO
  from stable_baselines3.common.evaluation import evaluate_policy

  model = RecurrentPPO("MlpLstmPolicy", "CartPole-v1", verbose=1)
  model.learn(5000)

  vec_env = model.get_env()
  mean_reward, std_reward = evaluate_policy(model, vec_env, n_eval_episodes=20, warn=False)
  print(mean_reward)

  model.save("ppo_recurrent")
  del model # remove to demonstrate saving and loading

  model = RecurrentPPO.load("ppo_recurrent")

  obs = vec_env.reset()
  # cell and hidden state of the LSTM
  lstm_states = None
  num_envs = 1
  # Episode start signals are used to reset the lstm states
  episode_starts = np.ones((num_envs,), dtype=bool)
  while True:
      action, lstm_states = model.predict(obs, state=lstm_states, episode_start=episode_starts, deterministic=True)
      obs, rewards, dones, info = vec_env.step(action)
      episode_starts = dones
      vec_env.render("human")



Results
-------

Report on environments with masked velocity (with and without framestack) can be found here: https://wandb.ai/sb3/no-vel-envs/reports/PPO-vs-RecurrentPPO-aka-PPO-LSTM-on-environments-with-masked-velocity--VmlldzoxOTI4NjE4

``RecurrentPPO`` was evaluated against PPO on:

- PendulumNoVel-v1
- LunarLanderNoVel-v2
- CartPoleNoVel-v1
- MountainCarContinuousNoVel-v0
- CarRacing-v0

How to replicate the results?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Clone the repo for the experiment:

.. code-block:: bash

   git clone https://github.com/DLR-RM/rl-baselines3-zoo
   cd rl-baselines3-zoo
   git checkout feat/recurrent-ppo


Run the benchmark (replace ``$ENV_ID`` by the envs mentioned above):

.. code-block:: bash

  python train.py --algo ppo_lstm --env $ENV_ID --eval-episodes 10 --eval-freq 10000


Parameters
----------

.. autoclass:: RecurrentPPO
  :members:
  :inherited-members:


RecurrentPPO Policies
---------------------

.. autoclass:: MlpLstmPolicy
  :members:
  :inherited-members:

.. autoclass:: sb3_contrib.common.recurrent.policies.RecurrentActorCriticPolicy
  :members:
  :noindex:

.. autoclass:: CnnLstmPolicy
  :members:

.. autoclass:: sb3_contrib.common.recurrent.policies.RecurrentActorCriticCnnPolicy
  :members:
  :noindex:

.. autoclass:: MultiInputLstmPolicy
  :members:

.. autoclass:: sb3_contrib.common.recurrent.policies.RecurrentMultiInputActorCriticPolicy
  :members:
  :noindex:
