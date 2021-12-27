.. _ppo_mask:

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

.. - Paper: https://arxiv.org/abs/2006.14171
.. - Blog post: https://costa.sh/blog-a-closer-look-at-invalid-action-masking-in-policy-gradient-algorithms.html


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


.. code-block:: python

  import numpy as np

  from sb3_contrib import RecurrentPPO
  from stable_baselines3.common.evaluation import evaluate_policy

  model = RecurrentPPO("MlpLstmPolicy", "CartPole-v1", verbose=1)
  model.learn(5000)

  mean_reward, std_reward = evaluate_policy(model, model.get_env(), n_eval_episodes=20, warn=False)
  print(mean_reward)

  model.save("ppo_recurrent")
  del model # remove to demonstrate saving and loading

  model = RecurrentPPO.load("ppo_recurrent")

  env = model.get_env()
  obs = env.reset()
  states = None
  num_envs = 1
  episode_starts = np.ones((num_envs,), dtype=bool)
  while True:
      action, states = model.predict(obs, state=states, episode_start=episode_starts, deterministic=True)
      obs, rewards, dones, info = env.step(action)
      episode_starts = dones
      env.render()



Results
-------

How to replicate the results?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Clone the repo for the experiment:

.. code-block:: bash

   git clone https://github.com/DLR-RM/rl-baselines3-zoo
   git checkout feat/recurrent-ppo

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
