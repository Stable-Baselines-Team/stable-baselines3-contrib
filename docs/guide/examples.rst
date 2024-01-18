.. _examples:

Examples
========

TQC
---

Train a Truncated Quantile Critics (TQC) agent on the Pendulum environment.

.. code-block:: python

  from sb3_contrib import TQC

  model = TQC("MlpPolicy", "Pendulum-v1", top_quantiles_to_drop_per_net=2, verbose=1)
  model.learn(total_timesteps=10_000, log_interval=4)
  model.save("tqc_pendulum")

QR-DQN
------

Train a Quantile Regression DQN (QR-DQN) agent on the CartPole environment.

.. code-block:: python

  from sb3_contrib import QRDQN

  policy_kwargs = dict(n_quantiles=50)
  model = QRDQN("MlpPolicy", "CartPole-v1", policy_kwargs=policy_kwargs, verbose=1)
  model.learn(total_timesteps=10_000, log_interval=4)
  model.save("qrdqn_cartpole")

MaskablePPO
-----------

Train a PPO with invalid action masking agent on a toy environment.

.. warning::
  You must use ``MaskableEvalCallback`` from ``sb3_contrib.common.maskable.callbacks`` instead of the base ``EvalCallback`` to properly evaluate a model with action masks.
  Similarly, you must use ``evaluate_policy`` from ``sb3_contrib.common.maskable.evaluation`` instead of the SB3 one.



.. code-block:: python

  from sb3_contrib import MaskablePPO
  from sb3_contrib.common.envs import InvalidActionEnvDiscrete

  env = InvalidActionEnvDiscrete(dim=80, n_invalid_actions=60)
  model = MaskablePPO("MlpPolicy", env, verbose=1)
  model.learn(5000)
  model.save("maskable_toy_env")

TRPO
----

Train a Trust Region Policy Optimization (TRPO) agent on the Pendulum environment.

.. code-block:: python

  from sb3_contrib import TRPO

  model = TRPO("MlpPolicy", "Pendulum-v1", gamma=0.9, verbose=1)
  model.learn(total_timesteps=100_000, log_interval=4)
  model.save("trpo_pendulum")


ARS
---

Train an agent using Augmented Random Search (ARS) agent on the Pendulum environment

.. code-block:: python

   from sb3_contrib import ARS

   model = ARS("LinearPolicy", "Pendulum-v1", verbose=1)
   model.learn(total_timesteps=10000, log_interval=4)
   model.save("ars_pendulum")

RecurrentPPO
------------

Train a PPO agent with a recurrent policy on the CartPole environment.


.. note::

  It is particularly important to pass the ``lstm_states``
  and ``episode_start`` argument to the ``predict()`` method,
  so the cell and hidden states of the LSTM are correctly updated.


.. code-block:: python

 import numpy as np

 from sb3_contrib import RecurrentPPO

 model = RecurrentPPO("MlpLstmPolicy", "CartPole-v1", verbose=1)
 model.learn(5000)

 vec_env = model.get_env()
 obs = vec_env.reset()
 # Cell and hidden state of the LSTM
 lstm_states = None
 num_envs = 1
 # Episode start signals are used to reset the lstm states
 episode_starts = np.ones((num_envs,), dtype=bool)
 while True:
     action, lstm_states = model.predict(obs, state=lstm_states, episode_start=episode_starts, deterministic=True)
     # Note: vectorized environment resets automatically
     obs, rewards, dones, info = vec_env.step(action)
     episode_starts = dones
     vec_env.render("human")
