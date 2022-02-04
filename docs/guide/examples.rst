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
