.. _reppo:

.. automodule:: stable_baselines3.reppo


REPPO
=====

`Relative Entropy Pathwise Policy Optimization (REPPO) <https://arxiv.org/abs/2507.11019>`_ is an on-policy reinforcement learning algorithm that uses pathwise (reparameterized) policy gradients with an accurate surrogate Q-function learned purely from on-policy data. Unlike score-function methods such as PPO and REINFORCE, pathwise gradients differentiate directly through the value function, yielding lower-variance gradient estimates and more stable training.

Key components of REPPO:

- **Pathwise Policy Gradients**: Uses reparameterized action sampling and differentiates through a learned Q-function, avoiding the high variance of score-function estimators (Eq 5)
- **On-Policy Q-Learning**: Trains a state-action value function (Q) from on-policy trajectories using entropy-regularized multi-step TD-λ targets (Eq 7-8), removing the need for replay buffers
- **KL-Constrained Updates**: Bounds policy deviation from the behavior policy using a clipped loss based on KL divergence, inspired by Relative Entropy Policy Search (Eq 15)
- **Dual Temperature Learning**: Automatically tunes separate Lagrange multipliers for entropy (α) and KL (β) regularization via gradient-based root finding (Eq 13-14)
- **Distributional Critic**: Uses HL-Gauss cross-entropy loss for value regression, improving training stability over MSE (Eq 9)
- **Auxiliary Self-Prediction Loss**: Trains the critic encoder with a latent forward prediction objective to stabilize representations (Eq 17)


.. rubric:: Available Policies

.. autosummary::
    :nosignatures:

    MlpPolicy
    ActorQPolicy


Notes
-----

- Original paper: https://arxiv.org/abs/2507.11019
- REPPO uses HL-Gauss (Histogram Loss with Gaussian smoothing) for distributional value learning: https://arxiv.org/abs/2403.03950

.. note::
    REPPO uses a different policy architecture than PPO. The ``ActorQPolicy`` combines:
    
    - A **stochastic actor** (tanh-squashed Gaussian policy) with state-dependent or fixed standard deviation
    - A **distributional Q-critic** that outputs categorical distributions over value bins via HL-Gauss encoding
    - Separate network branches for actor and critic, each with its own optimizer

.. note::
    The entropy and KL divergence temperature coefficients (``alpha_temp`` and ``alpha_kl``) are automatically 
    learned during training. The algorithm optimizes the logarithm of these temperatures for numerical stability.

.. note::
    REPPO defaults to using SiLU activation functions and LayerNorm in the critic, which differs from 
    PPO's default architecture. The critic has 201 bins by default spanning a value range of [-10, 10].


Can I use?
----------

-  Recurrent policies: ❌
-  Multi processing: ✔️
-  Gym spaces:


============= ====== ===========
Space         Action Observation
============= ====== ===========
Discrete      ❌     ✔️
Box           ✔️     ✔️
MultiDiscrete ❌     ✔️
MultiBinary   ❌     ✔️
Dict          ❌     ✔️
============= ====== ===========


Example
-------

This example is only to demonstrate the use of the library and its functions, and the trained agents may not solve the environments. Optimized hyperparameters can be found in RL Zoo `repository <https://github.com/DLR-RM/rl-baselines3-zoo>`_.

Train a REPPO agent on ``Pendulum-v1`` using 4 environments.

.. code-block:: python

  import gymnasium as gym

  from sb3_contrib.reppo import REPPO
  from stable_baselines3.common.env_util import make_vec_env

  # Parallel environments
  vec_env = make_vec_env("Pendulum-v1", n_envs=4)

  model = REPPO("MlpPolicy", vec_env, verbose=1)
  model.learn(total_timesteps=50000)
  model.save("reppo_pendulum")

  del model # remove to demonstrate saving and loading

  model = REPPO.load("reppo_pendulum")

  obs = vec_env.reset()
  while True:
      action, _states = model.predict(obs)
      obs, rewards, dones, info = vec_env.step(action)
      vec_env.render("human")


.. note::

  REPPO is designed for GPU-parallelized environments (the paper evaluates on 1024 parallel envs
  on GPU). 


Results
-------

.. note::

  Results for REPPO are still being collected. The algorithm is particularly designed for continuous control 
  tasks with Box action spaces where distributional value learning and entropy regularization provide benefits.


How to replicate the results?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Clone the `rl-zoo repo <https://github.com/DLR-RM/rl-baselines3-zoo>`_:

.. code-block:: bash

  git clone https://github.com/DLR-RM/rl-baselines3-zoo
  cd rl-baselines3-zoo/


Run the benchmark (replace ``$ENV_ID`` by continuous control environments):

.. code-block:: bash

  python train.py --algo reppo --env $ENV_ID --eval-episodes 10 --eval-freq 10000


Parameters
----------

.. autoclass:: REPPO
  :members:
  :inherited-members:


.. _reppo_policies:

REPPO Policies
--------------

.. autoclass:: MlpPolicy
  :members:
  :inherited-members:

.. autoclass:: ActorQPolicy
  :members:
  :noindex:

