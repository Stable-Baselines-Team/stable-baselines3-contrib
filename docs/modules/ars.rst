.. _ars:

.. automodule:: sb3_contrib.ars


ARS
===


Augmented Random Search (ARS) is a simple reinforcement algorithm that uses a direct random search over policy
parameters. It can be surprisingly effective compared to more sophisticated algorithms. In the `original paper <https://arxiv.org/abs/1803.07055>`_ the authors showed that linear policies trained with ARS were competitive with deep reinforcement learning for the MuJuCo locomotion tasks.

SB3s implementation allows for linear policies without bias or squashing function, it also allows for training MLP policies, which include linear policies with bias and squashing functions as a special case.

Normally one wants to train ARS with several seeds to properly evaluate.

.. warning::

  ARS multi-processing is different from the classic Stable-Baselines3 multi-processing: it runs n environments
  in parallel but asynchronously. This asynchronous multi-processing is considered experimental
  and does not fully support callbacks: the ``on_step()`` event is called artificially after the evaluation episodes are over.


.. rubric:: Available Policies

.. autosummary::
    :nosignatures:

    LinearPolicy
    MlpPolicy


Notes
-----

- Original paper: https://arxiv.org/abs/1803.07055
- Original Implementation: https://github.com/modestyachts/ARS


Can I use?
----------

-  Recurrent policies: ❌
-  Multi processing:  ✔️ (cf. example)
-  Gym spaces:


============= ====== ===========
Space         Action Observation
============= ====== ===========
Discrete      ✔️       ✔️
Box           ✔️       ✔️
MultiDiscrete ❌       ✔️
MultiBinary   ❌       ✔️
Dict          ❌       ❌
============= ====== ===========


Example
-------

.. code-block:: python

   from sb3_contrib import ARS

   # Policy can be LinearPolicy or MlpPolicy
   model = ARS("LinearPolicy", "Pendulum-v1", verbose=1)
   model.learn(total_timesteps=10000, log_interval=4)
   model.save("ars_pendulum")


With experimental asynchronous multi-processing:

.. code-block:: python

	from sb3_contrib import ARS
	from sb3_contrib.common.vec_env import AsyncEval

	from stable_baselines3.common.env_util import make_vec_env

	env_id = "CartPole-v1"
	n_envs = 2

	model = ARS("LinearPolicy", env_id, n_delta=2, n_top=1, verbose=1)
	# Create env for asynchronous evaluation (run in different processes)
	async_eval = AsyncEval([lambda: make_vec_env(env_id) for _ in range(n_envs)], model.policy)

	model.learn(total_timesteps=200_000, log_interval=4, async_eval=async_eval)


Results
-------

Replicating results from the original paper, which used the Mujoco benchmarks. Same parameters from the original paper, using 8 seeds.

============= ============
Environments     ARS
============= ============
\
HalfCheetah   4398 +/- 320
Swimmer       241 +/- 51
Hopper        3320 +/- 120
============= ============

How to replicate the results?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Clone RL-Zoo and checkout the branch ``feat/ars``

.. code-block:: bash

  git clone https://github.com/DLR-RM/rl-baselines3-zoo
  cd rl-baselines3-zoo/
  git checkout feat/ars

Run the benchmark. The following code snippet trains 8 seeds in parallel

.. code-block:: bash

   for ENV_ID in Swimmer-v3 HalfCheetah-v3 Hopper-v3
   do
       for SEED_NUM in {1..8}
       do
           SEED=$RANDOM
           python train.py --algo ars --env $ENV_ID --eval-episodes 10 --eval-freq 10000 -n 20000000 --seed  $SEED &
           sleep 1
       done
       wait
   done

Plot the results:

.. code-block:: bash

  python scripts/all_plots.py -a ars -e HalfCheetah Swimmer Hopper -f logs/ -o logs/ars_results -max 20000000
  python scripts/plot_from_file.py -i logs/ars_results.pkl -l ARS



Parameters
----------

.. autoclass:: ARS
  :members:
  :inherited-members:

.. _ars_policies:

ARS Policies
-------------

.. autoclass:: sb3_contrib.ars.policies.ARSPolicy
  :members:
  :noindex:

.. autoclass:: LinearPolicy
  :members:
  :inherited-members:

.. autoclass:: MlpPolicy
  :members:
