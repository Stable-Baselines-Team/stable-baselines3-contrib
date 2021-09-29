.. _ars:

.. automodule:: sb3_contrib.ars


ARS
===


Augmented Random Search (ARS) is a simple reinforcement algorithm that uses a direct random search over policy
parameters. In the `original paper <https://arxiv.org/abs/1803.07055>`_ The authors showed that linear policies trained
with ARS were competititve with deep reinforcement learning for the MuJuCo locomotion tasks.

SB3s implementation also allows for training MlP policies, which include linear policies with bias and squashing functions as a special case.


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
-  Multi processing:  ❌
-  Gym spaces:


============= ====== ===========
Space         Action Observation
============= ====== ===========
Discrete      ✔️       ✔️
Box           ✔️       ✔️
MultiDiscrete ❌       ❌
MultiBinary   ❌       ❌
Dict          ❌       ❌
============= ====== ===========


Example
-------

.. code-block:: python

   from sb3_contrib import ARS

   # Policy can be LinearPolicy or MlpPolicy
   model = ARS("LinearPolicy", "Pendulum-v0", verbose=1)
   model.learn(total_timesteps=10000, log_interval=4)
   model.save("ars_pendulum")


Results
-------

Replicating results from the original paper, which used the Mujoco benchmarks. Same parameters from the original paper, using 8 seeds.

|Environments|    ARS     |
|------------|------------|
|            |logs        |
|HalfCheetah |4439 +/- 233|
|Swimmer     |242 +/- 50  |
|Hopper      |3520 +/- 0  |


How to replicate the results?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
clone RL-Zoo and checkout the branch ``feat/ars``

.. code-block:: bash

  git clone https://github.com/DLR-RM/rl-baselines3-zoo
  cd rl-baselines3-zoo/
  git checkout feat/ars

Run the benchmark (replace ``$ENV_ID`` by the envs mentioned above):

.. code-block:: bash


for ENV_ID in Swimmer-v2 HalfCheetah-v2 Hopper-v2
do
    for SEED_NUM in {1..8}
    do
        SEED=$RANDOM
        python train.py --algo ars --env $ENV_ID --eval-episodes 10 --eval-freq 10000 --seed  $SEED &
        sleep 1
    done
done

wait


Plot the results:

.. code-block:: bash

  python scripts/all_plots.py -a ars -e HalfCheetah Swimmer Hopper -f logs/ -o logs/ars_results -max 20000000 
  python scripts/plot_from_file.py -i logs/ars_results.pkl -l ARS


Comments
--------


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
