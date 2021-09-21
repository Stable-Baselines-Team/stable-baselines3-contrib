.. _ars:

.. automodule:: sb3_contrib.ars


ARS
===


Augmented Random Search (ARS) is a simple reinforcement algorithm that uses a direct random search over policy
parameters. In the `original paper <https://arxiv.org/abs/1803.07055>`_ The authors showed that linear policies trained
with ARS were competititve with deep reinforcement learning for the MuJuCo locomotion tasks.


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
-  Multi processing:  ✔️
-  Gym spaces:


============= ====== ===========
Space         Action Observation
============= ====== ===========
Discrete      ❌      ️ ❌
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

TODO

Comments
--------

TODO

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
