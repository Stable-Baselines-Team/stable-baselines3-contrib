RL Algorithms
=============

This table displays the rl algorithms that are implemented in the Stable Baselines3 contrib project,
along with some useful characteristics: support for discrete/continuous actions, multiprocessing.


============ =========== ============ ================= =============== ================
Name         ``Box``     ``Discrete`` ``MultiDiscrete`` ``MultiBinary`` Multi Processing
============ =========== ============ ================= =============== ================
ARS          ✔️          ❌️            ❌                ❌                ✔️
MaskablePPO  ❌           ✔️             ✔️                ✔️               ✔️
QR-DQN       ️❌          ️✔️            ❌                ❌                ✔️
RecurrentPPO ✔️           ✔️             ✔️                ✔️               ✔️
TQC          ✔️          ❌            ❌                ❌                ✔️
TRPO         ✔️          ✔️             ✔️                ✔️                ✔️
============ =========== ============ ================= =============== ================


.. note::
  ``Tuple`` observation spaces are not supported by any environment,
  however, single-level ``Dict`` spaces are supported.

Actions ``gym.spaces``:

-  ``Box``: A N-dimensional box that contains every point in the action
   space.
-  ``Discrete``: A list of possible actions, where each timestep only
   one of the actions can be used.
-  ``MultiDiscrete``: A list of possible actions, where each timestep only one action of each discrete set can be used.
- ``MultiBinary``: A list of possible actions, where each timestep any of the actions can be used in any combination.
