RL Algorithms
=============

This table displays the rl algorithms that are implemented in the Stable Baselines3 contrib project,
along with some useful characteristics: support for discrete/continuous actions, multiprocessing.


============ =========== ============ ================= =============== ================
Name         ``Box``     ``Discrete`` ``MultiDiscrete`` ``MultiBinary`` Multi Processing
============ =========== ============ ================= =============== ================
TQC          ✔️          ❌            ❌                ❌              ❌
QR-DQN       ️❌          ️✔️            ❌                ❌              ❌
============ =========== ============ ================= =============== ================


.. note::
    Non-array spaces such as ``Dict`` or ``Tuple`` are not currently supported by any algorithm.

Actions ``gym.spaces``:

-  ``Box``: A N-dimensional box that contains every point in the action
   space.
-  ``Discrete``: A list of possible actions, where each timestep only
   one of the actions can be used.
-  ``MultiDiscrete``: A list of possible actions, where each timestep only one action of each discrete set can be used.
- ``MultiBinary``: A list of possible actions, where each timestep any of the actions can be used in any combination.
