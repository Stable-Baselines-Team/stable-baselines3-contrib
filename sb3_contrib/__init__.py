import os

import gym

from sb3_contrib.ars import ARS
from sb3_contrib.diayn import DIAYN
from sb3_contrib.icm import ICM
from sb3_contrib.ppo_mask import MaskablePPO
from sb3_contrib.qrdqn import QRDQN
from sb3_contrib.rnd import RND
from sb3_contrib.skew_fit import SkewFit
from sb3_contrib.tqc import TQC
from sb3_contrib.trpo import TRPO
from sb3_contrib.vime import VIME

# Read version from file
version_file = os.path.join(os.path.dirname(__file__), "version.txt")
with open(version_file, "r") as file_handler:
    __version__ = file_handler.read().strip()
