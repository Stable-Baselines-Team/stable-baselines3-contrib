import os

from sb3_contrib.ppo_mask import MaskablePPO
from sb3_contrib.qrdqn import QRDQN
from sb3_contrib.tqc import TQC

# Read version from file
version_file = os.path.join(os.path.dirname(__file__), "version.txt")
with open(version_file, "r") as file_handler:
    __version__ = file_handler.read().strip()
