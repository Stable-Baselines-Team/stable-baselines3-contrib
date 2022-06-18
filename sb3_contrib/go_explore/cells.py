from typing import Callable

import numpy as np

CellFactory = Callable[[np.ndarray], np.ndarray]


def cell_is_obs(observations: np.ndarray) -> np.ndarray:
    """
    Compute the cells.

    :param observations: Observations
    :return: An array of cells
    """
    return observations.copy()
