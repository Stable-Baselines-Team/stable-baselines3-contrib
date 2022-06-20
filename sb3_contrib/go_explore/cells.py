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


class Downscale:
    """
    Downscale.

    Rounding, but extended to every float.

    :param decimals: Decimals, can be float

    Example:
    >>> a
    array([0.001, 0.01, 0.1, 1.0, 10.0, 100.0])
    >>> cell_factory = Downscale(decimals=0)
    >>> cell_factory(a)
    array([  0.,   0.,   0.,   1.,  10., 100.])
    >>> cell_factory = Downscale(decimals=1)
    >>> cell_factory(a)
    array([  0. ,   0. ,   0.1,   1. ,  10. , 100. ])
    >>> cell_factory = Downscale(decimals=-1)
    >>> cell_factory(a)
    array([  0.,   0.,   0.,   0.,  10., 100.])
    >>> cell_factory = Downscale(decimals=0.3)
    >>> cell_factory(a)
    array([  0.        ,   0.        ,   0.        ,   1.00237447,
            10.02374467, 100.23744673])
    """

    def __init__(self, decimals) -> None:
        self.decimals = decimals

    def __call__(self, observation: np.ndarray) -> np.ndarray:
        """
        Compute the cells.

        :param observations: Observations
        :return: An array of cells
        """
        cell = np.round(observation * 10**self.decimals) / 10**self.decimals
        cell = cell + np.zeros_like(cell)  # Avoid -0.0
        return cell
