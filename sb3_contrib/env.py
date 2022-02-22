from typing import Dict, Tuple, Union

import gym
import numpy as np
import pygame
from gym import spaces

BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
RED = (255, 0, 0)


def get_intersect(A: np.ndarray, B: np.ndarray, C: np.ndarray, D: np.ndarray) -> Union[bool, np.ndarray]:
    """
    Get the intersection of [A, B] and [C, D]. Return False if segment don't cross.

    :param A: Point of the first segment
    :param B: Point of the first segment
    :param C: Point of the second segment
    :param D: Point of the second segment
    :return: The intersection if any
    """
    det = (B[0] - A[0]) * (C[1] - D[1]) - (C[0] - D[0]) * (B[1] - A[1])
    if det == 0:
        # Parallel
        return False
    else:
        t1 = ((C[0] - A[0]) * (C[1] - D[1]) - (C[0] - D[0]) * (C[1] - A[1])) / det
        t2 = ((B[0] - A[0]) * (C[1] - A[1]) - (C[0] - A[0]) * (B[1] - A[1])) / det
        if t1 > 1 or t1 < 0 or t2 > 1 or t2 < 0:
            # not intersect
            return False
        else:
            xi = A[0] + t1 * (B[0] - A[0])
            yi = A[1] + t1 * (B[1] - A[1])
            return np.array([xi, yi])


def to_pix(X: np.ndarray) -> np.ndarray:
    return X * np.array([1, -1]) * 20 + 250


class MyMaze(gym.Env):
    """My custom maze."""

    action_space = spaces.Box(-1, 1, (2,))
    observation_space = spaces.Box(-10, 10, (2,))

    walls = [
        [[-12, -12], [-12, 12]],
        [[-10, 8], [-10, 10]],
        [[-10, 0], [-10, 6]],
        [[-10, -4], [-10, -2]],
        [[-10, -10], [-10, -6]],
        [[-8, 4], [-8, 8]],
        [[-8, -4], [-8, 0]],
        [[-8, -8], [-8, -6]],
        [[-6, 8], [-6, 10]],
        [[-6, 4], [-6, 6]],
        [[-6, 0], [-6, 2]],
        [[-6, -6], [-6, -4]],
        [[-4, 2], [-4, 8]],
        [[-4, -2], [-4, 0]],
        [[-4, -10], [-4, -6]],
        [[-2, 8], [-2, 12]],
        [[-2, 2], [-2, 6]],
        [[-2, -4], [-2, -2]],
        [[0, 6], [0, 12]],
        [[0, 2], [0, 4]],
        [[0, -8], [0, -6]],
        [[2, 8], [2, 10]],
        [[2, -8], [2, 6]],
        [[4, 10], [4, 12]],
        [[4, 4], [4, 6]],
        [[4, 0], [4, 2]],
        [[4, -6], [4, -2]],
        [[4, -10], [4, -8]],
        [[6, 10], [6, 12]],
        [[6, 6], [6, 8]],
        [[6, 0], [6, 2]],
        [[6, -8], [6, -6]],
        [[8, 10], [8, 12]],
        [[8, 4], [8, 6]],
        [[8, -4], [8, 2]],
        [[8, -10], [8, -8]],
        [[10, 10], [10, 12]],
        [[10, 4], [10, 8]],
        [[10, -2], [10, 0]],
        [[12, -12], [12, 12]],
        [[-12, 12], [12, 12]],
        [[-12, 10], [-10, 10]],
        [[-8, 10], [-6, 10]],
        [[-4, 10], [-2, 10]],
        [[2, 10], [4, 10]],
        [[-8, 8], [-2, 8]],
        [[2, 8], [8, 8]],
        [[-10, 6], [-8, 6]],
        [[-6, 6], [-2, 6]],
        [[6, 6], [8, 6]],
        [[0, 4], [6, 4]],
        [[-10, 2], [-6, 2]],
        [[-2, 2], [0, 2]],
        [[8, 2], [10, 2]],
        [[-4, 0], [-2, 0]],
        [[2, 0], [4, 0]],
        [[6, 0], [8, 0]],
        [[-6, -2], [2, -2]],
        [[4, -2], [10, -2]],
        [[-12, -4], [-8, -4]],
        [[-4, -4], [-2, -4]],
        [[0, -4], [6, -4]],
        [[8, -4], [10, -4]],
        [[-8, -6], [-6, -6]],
        [[-2, -6], [0, -6]],
        [[6, -6], [10, -6]],
        [[-12, -8], [-6, -8]],
        [[-2, -8], [2, -8]],
        [[4, -8], [6, -8]],
        [[8, -8], [10, -8]],
        [[-10, -10], [-8, -10]],
        [[-4, -10], [4, -10]],
        [[-12, -12], [12, -12]],
    ]

    def __init__(self) -> None:
        self.win = None

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict]:
        new_pos = self.pos + action
        for wall in self.walls:
            intersection = get_intersect(wall[0], wall[1], self.pos, new_pos)
            if intersection is not False:
                new_pos = self.pos
        self.pos = new_pos
        self.render()
        return self.pos.copy(), 0.0, False, {}

    def reset(self) -> np.ndarray:
        self.pos = np.zeros(2)
        return self.pos.copy()

    def render(self, mode="human") -> None:
        if self.win is None:
            pygame.init()
            self.win = pygame.display.set_mode((500, 500))
            self.win.fill(BLACK)
        center = to_pix(self.pos)
        pygame.draw.circle(self.win, RED, center, radius=1)
        for wall in self.walls:
            start_pos, end_pos = to_pix(wall[0]), to_pix(wall[1])
            pygame.draw.line(self.win, WHITE, start_pos, end_pos, width=3)
        pygame.display.update()
