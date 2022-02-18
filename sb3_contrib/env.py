from typing import Union
import gym
from gym import spaces
import numpy as np
import pygame


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


def to_pix(x, y):
    return x * 20 + 250, -y * 20 + 250


class MyMaze(object):
    """My custom maze."""

    action_space = spaces.Box(-1, 1, (2,))
    observation_space = spaces.Box(-10, 10, (2,))

    walls = [
        [np.array([-3, 3]), np.array([3, 3])],
        [np.array([3, 3]), np.array([3, -3])],
        [np.array([3, -3]), np.array([-3, -3])],
        [np.array([-3, -3]), np.array([-3, 3])],
        [np.array([-3, 1]), np.array([1, 1])],
        [np.array([-1, -1]), np.array([3, -1])],
    ]

    def __init__(self) -> None:
        self.win = None

    def step(self, action):
        new_pos = self.pos + action
        for wall in self.walls:
            intersection = get_intersect(wall[0], wall[1], self.pos, new_pos)
            if intersection is not False:
                print("intersection!", intersection)
                new_pos = 0.9 * intersection + 0.1 * self.pos
        self.pos = new_pos
        return self.pos.copy(), 0.0, False, {}

    def reset(self):
        self.pos = np.zeros(2)

    def render(self, mode="human"):
        if self.win is None:
            pygame.init()
            self.win = pygame.display.set_mode((500, 500))
        black = (0, 0, 0)
        red = (255, 0, 0)
        # self.win.fill(black)
        for wall in self.walls:
            start_pos, end_pos = to_pix(*wall[0]), to_pix(*wall[1])
            pygame.draw.line(self.win, red, start_pos, end_pos)
        x, y = to_pix(*self.pos)
        pygame.draw.circle(self.win, red, (x, y), 2)
        pygame.display.update()
        pygame.time.delay(50)
