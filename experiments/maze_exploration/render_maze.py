from typing import List

import numpy as np
import pygame
from PIL import Image
from pygame import gfxdraw

SCREEN_DIM = 500
BOUND = 13
SCALE = SCREEN_DIM / (BOUND * 2)
OFFSET = SCREEN_DIM // 2

BLACK = pygame.Color(0, 0, 0)
WHITE = pygame.Color(255, 255, 255)
GRAY = pygame.Color(192, 192, 192)
GREEN = pygame.Color(0, 200, 0)
RED = pygame.Color(255, 0, 0)
BLUE = pygame.Color(0, 0, 255)


def draw_filled_circle(surf, pos, radius=1, color=RED):
    x, y = pos * SCALE + OFFSET
    gfxdraw.filled_circle(surf, int(x), int(y), radius, color)


def draw_circle(surf, pos, radius=3, color=GREEN):
    x, y = pos * SCALE + OFFSET
    gfxdraw.circle(surf, int(x), int(y), radius, color)


def draw_line(surf, point_a, point_b, color=BLACK):
    x1, y1 = point_a * SCALE + OFFSET
    x2, y2 = point_b * SCALE + OFFSET
    gfxdraw.line(surf, int(x1), int(y1), int(x2), int(y2), color)


def render_and_save(envs: List, filename: str, goals=None, bg=WHITE, grid=None):
    pygame.init()
    screen = pygame.display.set_mode((SCREEN_DIM, SCREEN_DIM))
    surf = pygame.Surface((SCREEN_DIM, SCREEN_DIM))

    # Background
    surf.fill(bg)

    if grid is not None:
        origin, width = grid
        for i in range(200):
            draw_line(surf, np.array([-13, (i - 100 + origin) * width]), np.array([13, (i - 100 + origin) * width]), GRAY)
            draw_line(surf, np.array([(i - 100 + origin) * width, -13]), np.array([(i - 100 + origin) * width, 13]), GRAY)

    # Draw visited points
    all_pos = np.concatenate([env.all_pos for env in envs])
    for pos in all_pos:
        draw_filled_circle(surf, pos)

    # Draw goals if any
    goals = [] if goals is None else goals
    for pos in goals:
        draw_filled_circle(surf, pos, color=BLUE)

    # Draw walls
    for point_a, point_b in envs[0].walls:
        draw_line(surf, point_a, point_b)

    surf = pygame.transform.flip(surf, flip_x=False, flip_y=True)
    screen.blit(surf, (0, 0))
    im = np.transpose(np.array(pygame.surfarray.pixels3d(screen)), axes=(1, 0, 2))
    im = Image.fromarray(im)
    im.save(filename)
