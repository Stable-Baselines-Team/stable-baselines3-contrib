import gym
import gym_continuous_maze
import numpy as np
import pygame
from PIL import Image
from pygame import gfxdraw

from sb3_contrib import DIAYN

SCREEN_DIM = 500
BOUND = 13
SCALE = SCREEN_DIM / (BOUND * 2)
OFFSET = SCREEN_DIM // 2

BLACK = pygame.Color(0, 0, 0)
WHITE = pygame.Color(255, 255, 255)
GREEN = pygame.Color(0, 200, 0)
RED = pygame.Color(255, 0, 0)
BLUE = pygame.Color(0, 0, 255)
C = [
    (0, 0, 0),
    (24, 13, 47),
    (53, 54, 88),
    (104, 107, 114),
    (139, 151, 182),
    (197, 205, 219),
    (100, 100, 100),
    (94, 233, 233),
    (40, 144, 220),
    (24, 49, 167),
    (5, 50, 57),
    (0, 95, 65),
    (8, 178, 59),
    (71, 246, 65),
    (232, 255, 117),
    (251, 190, 130),
    (222, 151, 81),
    (182, 104, 49),
    (138, 73, 38),
    (70, 28, 20),
    (30, 9, 13),
    (114, 13, 13),
    (129, 55, 4),
    (218, 36, 36),
    (239, 110, 16),
    (236, 171, 17),
    (236, 233, 16),
    (247, 141, 141),
    (249, 78, 109),
    (193, 36, 88),
    (132, 18, 82),
    (61, 8, 59),
]
COLORS = [pygame.Color(*c) for c in C]


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


def render_and_save(env, buffer, filename: str, goals=None, bg=WHITE):
    pygame.init()
    screen = pygame.display.set_mode((SCREEN_DIM, SCREEN_DIM))
    surf = pygame.Surface((SCREEN_DIM, SCREEN_DIM))

    # Background
    surf.fill(bg)

    # Draw visited points

    poss = buffer.next_observations["observation"][: buffer.pos]
    skills = buffer.next_observations["skill"][: buffer.pos]
    for pos, skill in zip(poss, skills):
        draw_filled_circle(surf, pos[0], color=COLORS[skill.argmax()])

    # Draw goals if any
    goals = [] if goals is None else goals
    for pos in goals:
        draw_filled_circle(surf, pos, color=BLUE)

    # Draw walls
    for point_a, point_b in env.walls:
        draw_line(surf, point_a, point_b)

    surf = pygame.transform.flip(surf, flip_x=False, flip_y=True)
    screen.blit(surf, (0, 0))
    im = np.transpose(np.array(pygame.surfarray.pixels3d(screen)), axes=(1, 0, 2))
    im = Image.fromarray(im)
    im.save(filename)


env = gym.make("ContinuousMaze-v0")
model = DIAYN(
    env,
    nb_skills=32,
    policy_kwargs=dict(net_arch=[256, 256, 256]),
    learning_starts=5000,
    verbose=1,
)
model.learn(500000)
render_and_save(env, model.replay_buffer, "diayn_500k.png")
