# -*- coding: utf-8 -*-

# suppress welcome message from pygame
import os
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "true"

import pygame
from pygame.locals import *

from go import BLACK, WHITE


class GUI:
    def __init__(self, conf):
        self.board_size = conf.BOARD_SIZE
        self.pass_action = conf.PASS
        self.line_space = 40
        self.stone_radius = 19
        self.star_radius = 4
        self.delta = 15
        self.font_size = 24

        self.go = None
        self.text = ''

        pygame.init()
        width = (self.board_size + 1) * self.line_space
        height = (self.board_size + 1) * self.line_space + 3 * self.font_size
        self.screen = pygame.display.set_mode((width, height), 0, 32)
        self.refresh()

    def draw_board(self):
        # fill background with wood color
        self.screen.fill((220, 179, 92))
        # draw lines
        for i in range(self.board_size):
            pygame.draw.line(
                self.screen,
                (0, 0, 0),
                ((i + 1) * self.line_space, self.line_space),
                ((i + 1) * self.line_space, self.board_size * self.line_space),
            )
            pygame.draw.line(
                self.screen,
                (0, 0, 0),
                (self.line_space, (i + 1) * self.line_space),
                (self.board_size * self.line_space, (i + 1) * self.line_space),
            )
        pygame.draw.line(
            self.screen,
            (0, 0, 0),
            (0, (self.board_size + 1) * self.line_space),
            (
                (self.board_size + 1) * self.line_space,
                (self.board_size + 1) * self.line_space,
            ),
        )
        # draw star marks for 19x19 board
        if self.board_size == 19:
            for x in [4, 10, 16]:
                for y in [4, 10, 16]:
                    pygame.draw.circle(
                        self.screen,
                        (0, 0, 0),
                        (x * self.line_space, y * self.line_space),
                        self.star_radius,
                    )

    def draw_stones(self):
        if self.go is None:
            return
        for x in range(self.board_size):
            for y in range(self.board_size):
                if self.go.board.color(x, y) == BLACK:
                    pygame.draw.circle(
                        self.screen,
                        (0, 0, 0),
                        ((x + 1) * self.line_space, (y + 1) * self.line_space),
                        self.stone_radius,
                    )
                elif self.go.board.color(x, y) == WHITE:
                    pygame.draw.circle(
                        self.screen,
                        (255, 255, 255),
                        ((x + 1) * self.line_space, (y + 1) * self.line_space),
                        self.stone_radius,
                    )
        # mark the most recent move
        if self.go.board.on_board(self.go.previous_x, self.go.previous_y):
            pygame.draw.rect(
                self.screen,
                (0, 0, 0) if self.go.turn == BLACK else (255, 255, 255),
                (
                    (self.go.previous_x + 1) * self.line_space
                    - self.stone_radius // 2,
                    (self.go.previous_y + 1) * self.line_space
                    - self.stone_radius // 2,
                    self.stone_radius,
                    self.stone_radius,
                ),
                1,
            )

    def draw_text(self):
        font = pygame.font.Font('freesansbold.ttf', self.font_size)
        text_surface = font.render(self.text, True, (0, 0, 0))
        text_rect = text_surface.get_rect()
        text_rect.topleft = (
            self.line_space,
            (self.board_size + 1) * self.line_space + self.font_size,
        )
        self.screen.blit(text_surface, text_rect)

    def refresh(self):
        self.draw_board()
        self.draw_stones()
        self.draw_text()
        pygame.display.update()

    def update_go(self, go):
        self.go = go
        self.refresh()

    def update_text(self, text):
        self.text = text
        self.refresh()

    def wait_for_action(self, go):
        while True:
            event = pygame.event.wait()
            if event.type == QUIT:
                exit(0)
            if event.type == MOUSEBUTTONDOWN:
                pressed_mouse_buttons = pygame.mouse.get_pressed()
                # if left mouse button is pressed
                if pressed_mouse_buttons[0] == 1:
                    mouse_x, mouse_y = pygame.mouse.get_pos()
                    x = mouse_x // self.line_space
                    y = mouse_y // self.line_space
                    intersections = [
                        (x - 1, y - 1),
                        (x, y - 1),
                        (x - 1, y),
                        (x, y),
                    ]
                    for x, y in intersections:
                        x_ = (x + 1) * self.line_space
                        y_ = (y + 1) * self.line_space
                        if abs(mouse_x - x_) <= self.delta \
                                and abs(mouse_y - y_) <= self.delta \
                                and go.legal_play(x, y):
                            return x * self.board_size + y
            if event.type == KEYDOWN:
                pressed_keys = pygame.key.get_pressed()
                if pressed_keys[K_ESCAPE]:
                    # return PASS if ESC is pressed
                    return self.pass_action

    def freeze(self):
        while True:
            event = pygame.event.wait()
            if event.type == QUIT:
                exit(0)
