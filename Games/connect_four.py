#!/usr/bin/python3

import numpy as np
import pygame
import sys
import math
from enums.colors import Colors
from enums.player import PLAYER


class ConnectFour:
    def __init__(self, number_of_rows=6, number_of_cols=7):
        self.NUM_ROWS = number_of_rows
        self.NUM_COLS = number_of_cols
        self.SQUARE_SIZE = 100
        self.OFFSET = 100
        self.WIDTH = self.NUM_COLS * self.SQUARE_SIZE
        self.HEIGHT = (self.NUM_ROWS+1) * self.SQUARE_SIZE
        self.RADIUS = int(self.SQUARE_SIZE/2 - 5)
        self.screen = self.set_screen()
        self.game_font = pygame.font.SysFont("monospace", 75)

    def create_game_board(self):
        """
        Returns game board as np array

        Returns:
        - board (np.ndarray): Numpy array represents game board with size
            NUM_ROWS X NUM_COLS
        """
        return np.zeros((self.NUM_ROWS, self.NUM_COLS))

    def set_screen(self):
        """
        Set screen with size WIDTH x HEIGHT
        """
        return pygame.display.set_mode((self.WIDTH, self.HEIGHT))

    def update_pygame(self):
        """
        Update game display after some changes on pygame object
        """
        pygame.display.update()

    def draw_board(self, board):
        """
        Draw board from 2D numpy array
        """
        for c in range(self.NUM_COLS):
            for r in range(self.NUM_ROWS):
                # Draw main board
                pygame.draw.rect(self.screen,
                                 Colors.BLUE.value,
                                 (
                                     c*self.SQUARE_SIZE,
                                     r*self.SQUARE_SIZE+self.OFFSET,
                                     self.SQUARE_SIZE,
                                     self.SQUARE_SIZE
                                 ))

                # Draw stones
                player_color = Colors.BLACK.value

                if board[r][c] == PLAYER.FIRST.value:
                    player_color = Colors.RED.value

                elif board[r][c] == PLAYER.SECOND.value:
                    player_color = Colors.YELLOW.value

                pygame.draw.circle(self.screen,
                                   player_color,
                                   (
                                       int(c*self.SQUARE_SIZE +
                                           self.SQUARE_SIZE/2),
                                       int(r*self.SQUARE_SIZE +
                                           self.SQUARE_SIZE+self.SQUARE_SIZE/2)
                                   ),
                                   self.RADIUS)

        self.update_pygame()

    def move(self, player, positions):
        # Reference position
        pos_x, pos_y = positions
        player = PLAYER.NONE
        color = Colors.BLACK.value
        if player is PLAYER.FIRST:
            color = Colors.RED.value
        elif player is PLAYER.SECOND:
            color = Colors.YELLOW.value
        pygame.draw.circle(self.screen, color,
                           (pos_x, pos_y), self.RADIUS)

    def move_mouse(self, posx, player_turn):
        pygame.draw.rect(self.screen, Colors.BLACK.value,
                         (0, 0, self.WIDTH, self.SQUARE_SIZE))
        color = Colors.BLACK.value
        if player_turn is PLAYER.FIRST:
            color = Colors.RED.value
        elif player_turn is PLAYER.SECOND:
            color = Colors.YELLOW.value
        pygame.draw.circle(self.screen, color,
                           (posx, int(self.SQUARE_SIZE/2)), self.RADIUS)

    def place_stone(self, board, posx, player_turn):
        pygame.draw.rect(self.screen, Colors.BLACK.value,
                         (0, 0, self.WIDTH, self.SQUARE_SIZE))
        player_value = player_turn.value
        col = int(math.floor(posx / self.SQUARE_SIZE))
        row, is_row_found = self.get_row(board, col)
        if is_row_found :
            board = self.set_stone_board(board, [row, col], player_value)
        else:
            print("No row is left in that column !")
        return board, is_row_found

    def flip_players(self, turn):
        if turn is PLAYER.FIRST:
            print("Next turn is for : ", PLAYER.SECOND)
            return PLAYER.SECOND
        elif turn is PLAYER.SECOND:
            print("Next turn is for : ", PLAYER.FIRST)
            return PLAYER.FIRST

    def get_row(self, board, col):
        is_row_found = False
        row_selected = None
        for r in reversed(range(self.NUM_ROWS)):
            if board[r][col] == 0:
                row_selected = r
                is_row_found = True
                break

        return row_selected, is_row_found

    def set_stone_board(self, board, position, piece):
        """
        Set position of a stone on the board
        """
        row, col = position
        board[row][col] = piece
        return board

    def print_board(self, board):
        print(np.flip(board, 0))

    def run(self):
        game_over = False
        turn = PLAYER.FIRST
        board = self.create_game_board()
        self.update_pygame()

        while not game_over:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    sys.exit()
                else:
                    if event.type == pygame.MOUSEMOTION:
                        self.move_mouse(event.pos[0], turn)

                    elif event.type == pygame.MOUSEBUTTONDOWN:
                        board, is_row_found = self.place_stone(board, event.pos[0], turn)
                        # Only flip players when there is left any place to move ! 
                        if is_row_found:
                            turn = self.flip_players(turn)

                self.update_pygame()

                self.draw_board(board)


if __name__ == "__main__":
    pygame.init()
    connect_two_game = ConnectFour()
    connect_two_game.run()
