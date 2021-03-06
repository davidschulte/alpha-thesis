import pygame
import numpy as np
from chinese_checkers.TinyChineseCheckersGame import ChineseCheckersGame as Game
MOVES = [[0, 1], [-1, 1], [1, -1], [1, 0]]
DIM = 9
Y_OFFSET = 17
X_OFFSET = 17
Y_STEP = 47.2 * 1.5
X_STEP = 54.5 * 1.5
R = 12
RED = (255,0,0)
GREEN = (0, 200, 0 )
LIGHT_GREEN = (0, 100, 0)
BLUE = (0,0,255,255)
YELLOW = (255,255,0)
PINK = (255,105,180)
BLACK = (0,0,0,0)
WHITE = (255, 255, 255)


ENDPOINTS_RED = [[0, 6], [2, 4], [2, 6]]
STARTPOINTS_RED = [[8, 2], [6, 2], [6, 4]]
ENDPOINTS_YELLOW = [[6, 4], [6, 6], [4, 6]]
STARTPOINTS_YELLOW = [[2, 2], [2, 4], [4, 2]]
ENDPOINTS_GREEN = [[6, 0], [6, 2], [4, 2]]
STARTPOINTS_GREEN = [[2, 8], [2, 6], [4, 6]]

AREAS = [ENDPOINTS_RED, STARTPOINTS_RED, ENDPOINTS_YELLOW, STARTPOINTS_YELLOW, ENDPOINTS_GREEN, STARTPOINTS_GREEN]

line_width = 3

# 0  1  2  3  4  5  6  7  8  9 10 11 12
START = np.array([[4, 4, 4, 4, 4, 4, 0, 4, 4],  # 0
                  [4, 4, 4, 4, 4, 0, 0, 4, 4],  # 1
                  [4, 4, 2, 2, 2, 0, 3, 3, 3],  # 2
                  [4, 4, 2, 2, 0, 0, 3, 3, 4],  # 3
                  [4, 4, 2, 0, 0, 0, 3, 4, 4],  # 4
                  [4, 0, 0, 0, 0, 0, 0, 4, 4],  # 5
                  [0, 0, 1, 1, 1, 0, 0, 4, 4],  # 6
                  [4, 4, 1, 1, 4, 4, 4, 4, 4],  # 7
                  [4, 4, 1, 4, 4, 4, 4, 4, 4]]).astype('int8')  # 8


# one = myfont.render('1', False, (0, 0, 0))
# two = myfont.render('2', False, (0, 0, 0))
# three = myfont.render('3', False, (0, 0, 0))
# four = myfont.render('4', False, (0, 0, 0))
class GUI:

    def __init__(self, timer):
        pygame.init()

        self.window = pygame.display.set_mode((524, 600))
        self.window.fill(WHITE)
        self.timer = timer
        self.draw_areas(self.window)
        self.draw_lines(self.window)
        self.old_board = START
        self.game = Game()
        self.logic = self.game.get_board()

        pygame.font.init()
        self.myfont = pygame.font.SysFont('Arial', 15)

    def draw_figure(self, surface, row, column, radius, color):
        """
        draws piece on board
        :param surface: surface to draw on
        :param row:     row of piece
        :param column:  columns of piece
        :param radius:  radius of circle in drawing
        :param color:   color of piece
        """
        y = Y_OFFSET + row * Y_STEP
        x = X_OFFSET + column * X_STEP + (row - 6) * X_STEP / 2
        pygame.draw.circle(surface, color, (int(x),int(y)), radius)

    def coordinates_to_pos(self, row, column):
        """
        converts rows and columns to coordinates in the frame
        """
        y = Y_OFFSET + row * Y_STEP
        x = X_OFFSET + column * X_STEP + (row - 6) * X_STEP / 2
        return y, x

    def pos_to_board_coordinates(self, pos):
        """
        converts coordinates to row and column
        if the coordinates do not belong to the board, -1,-1 is returned
        """
        (pos_x, pos_y) = pos
        for row in range(DIM):
            for column in range(DIM):
                y, x = self.coordinates_to_pos(row, column)
                if (pos_y - y)**2 + (pos_x - x) ** 2 < R ** 2:
                    return row, column

        return -1, -1

    def draw_line(self, surface, y1, x1, y2, x2, color):
        pygame.draw.line(surface,color, (x1, y1), (x2, y2), line_width)

    def draw_lines(self, surface):
        """
        draws connection lines between fields
        """

        for row in range(DIM):
            for column in range(DIM):
                for m in range(4):
                    n_y, n_x = row + MOVES[m][0], column + MOVES[m][1]
                    if n_y < DIM and n_x < DIM and n_y > -1 and n_x > -1:
                        if START[row,column] != 4 and START[n_y,n_x] != 4:
                            y1_pos, x1_pos = self.coordinates_to_pos(row, column)
                            y2_pos, x2_pos = self.coordinates_to_pos(n_y, n_x)
                            self.draw_line(surface, y1_pos, x1_pos, y2_pos, x2_pos, BLACK)

    def draw_areas(self, surface):
        """
        colors the areas of the board
        """
        for area in range(6):
            color = GREEN
            if area < 4:
                color = YELLOW
            if area < 2:
                color = RED

            a = AREAS[area]
            coordinates = [0] * 3
            for p in range(3):
                row, column = a[p]
                y_c, x_c = self.coordinates_to_pos(row, column)
                coordinates[p] = (x_c, y_c)

            pygame.draw.polygon(surface, color, coordinates)

    def draw_board(self, board, current_player, wait):
        """
        draws board
        :param board:           current board
        :param current_player:  current player
        :param wait:            boolean that denotes if there should be a pause between displays
        """
        for event in pygame.event.get():
            pass

        for y in range(DIM):
            for x in range(DIM):
                if board[y, x] in [0, 1, 2, 3]:
                    if board[y, x] == 0:
                        color = WHITE
                    elif board[y, x] == 1:
                        color = RED
                    elif board[y, x] == 2:
                        color = YELLOW
                    elif board[y, x] == 3:
                        color = GREEN

                    if self.old_board[y,x] == board[y,x]:
                        self.draw_figure(self.window, y, x, R, BLACK)
                    else:
                        self.draw_figure(self.window, y, x, R, PINK)
                    self.draw_figure(self.window, y, x, R-2, color)
                    # draw_figure(window, y, x, 2, BLACK)

        step_display = self.myfont.render("Player " + str(current_player), False, (0, 0, 0))
        pygame.draw.rect(self.window, WHITE, [10, 10, 50, 30])
        if current_player >= 0:
            self.window.blit(step_display, (10,10))
        pygame.display.update()

        self.old_board = board

        if wait:
            pygame.time.wait(self.timer)
            # timer -= 1

    def draw_possibles(self, possible_board):
        """
        draws possible fields to move, when a piece was selected
        """
        for y in range(DIM):
            for x in range(DIM):
                if possible_board[y, x] in [1, 2, 3]:
                    self.draw_figure(self.window, y, x, R, BLACK)
                    self.draw_figure(self.window, y, x, R-2, PINK)
        pygame.display.update()

    def get_action(self, board, player):
        """
        :param board: board
        :param player: current player
        :return: returns action, after a human player chose it
        """
        selected = False
        possible_board = None
        start_y, start_x, end_y, end_x = -1, -1, -1, -1
        while True:

            for event in pygame.event.get():
                if event.type == pygame.MOUSEBUTTONUP:
                    pos = pygame.mouse.get_pos()
                    y, x = self.pos_to_board_coordinates(pos)
                    if y != -1:
                        if selected:
                            if possible_board[y, x] in [1, 2, 3]:
                                end_y, end_x = y, x
                                action = self.logic.get_action_by_coordinates(start_y, start_x, end_y, end_x, player)
                                return action
                            else:
                                selected = False
                                self.draw_board(board, player, False)

                        else:
                            if board[y, x] == player:
                                start_y, start_x = y, x
                                possible_board = self.game.get_possible_board(y, x, board, player)
                                self.draw_possibles(possible_board)
                                selected = True

    def snapshot(self, board, filename):
        """
        saves a screenshot of the game
        """
        self.old_board = board
        self.draw_board(board, -1, False)
        pygame.image.save(self.window, filename)
