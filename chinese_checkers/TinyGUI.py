import pygame
import numpy as np
from chinese_checkers.TinyChineseCheckersLogic import Board as logic
MOVES = [[0, 1], [-1, 1], [1, -1], [1, 0]]
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
        self.logic = logic()

        pygame.font.init()
        self.myfont = pygame.font.SysFont('Arial', 15)

    def draw_figure(self, surface, row, column, radius, color):
        y = Y_OFFSET + row * Y_STEP
        x = X_OFFSET + column * X_STEP + (row - 6) * X_STEP / 2
        pygame.draw.circle(surface, color, (int(x),int(y)), radius)

    def coordinates_to_pos(self, row, column):
        y = Y_OFFSET + row * Y_STEP
        x = X_OFFSET + column * X_STEP + (row - 6) * X_STEP / 2
        return y, x

    def pos_to_board_coordinates(self, pos):
        (pos_x, pos_y) = pos
        for row in range(9):
            for column in range(9):
                y, x = self.coordinates_to_pos(row, column)
                if (pos_y - y)**2 + (pos_x - x) ** 2 < R ** 2:
                    return row, column

        return -1, -1

    def draw_line(self, surface, y1, x1, y2, x2, color):
        pygame.draw.line(surface,color, (x1, y1), (x2, y2), line_width)

    def draw_lines(self, surface):
        for row in range(9):
            for column in range(9):
                for m in range(4):
                    n_y, n_x = row + MOVES[m][0], column + MOVES[m][1]
                    if n_y < 9 and n_x < 9 and n_y > -1 and n_x > -1:
                        if START[row,column] != 4 and START[n_y,n_x] != 4:
                            y1_pos, x1_pos = self.coordinates_to_pos(row, column)
                            y2_pos, x2_pos = self.coordinates_to_pos(n_y, n_x)
                            self.draw_line(surface, y1_pos, x1_pos, y2_pos, x2_pos, BLACK)

    def draw_areas(self, surface):
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



    def draw_board(self, board, step):
        # timer = self.timer
        # while timer > 0:
        for event in pygame.event.get():
            pass
        # window.blit(bg, (40, 0))

        for y in range(9):
            for x in range(9):
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


        step_display = self.myfont.render("Step " + str(step), False, (0, 0, 0))
        pygame.draw.rect(self.window, WHITE, [10, 10, 50, 30])
        self.window.blit(step_display, (10,10))
        pygame.display.update()

        self.old_board = board

        pygame.time.wait(100)
            # timer -= 1

    def draw_possibles(self, possible_board):
        for y in range(9):
            for x in range(9):
                if possible_board[y, x] == 1:
                    self.draw_figure(self.window, y, x, R, BLACK)
                    self.draw_figure(self.window, y, x, R-2, PINK)
        pygame.display.update()



    def get_action(self, board):
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
                            if possible_board[y, x] == 1:
                                end_y, end_x = y, x
                                action = self.logic.get_action_by_coordinates(start_y, start_x, end_y, end_x)
                                return action
                            else:
                                selected = False
                                self.draw_board(board, 0)

                        else:
                            if board[y, x] == 1:
                                start_y, start_x = y, x
                                possible_board = self.logic.get_possible_board(y, x, board)
                                self.draw_possibles(possible_board)
                                selected = True

