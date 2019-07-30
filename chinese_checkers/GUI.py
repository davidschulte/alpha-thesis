import pygame
import numpy as np
MOVES = [[0, 1], [-1, 1], [1, -1], [1, 0]]
Y_OFFSET = 17
X_OFFSET = 17
Y_STEP = 47.2
X_STEP = 54.5
R = 12
RED = (255,0,0)
GREEN = (0, 200, 0 )
LIGHT_GREEN = (0, 100, 0)
BLUE = (0,0,255,255)
YELLOW = (255,255,0)
PINK = (255,105,180)
BLACK = (0,0,0,0)
WHITE = (255, 255, 255)

ENDPOINTS_RED = [[0, 9], [3, 6], [3, 9]]
STARTPOINTS_RED = [[12, 3], [9, 3], [9, 6]]
ENDPOINTS_YELLOW = [[9, 9], [9, 6], [6, 9]]
STARTPOINTS_YELLOW = [[3, 3], [6, 3], [3, 6]]
ENDPOINTS_GREEN = [[9, 0], [6, 3], [9, 3]]
STARTPOINTS_GREEN = [[3, 12], [3, 9], [6, 9]]

AREAS = [ENDPOINTS_RED, STARTPOINTS_RED, ENDPOINTS_YELLOW, STARTPOINTS_YELLOW, ENDPOINTS_GREEN, STARTPOINTS_GREEN]

line_width = 3
                 # 0  1  2  3  4  5  6  7  8  9 10 11 12
START = np.array([[4, 4, 4, 4, 4, 4, 4, 4, 4, 0, 4, 4, 4], #0
                  [4, 4, 4, 4, 4, 4, 4, 4, 0, 0, 4, 4, 4], #1
                  [4, 4, 4, 4, 4, 4, 4, 0, 0, 0, 4, 4, 4], #2
                  [4, 4, 4, 2, 2, 2, 2, 0, 0, 3, 3, 3, 3], #3
                  [4, 4, 4, 2, 2, 2, 0, 0, 0, 3, 3, 3, 4], #4
                  [4, 4, 4, 2, 2, 0, 0, 0, 0, 3, 3, 4, 4], #5
                  [4, 4, 4, 2, 0, 0, 0, 0, 0, 3, 4, 4, 4], #6
                  [4, 4, 0, 0, 0, 0, 0, 0, 0, 0, 4, 4, 4], #7
                  [4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 4, 4], #8
                  [0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 4, 4, 4], #9
                  [4, 4, 4, 1, 1, 1, 4, 4, 4, 4, 4, 4, 4], #10
                  [4, 4, 4, 1, 1, 4, 4, 4, 4, 4, 4, 4, 4], #11
                  [4, 4, 4, 1, 4, 4, 4, 4, 4, 4, 4, 4, 4]]).astype('int8')#12

                # 0  1  2  3  4  5  6  7  8  9 10 11 12
GRID = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0], #0
                 [0, 0, 0, 0, 0, 0, 0, 0, 2, 3, 0, 0, 0], #1
                 [0, 0, 0, 0, 0, 0, 0, 1, 4, 1, 0, 0, 0], #2
                 [0, 0, 0, 0, 0, 0, 2, 3, 2, 3, 0, 0, 0], #3
                 [0, 0, 0, 0, 0, 1, 4, 1, 4, 1, 0, 0, 0], #4
                 [0, 0, 0, 0, 2, 3, 2, 3, 2, 3, 0, 0, 0], #5
                 [0, 0, 0, 1, 4, 1, 4, 1, 4, 1, 0, 0, 0], #6
                 [0, 0, 0, 3, 2, 3, 2, 3, 2, 0, 0, 0, 0], #7
                 [0, 0, 0, 1, 4, 1, 4, 1, 0, 0, 0, 0, 0], #8
                 [0, 0, 0, 3, 2, 3, 2, 0, 0, 0, 0, 0, 0], #9
                 [0, 0, 0, 1, 4, 1, 0, 0, 0, 0, 0, 0, 0], #10
                 [0, 0, 0, 3, 2, 0, 0, 0, 0, 0, 0, 0, 0], #11
                 [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],]).astype('int8') #12


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

        pygame.font.init()
        self.myfont = pygame.font.SysFont('Arial', 15)

    def draw_figure(self, surface, row, column, radius, color):
        y = Y_OFFSET + row * Y_STEP
        x = X_OFFSET + column * X_STEP + (row - 9) * X_STEP / 2
        pygame.draw.circle(surface, color, (int(x),int(y)), radius)


    def coordinates_to_pos(self, row, column):
        y = Y_OFFSET + row * Y_STEP
        x = X_OFFSET + column * X_STEP + (row - 9) * X_STEP / 2
        return y, x


    def draw_line(self, surface, y1, x1, y2, x2, color):
        pygame.draw.line(surface,color, (x1, y1), (x2, y2), line_width)


    def draw_lines(self, surface):
        for row in range(13):
            for column in range(13):
                for m in range(4):
                    n_y, n_x = row + MOVES[m][0], column + MOVES[m][1]
                    if n_y < 13 and n_x < 13 and n_y > -1 and n_x > -1:
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

        for y in range(13):
            for x in range(13):
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

        # pygame.time.wait(500)
            # timer -= 1
