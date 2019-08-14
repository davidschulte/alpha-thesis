import numpy as np

HEIGHT = 13
WIDTH = 13
OUT = 4
EMPTY = 0
LEFT = 0
RIGHT = 1
LEFT_UP = 2
RIGHT_UP = 3
LEFT_DOWN = 4
RIGHT_DOWN = 5
MOVES = [[0, -1], [0, 1], [-1, 0], [-1, 1], [1, -1], [1, 0]]

GOLD = 3
SILVER = 1
BRONZE = 1

PRIZES = [GOLD, SILVER, BRONZE]

ACTION_SIZE_OFFSET = [49 * 6, 16 * 16, 12 * 12, 12 * 12, 9 * 9]
ACTION_SUB_SPACE = [6, 16, 12, 12, 9]

                       # 0  1  2  3  4  5  6  7  8  9 10 11 12
EMPTY_BOARD = np.array([[4, 4, 4, 4, 4, 4, 4, 4, 4, 0, 4, 4, 4], #0
                        [4, 4, 4, 4, 4, 4, 4, 4, 0, 0, 4, 4, 4], #1
                        [4, 4, 4, 4, 4, 4, 4, 0, 0, 0, 4, 4, 4], #2
                        [4, 4, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], #3
                        [4, 4, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4], #4
                        [4, 4, 4, 0, 0, 0, 0, 0, 0, 0, 0, 4, 4], #5
                        [4, 4, 4, 0, 0, 0, 0, 0, 0, 0, 4, 4, 4], #6
                        [4, 4, 0, 0, 0, 0, 0, 0, 0, 0, 4, 4, 4], #7
                        [4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 4, 4], #8
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 4, 4], #9
                        [4, 4, 4, 0, 0, 0, 4, 4, 4, 4, 4, 4, 4], #10
                        [4, 4, 4, 0, 0, 4, 4, 4, 4, 4, 4, 4, 4], #11
                        [4, 4, 4, 0, 4, 4, 4, 4, 4, 4, 4, 4, 4]]).astype('int8')#12.

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
                  [4, 4, 4, 1, 4, 4, 4, 4, 4, 4, 4, 4, 4]]).astype('int8')#12.

                #0  1  2  3  4  5  6  7  8  9 10 11 12
END = np.array([[4, 4, 4, 4, 4, 4, 4, 4, 4, 1, 4, 4, 4], #0
                [4, 4, 4, 4, 4, 4, 4, 4, 1, 1, 4, 4, 4], #1
                [4, 4, 4, 4, 4, 4, 4, 1, 1, 1, 4, 4, 4], #2
                [4, 4, 4, 4, 4, 4, 1, 1, 1, 1, 4, 4, 4], #3
                [4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4], #4
                [4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4], #5
                [4, 4, 4, 3, 4, 4, 4, 4, 4, 2, 4, 4, 4], #6
                [4, 4, 3, 3, 4, 4, 4, 4, 2, 2, 4, 4, 4], #7
                [4, 3, 3, 3, 4, 4, 4, 2, 2, 2, 4, 4, 4], #8
                [3, 3, 3, 3, 4, 4, 2, 2, 2, 2, 4, 4, 4], #9
                [4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4], #10
                [4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4], #11
                [4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4]]).astype('int8')#12

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

ROTATION_LEFT = np.array([  0,  0,  0,  0,  0,  0,  0,  0,  0,117,  0,  0,  0,
           0,  0,  0,  0,  0,  0,  0,  0,118,105,  0,  0,  0,
           0,  0,  0,  0,  0,  0,  0,119,106, 93,  0,  0,  0,
           0,  0,  0,159,146,133,120,107, 94, 81, 68, 55, 42,
           0,  0,  0,147,134,121,108, 95, 82, 69, 56, 43,  0,
           0,  0,  0,135,122,109, 96, 83, 70, 57, 44,  0,  0,
           0,  0,  0,123,110, 97, 84, 71, 58, 45,  0,  0,  0,
           0,  0,124,111, 98, 85, 72, 59, 46, 33,  0,  0,  0,
           0,125,112, 99, 86, 73, 60, 47, 34, 21,  0,  0,  0,
           126,113,100, 87, 74, 61, 48, 35, 22,  9,  0,  0,  0,
           0,  0,  0, 75, 62, 49,  0,  0,  0,  0,  0,  0,  0,
           0,  0,  0, 63, 50,  0,  0,  0,  0,  0,  0,  0,  0,
           0,  0,  0, 51,  0,  0,  0,  0,  0,  0,  0,  0,  0])

ROTATION_RIGHT = np.array([ 0, 0, 0, 0, 0, 0, 0, 0, 0, 126, 0, 0, 0,
           0, 0, 0, 0, 0, 0, 0, 0, 113, 125, 0, 0, 0,
           0, 0, 0, 0, 0, 0, 0, 100, 112, 124, 0, 0, 0,
           0, 0, 0, 51, 63, 75, 87, 99, 111, 123, 135, 147, 159,
           0, 0, 0, 50, 62, 74, 86, 98, 110, 122, 134, 146, 0,
           0, 0, 0, 49, 61, 73, 85, 97, 109, 121, 133, 0, 0,
           0, 0, 0, 48, 60, 72, 84, 96, 108, 120, 0, 0, 0,
           0, 0, 35, 47, 59, 71, 83, 95, 107, 119, 0, 0, 0,
           0, 22, 34, 46, 58, 70, 82, 94, 106, 118, 0, 0, 0,
           9, 21, 33, 45, 57, 69, 81, 93, 105, 117, 0, 0, 0,
           0, 0, 0, 44, 56, 68, 0, 0, 0, 0, 0, 0, 0,
           0, 0, 0, 43, 55, 0, 0, 0, 0, 0, 0, 0, 0,
           0, 0, 0, 42, 0, 0, 0, 0, 0, 0, 0, 0, 0])



class Board():

    def __init__(self):
        self.scores = np.array([0, 0, 0])
        self.scores_temporary = np.array([0, 0, 0])

    def get_start(self):
        return np.copy(START)

    # 1=left, 2=right, 3=lu, 4=ru, 5=ld, 6=rd
    def get_neighbor(self, y, x, dir, board):
        if y == 0 and dir in [LEFT_UP, RIGHT_UP]:
            return 0, 0, OUT
        if y == HEIGHT - 1 and dir in [LEFT_DOWN, RIGHT_DOWN]:
            return 0, 0, OUT
        if x == 0 and dir in [LEFT, LEFT_DOWN]:
            return 0, 0, OUT
        if x == WIDTH - 1 and dir in [RIGHT, RIGHT_UP]:
            return 0, 0, OUT

        yN, xN = y + MOVES[dir][0], x + MOVES[dir][1]

        if GRID[yN, xN] == 0:
            return 0, 0, OUT

        return yN, xN, board[yN, xN]

    def step(self, y, x, direction, board, step_size):
        if y < step_size and direction in [LEFT_UP, RIGHT_UP]:
            return 0, 0, OUT
        if y > HEIGHT - 1 - step_size and direction in [LEFT_DOWN, RIGHT_DOWN]:
            return 0, 0, OUT
        if x < step_size and direction in [LEFT, LEFT_DOWN]:
            return 0, 0, OUT
        if x > WIDTH - 1 - step_size and direction in [RIGHT, RIGHT_UP]:
            return 0, 0, OUT

        yN, xN = y + step_size * MOVES[direction][0], x + step_size * MOVES[direction][1]

        if GRID[yN, xN] == 0:
            return 0, 0, OUT

        return yN, xN, board[yN, xN]

    def get_jumps(self, y, x, board):
        jumps = []
        for direction in range(6):
            y_nn, x_nn, field_nn = self.step(y, x, direction, board, 2)
            if field_nn == EMPTY and self.right_zone(y, x, y_nn, x_nn):
                _, _, field_n = self.step(y, x, direction, board, 1)
                if field_n != 0:
                    jumps.append((y_nn, x_nn))
        return jumps

    def get_neighbors(self, y, x, board):
        neighbors = []
        for dir in [LEFT, RIGHT, LEFT_UP, RIGHT_UP, LEFT_DOWN, RIGHT_DOWN]:
            (yN, xN, valN) = self.get_neighbor(y, x, dir, board)
            if valN != OUT:
                neighbors.append((yN, xN, valN, dir))
        return neighbors

    def get_reachables_direct(self, y, x, board):
        reachables = []
        neighbors = self.get_neighbors(y, x, board)
        for (yN, xN, valN, dirN) in neighbors:
            if valN == EMPTY and self.right_zone(y, x, yN, xN):
                reachables.append((yN, xN, dirN))

        return reachables

    def get_reachables_jump(self, y, x, board):
        reachables = [(y,x)]
        start_index = 0
        while True:
            end_index = len(reachables)
            for i in range(start_index, end_index):
                (y, x) = reachables[i]
                jump_list = self.get_jumps(y, x, board)
                for j in range(len(jump_list)):
                    (y_nn, x_nn) = jump_list[j]
                    if (y_nn, x_nn) not in reachables:
                        reachables.append((y_nn, x_nn))

            if start_index == end_index:
                del reachables[0]
                return reachables
            start_index = end_index

    def right_zone(self, y_start, x_start, y_end, x_end):
        # if START[y_start, x_start] != 1 and START[y_end, x_end] == 1:
        #     return False
        if END[y_start, x_start] == 1 and END[y_end, x_end] != 1:
            return False
        return True

    def move(self, y_start, x_start, y_end, x_end, board, player):
        board[y_start, x_start] = EMPTY
        board[y_end, x_end] = player
        return board

    def get_done(self, board, player, color_matters):
        if color_matters:
            return False not in (board[END == player] == player)
        else:
            return False not in (board[END == player] != EMPTY)

    def get_win_state(self, board, temporary):
        if temporary:
            scores = self.scores_temporary
            for player in [1, 2, 3]:
                if not self.get_done(board, player, False):
                    scores[player - 1] = 0
        else:
            scores = self.scores

        first_found = False

        still_playing = []
        for player in [1, 2, 3]:
            if scores[player - 1] == 0:
                still_playing.append(player)

        prize = PRIZES[-len(still_playing)]
        if len(still_playing) < 3:
            color_matters = False
        else:
            color_matters = True
        for player in still_playing:
            if self.get_done(board, player, color_matters):
                scores[player - 1] = prize
                still_playing.remove(player)
                first_found = True

        if not temporary:
            self.scores_temporary = np.copy(scores)

        if first_found:
            return self.get_win_state(board, temporary)

        return scores

    def get_next_player(self, player):
        return player % 3 + 1

    def get_previous_player(self, player):
        previous = player - 1
        if previous == 0:
            previous = 3
        return previous

    def get_legal_moves(self, board, player):
        legal_moves_direct = []
        legal_moves_jumping = []

        player_y_list, player_x_list = np.where(board == player)
        for i in range(len(player_y_list)):
            y_start, x_start = (player_y_list[i], player_x_list[i])
            reachables_direct = self.get_reachables_direct(y_start, x_start, board)
            reachables_jumping = self.get_reachables_jump(y_start, x_start, board)
            for (_, _, dir) in reachables_direct:
                legal_moves_direct.append((y_start, x_start, dir))

            for (y_end, x_end) in reachables_jumping:
                legal_moves_jumping.append((y_start, x_start, y_end, x_end))

        return legal_moves_direct, legal_moves_jumping

    def encode_move_direct(self, y_start, x_start, direction):
        start = self.encode_coordinates(y_start, x_start)

        return start * ACTION_SUB_SPACE[0] + direction

    def encode_move_jumping(self, y_start, x_start, y_end, x_end):
        grid_no, start = self.encode_coordinates_grid(y_start, x_start)
        grid_nu, end = self.encode_coordinates_grid(y_end, x_end)

        # encoded = sum(ACTION_SIZE_OFFSET[0:grid_no]) - 1 + start * ACTION_SUB_SPACE[grid_no] + end
        return sum(ACTION_SIZE_OFFSET[0:grid_no]) + start * ACTION_SUB_SPACE[grid_no] + end

    def decode_move(self, move):
        grid = 0

        while move > ACTION_SIZE_OFFSET[grid]:
            move -= ACTION_SIZE_OFFSET[grid]
            grid += 1

        start_position, direction = divmod(move, ACTION_SUB_SPACE[grid])

        if grid == 0:
            y_start, x_start = self.decode_coordinates(start_position)
            y_end, x_end = y_start + MOVES[direction][0], x_start + MOVES[direction][1]
        else:
            y_start, x_start = self.decode_coordinates_grid(start_position, grid)
            y_end, x_end = self.decode_coordinates_grid(direction, grid)

        return y_start, x_start, y_end, x_end

    def decode_coordinates(self, encoded):
        y_coordinates, x_coordinates = np.where(GRID != 0)
        return y_coordinates[encoded], x_coordinates[encoded]

    def decode_coordinates_grid(self, encoded, grid_no):
        y_coordinates, x_coordinates = np.where(GRID == grid_no)
        return y_coordinates[encoded], x_coordinates[encoded]

    # def encode_coordinates(self, y, x):
    #     y_coordinates, x_coordinates = np.where(GRID != 0)
    #     y_fits = np.where(y_coordinates == y)
    #     x_fits = np.where(x_coordinates == x)
    #     index_list = np.intersect1d(y_fits, x_fits)
    #     if index_list[0] != self.alternative_encode_coordinates(y, x):
    #         print("ERROR!")
    #     return index_list[0]

    def encode_coordinates(self, y, x):
        if y < 7:
            g_base = (y**2 + y) / 2
            g_plus = y + x - 8
        else:
            g_base = 49 - (13 - y) * (14 - y ) / 2
            g_plus = x - 2

        return int(g_base + g_plus - 1)

    def encode_coordinates_grid(self, y, x):
        grid_no = GRID[y,x]

        if grid_no == 1:
            if y < 7:
                g_base = y * (y + 2 ) / 8
                g_plus = (x + y + 1) / 2 - 4
            else:
                g_base = 16 - (14 - y) * (16 - y) / 8
                g_plus = (x + 1) / 2 - 1
        elif grid_no == 2:
            if y < 7:
                g_base = (y - 1) * (y + 1) / 8
                g_plus = (x + y + 1) / 2 - 4
            else:
                g_base = 12 - (13 - y) * (15 - y) / 8
                g_plus = x / 2 - 1
        elif grid_no == 3:
            if y < 7:
                g_base = (y - 1) * (y + 1) / 8
                g_plus = (x + y) / 2 - 4
            else:
                g_base = 12 - (13 - y) * (15 - y) / 8
                g_plus = (x - 1) / 2
        else:
            if y < 7:
                g_base = (y - 2) * y / 8
                g_plus = (x + y) / 2 - 4
            else:
                g_base = 9 - (12 - y) * (14 - y) / 8
                g_plus = x / 2 - 1

        return grid_no, int(g_base + g_plus - 1)

    def rotate_board(self, rotation_board, start_player, end_player):
        rotation_board = rotation_board.reshape(13 * 13)

        if self.get_next_player(start_player) == end_player:
            right = False
        else:
            right = True

        rotation_board = self.rotate(rotation_board, right)

        return rotation_board.reshape(13, 13)
        # rotated_board = np.copy(board)
        # for i in range(121):
        #     rotated_board[PLAYER1Board == i+1] = board[reference_board == i+1]
        #
        # return rotated_board

    def rotate(self, rotation_board, right):
        if right:
            rotation_index_board = ROTATION_RIGHT
        else:
            rotation_index_board = ROTATION_LEFT
        rotated_board = np.copy(EMPTY_BOARD.reshape(13 * 13))
        for i in range(len(rotation_board)):
            if rotation_board[i] not in [OUT, EMPTY]:
                rotated_board[rotation_index_board[i]] = rotation_board[i]

        return rotated_board
