import numpy as np

HEIGHT = 17
WIDTH = 17
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
BRONZE = 0

PRIZES = [GOLD, SILVER, BRONZE]

ACTION_SIZE_OFFSET = [81 * 6, 25 * 25, 20 * 20, 20 * 20, 16 * 16]
ACTION_SUB_SPACE = [6, 25, 20, 20, 16]

#  0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16
START = np.array([(4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 0, 4, 4, 4, 4),  # 0
                  (4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 0, 0, 4, 4, 4, 4),  # 1
                  (4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 0, 0, 0, 4, 4, 4, 4),  # 2
                  (4, 4, 4, 4, 4, 4, 4, 4, 4, 0, 0, 0, 0, 4, 4, 4, 4),  # 3
                  (4, 4, 4, 4, 2, 2, 2, 2, 2, 0, 0, 0, 3, 3, 3, 3, 3),  # 4
                  (4, 4, 4, 4, 2, 2, 2, 2, 0, 0, 0, 0, 3, 3, 3, 3, 4),  # 5
                  (4, 4, 4, 4, 2, 2, 2, 0, 0, 0, 0, 0, 3, 3, 3, 4, 4),  # 6
                  (4, 4, 4, 4, 2, 2, 0, 0, 0, 0, 0, 0, 3, 3, 4, 4, 4),  # 7
                  (4, 4, 4, 4, 2, 0, 0, 0, 0, 0, 0, 0, 3, 4, 4, 4, 4),  # 8
                  (4, 4, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 4, 4, 4),  # 9
                  (4, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 4, 4, 4),  # 10
                  (4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 4, 4, 4),  # 11
                  (0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 4, 4, 4, 4),  # 12
                  (4, 4, 4, 4, 1, 1, 1, 1, 4, 4, 4, 4, 4, 4, 4, 4, 4),  # 13
                  (4, 4, 4, 4, 1, 1, 1, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4),  # 14
                  (4, 4, 4, 4, 1, 1, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4),  # 15
                  (4, 4, 4, 4, 1, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4)])  # 16

#  0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16
END = np.array([(4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 1, 4, 4, 4, 4),  # 0
                (4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 1, 1, 4, 4, 4, 4),  # 1
                (4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 1, 1, 1, 4, 4, 4, 4),  # 2
                (4, 4, 4, 4, 4, 4, 4, 4, 4, 1, 1, 1, 1, 4, 4, 4, 4),  # 3
                (4, 4, 4, 4, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0),  # 4
                (4, 4, 4, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4),  # 5
                (4, 4, 4, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 4),  # 6
                (4, 4, 4, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 4, 4),  # 7
                (4, 4, 4, 4, 3, 0, 0, 0, 0, 0, 0, 0, 2, 4, 4, 4, 4),  # 8
                (4, 4, 4, 3, 3, 0, 0, 0, 0, 0, 0, 2, 2, 4, 4, 4, 4),  # 9
                (4, 4, 3, 3, 3, 0, 0, 0, 0, 0, 2, 2, 2, 4, 4, 4, 4),  # 10
                (4, 3, 3, 3, 3, 0, 0, 0, 0, 2, 2, 2, 2, 4, 4, 4, 4),  # 11
                (3, 3, 3, 3, 3, 0, 0, 0, 2, 2, 2, 2, 2, 4, 4, 4, 4),  # 12
                (4, 4, 4, 4, 0, 0, 0, 0, 4, 4, 4, 4, 4, 4, 4, 4, 4),  # 13
                (4, 4, 4, 4, 0, 0, 0, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4),  # 14
                (4, 4, 4, 4, 0, 0, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4),  # 15
                (4, 4, 4, 4, 0, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4)])  # 16

# 0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16
GRID = np.array([(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0),  # 0
                 (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 2, 0, 0, 0, 0),  # 1
                 (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 4, 1, 0, 0, 0, 0),  # 2
                 (0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 2, 3, 2, 0, 0, 0, 0),  # 3
                 (0, 0, 0, 0, 0, 0, 0, 0, 1, 4, 1, 4, 1, 0, 0, 0, 0),  # 4
                 (0, 0, 0, 0, 0, 0, 0, 3, 2, 3, 2, 3, 2, 0, 0, 0, 0),  # 5
                 (0, 0, 0, 0, 0, 0, 1, 4, 1, 4, 1, 4, 1, 0, 0, 0, 0),  # 6
                 (0, 0, 0, 0, 0, 3, 2, 3, 2, 3, 2, 3, 2, 0, 0, 0, 0),  # 7
                 (0, 0, 0, 0, 1, 4, 1, 4, 1, 4, 1, 4, 1, 0, 0, 0, 0),  # 8
                 (0, 0, 0, 0, 2, 3, 2, 3, 2, 3, 2, 3, 0, 0, 0, 0, 0),  # 9
                 (0, 0, 0, 0, 1, 4, 1, 4, 1, 4, 1, 0, 0, 0, 0, 0, 0),  # 10
                 (0, 0, 0, 0, 2, 3, 2, 3, 2, 3, 0, 0, 0, 0, 0, 0, 0),  # 11
                 (0, 0, 0, 0, 1, 4, 1, 4, 1, 0, 0, 0, 0, 0, 0, 0, 0),  # 12
                 (0, 0, 0, 0, 2, 3, 2, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0),  # 13
                 (0, 0, 0, 0, 1, 4, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),  # 14
                 (0, 0, 0, 0, 2, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),  # 15
                 (0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)])  # 16

ROTATION_LEFT = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 204, 0, 0, 0, 0,
                 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 205, 188, 0, 0, 0, 0,
                 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 206, 189, 172, 0, 0, 0, 0,
                 0, 0, 0, 0, 0, 0, 0, 0, 0, 207, 190, 173, 156, 0, 0, 0, 0,
                 0, 0, 0, 0, 276, 259, 242, 225, 208, 191, 174, 157, 140, 123, 106, 89, 72,
                 0, 0, 0, 0, 260, 243, 226, 209, 192, 175, 158, 141, 124, 107, 90, 73, 0,
                 0, 0, 0, 0, 244, 227, 210, 193, 176, 159, 142, 125, 108, 91, 74, 0, 0,
                 0, 0, 0, 0, 228, 211, 194, 177, 160, 143, 126, 109, 92, 75, 0, 0, 0,
                 0, 0, 0, 0, 212, 195, 178, 161, 144, 127, 110, 93, 76, 0, 0, 0, 0,
                 0, 0, 0, 213, 196, 179, 162, 145, 128, 111, 94, 77, 60, 0, 0, 0, 0,
                 0, 0, 214, 197, 180, 163, 146, 129, 112, 95, 78, 61, 44, 0, 0, 0, 0,
                 0, 215, 198, 181, 164, 147, 130, 113, 96, 79, 62, 45, 28, 0, 0, 0, 0,
                 216, 199, 182, 165, 148, 131, 114, 97, 80, 63, 46, 29, 12, 0, 0, 0, 0,
                 0, 0, 0, 0, 132, 115, 98, 81, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                 0, 0, 0, 0, 116, 99, 82, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                 0, 0, 0, 0, 100, 83, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                 0, 0, 0, 0, 84, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]



class Board():

    def __init__(self, np_pieces=None):
        if np_pieces is None:
            self.np_pieces = START
        else:
            self.np_pieces = np_pieces
        self.scores = np.array([0, 0, 0])
        self.scores_temporary = np.array([0, 0, 0])

    def get_start(self):
        return START

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

    def get_neighbors(self, y, x, board):
        neighbors = []
        for dir in [LEFT, RIGHT, LEFT_UP, RIGHT_UP, LEFT_DOWN, RIGHT_DOWN]:
            (yN, xN, valN) = self.get_neighbor(y, x, dir, board)
            if valN != OUT:
                neighbors.append((yN, xN, valN, dir))
        return neighbors

    def get_reachables_direct(self, y, x, board):
        reachables = [];
        neighbors = self.get_neighbors(y, x, board)
        for (yN, xN, valN, dirN) in neighbors:
            if valN == EMPTY and self.right_zone(y, x, yN, xN):
                reachables.append((yN, xN, dirN))

        return reachables

    def get_reachables_jump(self, y, x, board, reachables=[]):
        neighbors = self.get_neighbors(y, x, board)
        for yN, xN, valN, dir in neighbors:
            if valN != EMPTY:
                (yNN, xNN, valNN) = self.get_neighbor(yN, xN, dir, board)
                if (yNN, xNN) not in reachables and valNN == EMPTY and self.right_zone(y, x, yNN, xNN):
                    reachables.append((yNN, xNN))
                    reachables = list(set().union(reachables, self.get_reachables_jump(yNN, xNN, board, reachables)))

        return reachables

    def right_zone(self, y_start, x_start, y_end, x_end):
        if START[y_start, x_start] != 1 and START[y_end, x_end] == 1:
            return False
        if END[y_start, x_start] == 1 and END[y_end, x_end] != 1:
            return False
        return True
    # def get_reachables(self, y, x, board, reachables_jump=[], jumping=False):
    #     neighbors = self.get_neighbors(y, x, board)
    #     if jumping:
    #         for (yN, xN, valN, dir) in neighbors:
    #             if valN not in [EMPTY, OUT]:
    #                 (yNN, xNN, valNN) = self.get_neighbor(yN, xN, dir, board)
    #                 if (yNN, xNN) not in reachables_jump and valNN == EMPTY:
    #                     reachables.append((yNN, xNN))
    #                     reachables = list(set().union(reachables, self.get_reachables(yNN, xNN, board, reachables, True)))
    #     else:
    #         for (yN, xN, valN, _) in neighbors:
    #             if valN == EMPTY:
    #                 reachables_direct.append((yN, xN))
    #         reachables = list(set().union(reachables, self.get_reachables(y, x, self.np_pieces, reachables, True)))
    #     return reachables

    def move(self, start_coordinates, end_coordinates, player):
        self.np_pieces[start_coordinates[0], start_coordinates[1]] = EMPTY
        self.np_pieces[end_coordinates[0], end_coordinates[1]] = player

    def get_done(self, board, player):
        return False not in (board[END == player] == player)

    def get_win_state(self, board, temporary):
        if temporary:
            scores = self.scores_temporary
            for player in [1, 2, 3]:
                if not self.get_done(board, player):
                    scores[player - 1] = 0
        else:
            scores = self.scores

        still_playing = []
        for player in [1, 2, 3]:
            if scores[player - 1] == 0:
                still_playing.append(player)

        prize = PRIZES[-len(still_playing)]
        for player in still_playing:
            if self.get_done(board, player):
                scores[player - 1] = prize
                still_playing.remove(player)

        if not temporary:
            self.scores_temporary = np.copy(scores)

        if len(still_playing) < 2:
            return True, scores
        else:
            return False, scores

    # def boardReduce(self):
    def get_next_player(self, player):
        return player % 3 + 1

    def get_legal_moves(self, board, player):
        # legal_moves = []
        # player_y_list, player_x_list = np.where(board == player)
        # for i in range(len(player_y_list)):
        #     y_start, x_start = (player_y_list[i], player_x_list[i])
        #     reachables = self.get_reachables(y_start, x_start, board)
        #     for y_end, x_end in reachables:
        #         legal_moves.append((y_start, x_start, y_end, x_end))
        # return legal_moves
        legal_moves_direct = []
        legal_moves_jumping = []

        player_y_list, player_x_list = np.where(board == player)
        for i in range(len(player_y_list)):
            y_start, x_start = (player_y_list[i], player_x_list[i])
            reachables_direct = self.get_reachables_direct(y_start, x_start, board)
            reachables_jumping = self.get_reachables_jump(y_start, x_start, board, [])
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

        encoded = sum(ACTION_SIZE_OFFSET[0:grid_no]) - 1 + start * ACTION_SUB_SPACE[grid_no] + end
        # if encoded >= 81 * 6 + 25 * 25 + 2 * 20 * 20 + 16 * 16 + 1 or grid_no != grid_nu or encoded == 1485:
        #     print("DEBUG")
        return sum(ACTION_SIZE_OFFSET[0:grid_no]) + start * ACTION_SUB_SPACE[grid_no] + end

    # def decode_coordinates(self, coded):
    #     (x_coordinates, y_coordinates) = np.where(START != OUT)
    #     return y_coordinates[coded], x_coordinates[coded]
    def decode_move(self, move):
        # if move == 1485:
        #     print("DEBUG")

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

    def encode_coordinates(self, y, x):
        y_coordinates, x_coordinates = np.where(GRID != 0)
        y_fits = np.where(y_coordinates == y)
        x_fits = np.where(x_coordinates == x)
        index_list = np.intersect1d(y_fits, x_fits)
        if len(index_list) < 1:
            print("DEBUG")
        return index_list[0]

    def encode_coordinates_grid(self, y, x):
        grid_no = GRID[y, x]
        y_coordinates, x_coordinates = np.where(GRID == grid_no)
        y_fits = np.where(y_coordinates == y)
        x_fits = np.where(x_coordinates == x)
        index_list = np.intersect1d(y_fits, x_fits)
        return grid_no, index_list[0]

    def rotate_board(self, rotation_board, start_player, end_player):
        rotation_board = rotation_board.reshape(17 * 17)

        if start_player < end_player:
            rotation_num = end_player - start_player
        else:
            rotation_num = 3 - (start_player - end_player)

        for _ in range(rotation_num):
            rotation_board = self.rotate_left(rotation_board)

        return rotation_board.reshape(17, 17)
        # rotated_board = np.copy(board)
        # for i in range(121):
        #     rotated_board[PLAYER1Board == i+1] = board[reference_board == i+1]
        #
        # return rotated_board

    def rotate_left(self, rotation_board):
        rotated_board = np.copy(START.reshape(17 * 17))
        for i in range(len(rotation_board)):
            if rotation_board[i] != OUT:
                rotated_board[ROTATION_LEFT[i]] = rotation_board[i]

        return rotated_board

    # def encode_board(self, board):
    #     return board[board != OUT]

    # def get_end_by_player(self, player):
    #     end_board = np.copy(END)
    #
    #     shift = player - 1;
    #     for p in [1,2,3]:
    #         end_board[END == p] = (p + shift) % 3
    #
    #     return end_board


