import numpy as np

HEIGHT = 17
WIDTH = 13
OUT = 4
EMPTY = 0
LEFT = 1
RIGHT = 2
LEFT_UP = 3
RIGHT_UP = 4
LEFT_DOWN = 5
RIGHT_DOWN = 6

GOLD = 3
SILVER = 1
BRONZE = 0

PRIZES = [GOLD, SILVER, BRONZE]

START = np.array([(4, 4, 4, 4, 4, 4, 0, 4, 4, 4, 4, 4, 4), #0
                 (4, 4, 4, 4, 4, 0, 0, 4, 4, 4, 4, 4, 4), #1
                 (4, 4, 4, 4, 4, 0, 0, 0, 4, 4, 4, 4, 4), #2
                 (4, 4, 4, 4, 0, 0, 0, 0, 4, 4, 4, 4, 4), #3
                 (2, 2, 2, 2, 2, 0, 0, 0, 3, 3, 3, 3, 3), #4
                 (2, 2, 2, 2, 0, 0, 0, 0, 3, 3, 3, 3, 4), #5
                 (4, 2, 2, 2, 0, 0, 0, 0, 0, 3, 3, 3, 4), #6
                 (4, 2, 2, 0, 0, 0, 0, 0, 0, 3, 3, 4, 4), #7
                 (4, 4, 2, 0, 0, 0, 0, 0, 0, 0, 3, 4, 4), #8
                 (4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 4), #9
                 (4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4), #10
                 (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4), #11
                 (0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0), #12
                 (4, 4, 4, 4, 1, 1, 1, 1, 4, 4, 4, 4, 4), #13
                 (4, 4, 4, 4, 4, 1, 1, 1, 4, 4, 4, 4, 4), #14
                 (4, 4, 4, 4, 4, 1, 1, 4, 4, 4, 4, 4, 4), #15
                 (4, 4, 4, 4, 4, 4, 1, 4, 4, 4, 4, 4, 4)]) #16

                # 0  1  2  3  4  5  6  7  8  9 10 11 12
END = np.array([ (0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0),#0
                 (0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0),#1
                 (0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0),#2
                 (0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0),#3
                 (0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0),#4
                 (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),#5
                 (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),#6
                 (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),#7
                 (0, 0, 3, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0),#8
                 (0, 3, 3, 0, 0, 0, 0, 0, 0, 2, 2, 0, 0),#9
                 (0, 3, 3, 3, 0, 0, 0, 0, 0, 2, 2, 2, 0),#10
                 (3, 3, 3, 3, 0, 0, 0, 0, 2, 2, 2, 2, 0),#11
                 (3, 3, 3, 3, 3, 0, 0, 0, 2, 2, 2, 2, 2),#12
                 (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),#13
                 (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),#14
                 (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),#15
                 (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)])#16

                          # 0   1   2   3   4   5   6   7   8   9  10  11  12
PLAYER1Board = np.array([(  0,  0,  0,  0,  0,  0,  1,  0,  0,  0,  0,  0,  0), #0
                         (  0,  0,  0,  0,  0,  2,  3,  0,  0,  0,  0,  0,  0), #1
                         (  0,  0,  0,  0,  4,  5,  6,  0,  0,  0,  0,  0,  0), #2
                         (  0,  0,  0,  0,  7,  8,  9, 10,  0,  0,  0,  0,  0), #3
                         ( 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23), #4
                         ( 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35,  0), #5
                         (  0, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46,  0), #6
                         (  0, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56,  0,  0), #7
                         (  0,  0, 57, 58, 59, 60, 61, 62, 63, 64, 65,  0,  0), #8
                         (  0, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75,  0,  0), #9
                         (  0, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86,  0), #10
                         ( 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98,  0), #11
                         ( 99,100,101,102,103,104,105,106,107,108,109,110,111), #12
                         (  0,  0,  0,  0,112,113,114,115,  0,  0,  0,  0,  0), #13
                         (  0,  0,  0,  0,  0,116,117,118,  0,  0,  0,  0,  0), #14
                         (  0,  0,  0,  0,  0,119,120,  0,  0,  0,  0,  0,  0), #15
                         (  0,  0,  0,  0,  0,  0,121,  0,  0,  0,  0,  0,  0)])#16

                          # 1   2   3   4   5   6   7   8   9  10  11  12  13
PLAYER2Board = np.array([(  0,  0,  0,  0,  0,  0, 99,  0,  0,  0,  0,  0,  0), #1
                         (  0,  0,  0,  0,  0,100, 87,  0,  0,  0,  0,  0,  0), #2
                         (  0,  0,  0,  0,  0,101, 88, 76,  0,  0,  0,  0,  0), #3
                         (  0,  0,  0,  0,102, 89, 77, 66,  0,  0,  0,  0,  0), #4
                         (121,119,116,112,103, 90, 78, 67, 57, 47, 36, 24, 11), #5
                         (120,117,113,104, 91, 79, 68, 58, 48, 37, 25, 12,  0), #6
                         (  0,118,114,105, 92, 80, 69, 59, 49, 38, 26, 13,  0), #7
                         (  0,115,106, 93, 81, 70, 60, 50, 39, 27, 14,  0,  0), #8
                         (  0,  0,107, 94, 82, 71, 61, 51, 40, 28, 15,  0,  0), #9
                         (  0,108, 95, 83, 72, 62, 52, 41, 29, 16,  7,  0,  0), #10
                         (  0,109, 96, 84, 73, 63, 53, 42, 30, 17,  8,  4,  0), #11
                         (110, 97, 85, 74, 64, 54, 43, 31, 18,  9,  5,  2,  0), #12
                         (111, 98, 86, 75, 65, 55, 44, 32, 19, 10,  6,  3,  1), #13
                         (  0,  0,  0,  0, 56, 45, 33, 20,  0,  0,  0,  0,  0), #14
                         (  0,  0,  0,  0,  0, 46, 34, 21,  0,  0,  0,  0,  0), #15
                         (  0,  0,  0,  0,  0, 35, 22,  0,  0,  0,  0,  0,  0), #16
                         (  0,  0,  0,  0,  0,  0, 23,  0,  0,  0,  0,  0,  0)]) #17

                          # 1   2   3   4   5   6   7   8   9  10  11  12  13
PLAYER3Board = np.array([(  0,  0,  0,  0,  0,  0,111,  0,  0,  0,  0,  0,  0), #1
                         (  0,  0,  0,  0,  0, 98,110,  0,  0,  0,  0,  0,  0), #2
                         (  0,  0,  0,  0,  0, 86, 97,109,  0,  0,  0,  0,  0), #3
                         (  0,  0,  0,  0, 75, 85, 96,108,  0,  0,  0,  0,  0), #4
                         ( 23, 35, 46, 56, 65, 74, 84, 95,107,115,118,120,121), #5
                         ( 22, 34, 45, 55, 64, 73, 83, 94,106,114,117,119,  0), #6
                         (  0, 21, 33, 44, 54, 63, 72, 82, 93,105,113,116,  0), #7
                         (  0, 20, 32, 43, 53, 62, 71, 81, 92,104,112,  0,  0), #8
                         (  0,  0, 19, 31, 42, 52, 61, 70, 80, 91,103,  0,  0), #9
                         (  0, 10, 18, 30, 41, 51, 60, 69, 79, 90,102,  0,  0), #10
                         (  0,  6,  9, 17, 29, 40, 50, 59, 68, 78, 89,101,  0), #11
                         (  3,  5,  8, 16, 28, 39, 49, 58, 67, 77, 88,100,  0), #12
                         (  1,  2,  4,  7, 15, 27, 38, 48, 57, 66, 76, 87, 99), #13
                         (  0,  0,  0,  0, 14, 26, 37, 47,  0,  0,  0,  0,  0), #14
                         (  0,  0,  0,  0,  0, 13, 25, 36,  0,  0,  0,  0,  0), #15
                         (  0,  0,  0,  0,  0, 12, 24,  0,  0,  0,  0,  0,  0), #16
                         (  0,  0,  0,  0,  0,  0, 11,  0,  0,  0,  0,  0,  0)]) #17

class Board():

    def __init__(self, np_pieces=None):
        if np_pieces is None:
            self.np_pieces = START
        else:
            self.np_pieces = np_pieces
        self.scores = [0, 0, 0]

    def get_start(self):
        return START

    # 1=left, 2=right, 3=lu, 4=ru, 5=ld, 6=rd
    def get_neighbor(self, y, x, dir, board):
        if y == 0 and dir in [LEFT_UP, RIGHT_UP]:
            return (0, 0, OUT)  ##critical!!
        if y == HEIGHT - 1 and dir in [LEFT_DOWN, RIGHT_DOWN]:
            return (0, 0, OUT)
        if x == 0 and dir in [LEFT, LEFT_UP, LEFT_DOWN]:
            return (0, 0, OUT)
        if x == WIDTH - 1 and dir in [RIGHT, RIGHT_UP, RIGHT_DOWN]:
            return (0, 0, OUT)

        if dir == LEFT:
            return (y, x - 1, board[y, x - 1])
        if dir == RIGHT:
            return (y, x + 1, board[y, x + 1])

        if y % 2 == 0:
            if dir == LEFT_UP:
                return (y - 1, x - 1, board[y - 1, x - 1])
            if dir == RIGHT_UP:
                return (y - 1, x, board[y - 1, x])
            if dir == LEFT_DOWN:
                return (y + 1, x - 1, board[y + 1, x - 1])
            if dir == RIGHT_DOWN:
                return (y + 1, x, board[y + 1, x])
        else:
            if dir == LEFT_UP:
                return (y - 1, x, board[y - 1, x])
            if dir == RIGHT_UP:
                return (y - 1, x + 1, board[y - 1, x + 1])
            if dir == LEFT_DOWN:
                return (y + 1, x, board[y + 1, x])
            if dir == RIGHT_DOWN:
                return (y + 1, x + 1, board[y + 1, x + 1])

    def get_neighbors(self, y, x, board):
        neighbors = []
        for dir in [LEFT, RIGHT, LEFT_UP, RIGHT_UP, LEFT_DOWN, RIGHT_DOWN]:
            (yN, xN, valN) = self.get_neighbor(y, x, dir, board)
            if valN != 4:
                neighbors.append((yN, xN, valN, dir))
        return neighbors

    def get_reachables(self, y, x, board, reachables=[], jumping=False):
        neighbors = self.get_neighbors(y, x, board)
        if jumping:
            for (yN, xN, valN, dir) in neighbors:
                if valN not in [EMPTY, OUT]:
                    (yNN, xNN, valNN) = self.get_neighbor(yN, xN, dir, board)
                    if (yNN, xNN) not in reachables and valNN == EMPTY:
                        reachables.append((yNN, xNN))
                        reachables = list(set().union(reachables, self.get_reachables(yNN, xNN, board, reachables, True)))
        else:
            for (yN, xN, valN, _) in neighbors:
                if valN == EMPTY:
                    reachables.append((yN, xN))
            reachables = list(set().union(reachables, self.get_reachables(y, x, self.np_pieces, reachables, True)))
        return reachables


    def move(self, player, start_coordinates, end_coordinates):
        self.np_pieces[start_coordinates[0], start_coordinates[1]] = EMPTY
        self.np_pieces[end_coordinates[0], end_coordinates[1]] = player

    def get_done(self, board, player):
        return False not in (board[END == player] == player)

    def get_win_state(self, board):
        still_playing = []
        for player in [1,2,3]:
            if self.scores[player - 1] == 0:
                still_playing.append(player)

        prize = PRIZES[-len(still_playing)]
        for player in still_playing:
            test = self.get_done(board, player)
            if self.get_done(board, player):
                self.scores[player - 1] = prize
                still_playing.remove(player)

        if len(still_playing) < 2:
            return (True, self.scores)
        else:
            return (False, self.scores)


    #def boardReduce(self):
    def get_next_player(self,player):
        return player % 3 + 1

    def get_legal_moves(self, board, player):
        legal_moves = [];
        player_y_list, player_x_list = np.where(board == player)
        for i in range(len(player_y_list)):
            y_start, x_start = (player_y_list[i], player_x_list[i])
            reachables = self.get_reachables(y_start, x_start, board)
            for y_end, x_end in reachables:
                legal_moves.append((y_start, x_start, y_end, x_end))
        return legal_moves

    def decode_coordinates(self, coded):
        (x_coordinates, y_coordinates) = np.where(START != OUT)
        return y_coordinates[coded], x_coordinates[coded]

    def encode_coordinates(self, y, x):
        (y_coordinates, x_coordinates) = np.where(START != OUT)
        y_fits = np.where(y_coordinates == y)
        x_fits = np.where(x_coordinates == x)
        index_list = np.intersect1d(y_fits, x_fits)
        return index_list[0]

    def rotate_board(self, board, player):
        if player == 1:
            return board

        if player == 2:
            reference_board = PLAYER2Board
        else:
            reference_board = PLAYER3Board

        rotated_board = np.copy(board)
        for i in range(121):
            rotated_board[PLAYER1Board == i+1] = board[reference_board == i+1]

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
