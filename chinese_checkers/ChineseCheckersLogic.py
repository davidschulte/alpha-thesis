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

GOLD = 3;
SILVER = 1;
BRONZE = 0;

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

    def get_reachables(self, y, x, reachables=[], jumping=False):
        neighbors = self.get_neighbors(y, x, self.np_pieces)
        if jumping:
            for (yN, xN, valN, dir) in neighbors:
                if valN not in [EMPTY, OUT]:
                    (yNN, xNN, valNN) = self.get_neighbor(yN, xN, dir, self.np_pieces)
                    if (yNN, xNN) not in reachables and valNN == EMPTY:
                        reachables.append((yNN, xNN))
                        reachables = list(set().union(reachables, self.get_reachables(yNN, xNN, self.np_pieces, reachables, True)))
        else:
            for (yN, xN, valN, _) in neighbors:
                if valN == EMPTY:
                    reachables.append((yN, xN))
            reachables = list(set().union(reachables, self.get_reachables(y, x, self.np_pieces, reachables, True)))
        return reachables


    def move(self, player, start_coordinates, end_coordinates):
        self.np_pieces[start_coordinates[0], start_coordinates[1]] = EMPTY
        self.np_pieces[end_coordinates[0], end_coordinates[1]] = player

    def get_done(self, player):
        return False not in (self.np_pieces[END == player] == player)

    def get_win_state(self):
        still_playing = []
        for player in [1,2,3]:
            if self.scores[player - 1] == 0:
                still_playing.append(player)

        prize = PRIZES[-len(still_playing)]
        for player in still_playing:
            if False not in (self.get_done(player)):
                self.scores[player - 1] = prize
                still_playing.remove(player)

        if len(still_playing) < 2:
            return (True, self.scores)
        else:
            return (False, self.scores)


    #def boardReduce(self):
    def get_next_player(self,player):
        return player % 3 + 1

    def get_legal_moves(self, player):
        legal_moves = [];
        player_y_list, player_x_list = np.where(self.np_pieces == player)
        for i in range(len(player_y_list)):
            y_start, x_start = (player_y_list[i], player_x_list[i])
            reachables = self.get_reachables(y_start, x_start)
            for y_end, x_end in reachables:
                legal_moves.append((y_start, x_start, y_end, x_end))
        return legal_moves

    def decode_coordinates(self, coded):
        (x_coordinates, y_coordinates) = np.where(START != OUT)
        return y_coordinates[coded], x_coordinates[coded]

    def encode_coordinates(self, y, x):
        (x_coordinates, y_coordinates) = np.where(START != OUT)
        y_fits = np.where(y_coordinates == y)
        x_fits = np.where(x_coordinates == x)
        index_list = np.intersect1d(y_fits, x_fits)
        return index_list[0]

    def encode_board(self, board):
        return board[board != OUT]

    # def get_end_by_player(self, player):
    #     end_board = np.copy(END)
    #
    #     shift = player - 1;
    #     for p in [1,2,3]:
    #         end_board[END == p] = (p + shift) % 3
    #
    #     return end_board
