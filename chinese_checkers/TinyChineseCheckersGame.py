from Game import Game
from .TinyChineseCheckersLogic import Logic
import numpy as np


class ChineseCheckersGame(Game):

    def __init__(self):
        Game.__init__(self)
        self.b = Logic()

    def getInitBoard(self):
        """
        :return: start board
        """
        return self.b.get_start()

    def getBoardSize(self):
        return 9, 9

    def getActionSize(self):
        return 25*6+9*9+2*6*6+4*4+1

    def getNextState(self, board, player, action):
        """
        executes a move
        :param board:   current board
        :param player:  current player
        :param action:  action taken
        :return:        the board after the move and the next player
        """
        next_board = np.copy(board)

        if action == self.getActionSize() - 1:
            return board, self.b.get_next_player(player)
        y_start, x_start, y_end, x_end = self.b.decode_move(action)

        if player != 1:
            next_board = self.b.rotate_board(next_board, 1, player)

        next_board = self.b.move(y_start, x_start, y_end, x_end, next_board, player)

        if player != 1:
            next_board = self.b.rotate_board(next_board, player, 1)

        return next_board, self.b.get_next_player(player)

    def getValidMoves(self, board, player):
        """
        :param board:   current board
        :param player:  current player
        :return:        a binary vector, where an entry 1 corresponds to a legal move and 0 to an illegal move
        """
        valids = [0]*self.getActionSize()
        if self.b.get_done(board, player, True):
            valids[-1] = 1
            return valids

        canonical_board = self.getCanonicalForm(board, player)
        legal_moves_direct, legal_moves_jumping = self.b.get_legal_moves(canonical_board)

        for y_start, x_start, direction in legal_moves_direct:
            valids[self.b.encode_move_direct(y_start, x_start, direction)] = 1

        for y_start, x_start, y_end, x_end in legal_moves_jumping:
            valids[self.b.encode_move_jumping(y_start, x_start, y_end, x_end)] = 1

        if sum(valids) == 0:
            valids[-1] = 1
        return valids

    def getGameEnded(self, board, temporary):
        """
        :param board:       current board
        :param temporary:   True if used inside the tree search simulations
                            False if used outside the tree search
        :return:            boolean that denotes if the game is over
        """
        scores = self.b.get_win_state(board, temporary)

        return np.copy(scores)

    def getCanonicalForm(self, board, player):
        """
        :param board:   current board
        :param player:  current player
        :return:        the rotated board r_s(s,p)
        """
        if player == 1:
            return board

        rotation_board = np.copy(board)
        players = [1, 2, 3]
        if player == 2:
            new_players = [3, 1, 2]
        else:
            new_players = [2, 3, 1]

        for p in range(len(players)):
            rotation_board[board == players[p]] = new_players[p]

        return self.b.rotate_board(rotation_board, 1, player)

    def stringRepresentation(self, board):
        """
        converts the board representation to a string, needed to work as a dictionary key in python
        :param board:   current board
        :return:        string representation of the board
        """
        return board.tostring()

    def reset_logic(self):
        self.b = Logic()

    def get_board(self):
        return self.b

    def get_next_player(self, player):
        return self.b.get_next_player(player)

    def get_action_by_coordinates(self, y_start, x_start, y_end, x_end):
        return self.b.get_action_by_coordinates(self, y_start, x_start, y_end, x_end)

    def get_possible_board(self, y, x, board, player):
        """
        :param y:       y position of piece
        :param x:       x position of piece
        :param board:   current board
        :param player:  player
        :return:        board that has positive entries where the specified piece can be moved to
        """

        if player == 1:
            player_revert = 1
        elif player == 2:
            player_revert = 3
        else:
            player_revert = 2

        possible_board = np.zeros((9, 9))
        canonical_board = self.getCanonicalForm(board, player)
        y_start, x_start = self.b.get_canonical_coordinates(y, x, player)
        reachables_direct = self.b.get_reachables_direct(y_start, x_start, canonical_board)
        reachables_jumping = self.b.get_reachables_jump(y_start, x_start, canonical_board)
        for (_, _, direction) in reachables_direct:
            change_y, change_x = self.b.get_direction_pos(direction)
            y_end, x_end = y_start + change_y, x_start + change_x
            possible_board[y_end, x_end] = 1

        for (y_end, x_end) in reachables_jumping:
            possible_board[y_end, x_end] = 1

        possible_board = self.getCanonicalForm(possible_board, player_revert)

        return possible_board
