from Game import Game
from .ChineseCheckersLogic import Board
import numpy as np


class ChineseCheckersGame(Game):

    def __init__(self):
        Game.__init__(self)
        self.b = Board()

    def getInitBoard(self):
        """
        Returns:
            startBo
            ard: a representation of the board (ideally this is the form
                        that will be the input to your neural network)
        """
        return self.b.get_start()

    def getBoardSize(self):
        """
        Returns:
            (x,y): a tuple of board dimensions
        """
        return 17, 17

    def getActionSize(self):
        """
        Returns:
            actionSize: number of all possible actions
        """
        return 81*6+25*25+2*20*20+16*16+1

    def getNextState(self, board, player, action):
        """
        Input:
            board: current board
            player: current player (1 or -1)
            action: action taken by current player

        Returns:
            nextBoard: board after applying action
            nextPlayer: player who plays in the next turn (should be -player)
        """
        # if player takes action on board, return next (board,player)
        # action must be a valid move
        if action == self.getActionSize() - 1:
            return board, self.b.get_next_player(player)
        y_start, x_start, y_end, x_end = self.b.decode_move(action)

        self.b.np_pieces = np.copy(board)
        if player != 1:
            self.b.np_pieces = self.b.rotate_board(self.b.np_pieces, 1, player)

        self.b.move((y_start, x_start), (y_end, x_end), player)

        if player != 1:
            self.b.np_pieces = self.b.rotate_board(self.b.np_pieces, player, 1)

        return self.b.np_pieces, self.b.get_next_player(player)

    def getValidMoves(self, board, player):
        """
        Input:
            board: current board
            player: current player

        Returns:
            validMoves: a binary vector of length self.getActionSize(), 1 for
                        moves that are valid from the current board and player,
                        0 for invalid moves
        """
        valids = [0]*self.getActionSize()
        if self.b.get_done(board, player):
            valids[-1] = 1
            return valids

        legal_moves_direct, legal_moves_jumping = self.b.get_legal_moves(board, player)

        for y_start, x_start, direction in legal_moves_direct:
            valids[self.b.encode_move_direct(y_start, x_start, direction)] = 1

        for y_start, x_start, y_end, x_end in legal_moves_jumping:
            valids[self.b.encode_move_jumping(y_start, x_start, y_end, x_end)] = 1

        return valids


    def getGameEnded(self, board, temporary):
        """
        Input:
            board: current board
            player: current player (1 or -1)

        Returns:
            r: 0 if game has not ended. 1 if player won, -1 if player lost,
               small non-zero value for draw.

        """
        ended, scores = self.b.get_win_state(board, temporary)

        return np.copy(scores)

    def getCanonicalForm(self, board, player):
        """
        Input:
            board: current board
            player: current player (1 or -1)

        Returns:
            canonicalBoard: returns canonical form of board. The canonical form
                            should be independent of player. For e.g. in chess,
                            the canonical form can be chosen to be from the pov
                            of white. When the player is white, we can return
                            board as is. When the player is black, we can invert
                            the colors and return the board.
        """
        # b = Board()
        # canonical_board = b.encode_board(board)
        # old_board = np.copy(canonical_board)
        # shift = player - 1
        # for p in [1,2,3]:
        #     canonical_board[old_board == p] = (p + shift) % 3
        #
        # return canonical_board
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



    def getSymmetries(self, board, pi):
        """
        Input:
            board: current board
            pi: policy vector of size self.getActionSize()

        Returns:
            symmForms: a list of [(board,pi)] where each tuple is a symmetrical
                       form of the board and the corresponding pi vector. This
                       is used when training the neural network from examples.
        """
        return [(board, pi)]

    def stringRepresentation(self, board):
        """
        Input:
            board: current board

        Returns:
            boardString: a quick conversion of board to a string format.
                         Required by MCTS for hashing.
        """
        return board.tostring()

    def reset_board(self):
        self.b = Board()
