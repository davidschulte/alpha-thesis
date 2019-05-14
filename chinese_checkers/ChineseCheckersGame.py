from Game import Game
from .ChineseCheckersLogic import Board
import numpy as np

class ChineseCheckersGame(Game):

    def __init__(self):
        Game.__init__(self)

    def getInitBoard(self):
        """
        Returns:
            startBo
            ard: a representation of the board (ideally this is the form
                        that will be the input to your neural network)
        """
        b = Board()
        return b.get_start()

    def getBoardSize(self):
        """
        Returns:
            (x,y): a tuple of board dimensions
        """
        return 17,13

    def getActionSize(self):
        """
        Returns:
            actionSize: number of all possible actions
        """
        return 121*121+1

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
            return board, board.get_next_player(player)
        b = Board()
        b.np_pieces = np.copy(board)
        move = (int(action/121), action%121)
        y_start, x_start = board.decode_coordinates(move[0])
        y_end, x_end = board.decode_coordinates(move[1])
        b.move((y_start, x_start), (y_end, x_end), player)

        return b.np_pieces, board.get_next_player(player)

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
        if board.get_done(player):
            valids[-1] = 1
            return valids
        for y_start, x_start, y_end, x_end in board.get_legal_moves(player):
            start = board.encode_coordinates(y_start, x_start)
            end = board.encode_coordinates(y_end, x_end)
            valids[start*121+end] = 1
        return valids


    def getGameEnded(self, board, player):
        """
        Input:
            board: current board
            player: current player (1 or -1)

        Returns:
            r: 0 if game has not ended. 1 if player won, -1 if player lost,
               small non-zero value for draw.

        """
        ended, scores = board.get_win_state(player)
        if(ended):
            return scores
        return None

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
        b = Board()
        canonical_board = b.encode_board(board)
        old_board = np.copy(canonical_board)
        shift = player - 1
        for p in [1,2,3]:
            canonical_board[old_board == p] = (p + shift) % 3

        return canonical_board

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
        return []

    def stringRepresentation(self, board):
        """
        Input:
            board: current board

        Returns:
            boardString: a quick conversion of board to a string format.
                         Required by MCTS for hashing.
        """
        b = Board()
        return b.encode_board(board).tostring()
