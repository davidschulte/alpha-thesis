from .TinyGUI import GUI
from .TinyChineseCheckersGame import ChineseCheckersGame
import numpy as np

class Evaluator:
    def __init__(self, player1, player2, player3, show=False):
        self.players = [player1, player2, player3]
        self.show = show
        if None in self.players:
            self.show = True
        self.gui = None
        self.game = ChineseCheckersGame()
        if show:
            self.gui = GUI(1)
        self.game = ChineseCheckersGame()

    def play_game(self):
        board = self.game.getInitBoard()
        curPlayer = 1
        iter_step = 1
        scores = [0, 0, 0]
        self.game.reset_board()
        if self.show:
            self.gui.draw_board(board, iter_step)

        while np.count_nonzero(scores) < 2:
            if scores[curPlayer-1] == 0:
                if self.players[curPlayer-1] is None:
                    a = self.gui.get_action(board)
                else:
                    pi = self.players[curPlayer-1].getActionProb(board, curPlayer)
                    a = np.random.choice(len(pi), p=pi)

                board, curPlayer = self.game.getNextState(board, curPlayer, a)
                if self.show:
                    self.gui.draw_board(board, iter_step)
                iter_step += 1
            else:
                a = self.game.getActionSize()-1
                board, curPlayer = self.game.getNextState(board, curPlayer, a)

            scores = self.game.getGameEnded(board, False)
        print(scores)
