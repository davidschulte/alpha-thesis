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

    def play_game(self, best_start):
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
                    pi = self.players[curPlayer-1].getActionProb(board, curPlayer, iter_step >= best_start)
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
        return scores, iter_step-1, self.check_second_rule(board, scores)

    def check_second_rule(self, board, scores):
        for p in range(3):
            if scores[p] > 0 and not self.game.get_board().get_done(board, p+1, True):
                return 1
        return 0
