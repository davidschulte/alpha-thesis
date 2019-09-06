import numpy as np

TEST = np.array([[4, 4, 4, 4, 4, 4, 1, 4, 4], #0
                 [4, 4, 4, 4, 4, 1, 1, 4, 4], #1
                 [4, 4, 0, 0, 1, 1, 2, 0, 3], #2
                 [4, 4, 0, 0, 0, 0, 0, 0, 4], #3
                 [4, 4, 3, 0, 0, 0, 1, 4, 4], #4
                 [4, 3, 3, 0, 0, 2, 2, 4, 4], #5
                 [3, 3, 0, 0, 2, 2, 2, 4, 4], #6
                 [4, 4, 0, 0, 4, 4, 4, 4, 4], #7
                 [4, 4, 0, 4, 4, 4, 4, 4, 4]]).astype('int8')#8

class Evaluator:
    def __init__(self, player1, player2, player3, game, gui, show=False):
        self.players = [player1, player2, player3]
        self.show = show
        self.game = game
        if None in self.players:
            self.show = True
        self.gui = gui

    def play_game(self, best_start, random_start):
        board = self.game. getInitBoard()
        # board = TEST
        curPlayer = 1
        # board = np.copy(TEST)
        # curPlayer = 2
        iter_step = 1
        scores = [0, 0, 0]
        self.game.reset_board()
        if self.show:
            self.gui.draw_board(board, curPlayer, False)

        while np.count_nonzero(scores) < 2:
            if scores[curPlayer-1] == 0:
                valids = self.game.getValidMoves(board, curPlayer)
                if self.players[curPlayer-1] is None and valids[-1] == 0:
                    a = self.gui.get_action(board, curPlayer)
                else:
                    if iter_step >= random_start:
                        pi = self.players[curPlayer-1].getActionProb(board, curPlayer, iter_step >= best_start)
                    else:
                        sum_valids = sum(valids)
                        pi = [x / float(sum_valids) for x in valids]
                    a = np.random.choice(len(pi), p=pi)

                board, curPlayer = self.game.getNextState(board, curPlayer, a)
                if self.show:
                    self.gui.draw_board(board, curPlayer, True)
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
