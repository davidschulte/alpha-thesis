import numpy as np
from random import randrange

class GreedyActor:

    def __init__(self, game):
        self. game = game
        self.y, self.x = self.game.getBoardSize()

    def predict(self, board, cur_player):
        v = np.array([0, 0, 0]).astype('float16')

        for player in [1, 2, 3]:
            canonicalBoard = self.game.getCanonicalForm(board, player)
            score = 0
            for row in range(self.y):
                for column in range(self.x):
                    if canonicalBoard[row, column] == 1:
                        score += (12 - row) / 100
            v[player-1] = score

        pi = [0] * self.game.getActionSize()
        for i in range(self.game.getActionSize()):
            pi[i] = randrange(95,100)
        pi = np.array(pi)
        pi = pi / sum(pi)

        if cur_player == 2:
            v = np.array([v[2], v[0], v[1]])
        elif cur_player == 3:
            v = np.array([v[1], v[2], v[0]])

        return pi, np.array(v)
