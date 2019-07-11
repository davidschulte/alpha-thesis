import numpy as np

class GreedyActor:

    def __init__(self, game):
        self. game = game
        self.y, self.x = self.game.getBoardSize()

    def predict(self, board):
        v = np.array([0, 0, 0])

        for player in [1, 2, 3]:
            canonicalBoard = self.game.getCanonicalForm(board, player)
            score = 0
            for row in range(self.y):
                for column in range(self.x):
                    if canonicalBoard[row, column] == 1:
                        score += 18 - self.y
            v[player-1] = score

            pi = [1 / self.game.getActionSize()] * self.game.getActionSize()

            return pi, v
