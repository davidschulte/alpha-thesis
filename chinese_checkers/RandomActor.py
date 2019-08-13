class RandomActor:

    def __init__(self, game):
        self.game = game

    def getActionProb(self, board, player):

        valids = self.game.getValidMoves(board, player)

        probs = valids
        sum_probs = sum(probs)
        probs = [x/float(sum_probs) for x in probs]

        return probs
