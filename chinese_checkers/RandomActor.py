class RandomActor:

    def __init__(self, game):
        self.game = game

    def getActionProb(self, canonicalBoard):

        valids = self.game.getValidMoves(canonicalBoard, 1)

        probs = valids
        sum_probs = sum(probs)
        probs = [x/float(sum_probs) for x in probs]

        return probs
