class VeryGreedyActor:

    def __init__(self, game):
        self.game = game

    def getActionProb(self, canonicalBoard, random):

        valids = self.game.getValidMoves(canonicalBoard, 1)

        if random:
            probs = valids
        else:
            probs = [0] * self.game.getActionSize()

            for move in range(len(probs)):
                if valids[move] == 1:
                    y_start, _, y_end, _ = self.game.get_board().decode_move(move)
                    progress = y_start - y_end
                    if progress > 0:
                        probs[move] = progress
                    elif progress == 0:
                        probs[move] = 0.5
                    else:
                        probs[move] = 0.2

        sum_probs = sum(probs)
        probs = [x/float(sum_probs) for x in probs]

        return probs
