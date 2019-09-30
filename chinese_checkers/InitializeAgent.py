class InitializeAgent:

    def __init__(self, game):
        self.game = game

    def get_action_prob(self, board, player, best):
        """
        :param board:   current board
        :param player:  current player
        :param best:    best selection, does not matter because varied selection is always used
        :return:        action probability vector
        """
        probs = [0] * self.game.getActionSize()

        valids = self.game.getValidMoves(board, player)
        if valids[-1] == 1:
            probs[-1] = 1
            return probs

        for move in range(len(probs)):
            if valids[move] == 1:
                y_start, _, y_end, _ = self.game.get_board().decode_move(move)
                progress = y_start - y_end
                if progress > 0:
                    probs[move] = 4
                elif progress == 0:
                    probs[move] = 2
                else:
                    probs[move] = 1

        sum_probs = sum(probs)
        probs = [x/float(sum_probs) for x in probs]

        return probs
