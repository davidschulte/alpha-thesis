import math
import numpy as np

EPS = 1e-8
DEPTHMAX = 30


class MCTS():
    """
    This class handles the MCTS tree.
    """

    def __init__(self, game, nnet, args):
        self.game = game
        self.nnet = nnet
        self.args = args
        self.Qsa = {}  # stores Q values for s,a (as defined in the paper)
        self.Nsa = {}  # stores #times edge s,a was visited
        self.Ns = {}  # stores #times board s was visited
        self.Ps = {}  # stores initial policy (returned by neural net)
        # self.C = {}
        self.Loop = {}
        self.Visited = []

        self.Es = {}  # stores game.getGameEnded ended for board s
        self.Vs = {}  # stores game.getValidMoves for board s

    def getActionProb(self, board, player, temp=1):
        """
        This function performs numMCTSSims simulations of MCTS starting from
        canonicalBoard.

        Returns:
            probs: a policy vector where the probability of the ith action is
                   proportional to Nsa[(s,a)]**(1./temp)
        """
        self.reset()

        s = self.game.stringRepresentation(board)

        for i in range(self.args.numMCTSSims):
            counts = [self.Nsa[(s, a, player)] if (s, a, player) in self.Nsa else 0 for a in
                      range(self.game.getActionSize())]
            # test = sum(counts)
            # nonzeros = np.count_nonzero(counts)
            # if nonzeros < 3 and i == self.args.numMCTSSims - 2:
            #     print("DEBUG")

            self.search(board, player, 0)

        # print(nonzeros)
        counts = [self.Nsa[(s, a, player)] if (s, a, player) in self.Nsa else 0 for a in range(self.game.getActionSize())]
        # test = sum(counts)
        # if np.count_nonzero(np.array(counts)) < 3:
        #     print("DEBUG")

        if temp == 0:
            bestA = np.argmax(counts)
            probs = [0] * len(counts)
            probs[bestA] = 1
            return probs

        # counts = [x**(1./temp) for x in counts]
        if sum(counts) == 0:
            print(board)
            print("Random Move by " + str(player) + "!")
            counts = self.game.getValidMoves(board, player)

        probs = [x / float(sum(counts)) for x in counts]
        # test = sum(probs)
        return probs

    def search(self, board, player, depth):
        """
        This function performs one iteration of MCTS. It is recursively called
        till a leaf node is found. The action chosen at each node is one that
        has the maximum upper confidence bound as in the paper.

        Once a leaf node is found, the neural network is called to return an
        initial policy P and a value v for the state. This value is propogated
        up the search path. In case the leaf node is a terminal state, the
        outcome is propogated up the search path. The values of Ns, Nsa, Qsa are
        updated.

        NOTE: the return values are the negative of the value of the current
        state. This is done since v is in [-1,1] and if v is the value of a
        state for the current player, then its value is -v for the other player.

        Returns:
            v: the negative of the value of the current canonicalBoard
        """

        s = self.game.stringRepresentation(board)

        scores = self.game.getGameEnded(board, True).astype('float16')

        canonicalBoard = self.game.getCanonicalForm(board, player)

        # if s in self.C and scores[0] == 0:
        #     if self.C[s] >= 3:
        #         scores[scores == 0] = 1
        #         scores[2] = 0
        #         print("LOOP")
        #         print(canonicalBoard)
        #         return np.array([scores[2], scores[0], scores[1]])

        if s not in self.Es:
            self.Es[s] = np.copy(scores)
        if np.count_nonzero(self.Es[s]) == 2:
            # terminal node
            return scores

        if depth == 0:
            self.Loop = [(s, player)]
        else:
            if (s, player) in self.Loop:
                print("Prevented Loop! Depth: " + str(depth))
                scores[player-1] = 0
                return scores
            else:
                self.Loop.append((s, player))

        if s in self.Visited:
            print("VISITED")
            scores[player-1] = 0
            return scores

        if (s, player) not in self.Ps or depth > DEPTHMAX:
            if depth > DEPTHMAX:

                print("CUT LEAF")
            # leaf node
            self.Ps[(s, player)], scores_nn = self.nnet.predict(canonicalBoard, player)
            valids = self.game.getValidMoves(canonicalBoard, 1)
            self.Ps[(s, player)] = self.Ps[(s, player)] * valids  # masking invalid moves
            sum_Ps_s = np.sum(self.Ps[(s, player)])
            if sum_Ps_s > 0:
                self.Ps[(s, player)] /= sum_Ps_s  # renormalize
            else:
                # if all valid moves were masked make all valid moves equally probable

                # NB! All valid moves may be masked if either your NNet architecture is insufficient or you've get overfitting or something else.
                # If you have got dozens or hundreds of these messages you should pay attention to your NNet and/or training process.
                print("All valid moves were masked, do workaround.")
                self.Ps[(s, player)] = self.Ps[(s, player)] + valids
                self.Ps[(s, player)] /= np.sum(self.Ps[(s, player)])

            self.Vs[(s, player)] = valids
            self.Ns[(s, player)] = 0

            for i in range(3):
                if scores[i] == 0:
                    scores[i] = scores_nn[i]

            return scores

        valids = self.Vs[(s, player)]
        cur_best = -float('inf')
        best_act = -1

        # pick the action with the highest upper confidence bound
        for a in range(self.game.getActionSize()):
            if valids[a]:
                if (s, a, player) in self.Qsa:
                    qsa = self.Qsa[(s, a, player)]
                    pssa = self.Ps[(s, player)][a]
                    ns = self.Ns[(s, player)]
                    nsa = self.Nsa[(s, a, player)]
                    exploitation = self.Qsa[(s, a, player)]
                    exploration = self.Ps[(s, player)][a] * math.sqrt(self.Ns[s, player]) / (1 + self.Nsa[(s, a, player)])
                    u = self.Qsa[(s, a, player)] + self.args.cpuct * self.Ps[(s, player)][a] * math.sqrt(self.Ns[s, player]) / (
                                1 + self.Nsa[(s, a, player)])
                    u = u
                else:
                    pssa = self.Ps[(s, player)][a]
                    nsa = self.Ns[(s, player)]
                    exploitation = 0
                    exploration = self.Ps[(s, player)][a] * math.sqrt(self.Ns[(s, player)] + EPS)
                    u = self.args.cpuct * self.Ps[(s, player)][a] * math.sqrt(self.Ns[(s, player)] + EPS)  # Q = 0 ?
                    u = u

                if u > cur_best:
                    if exploitation > 0:
                        expl = True
                    else:
                        expl = False
                    cur_best = u
                    best_act = a

        # if expl:
        #     print("X")
        # else:
        #     print("O")

        a = best_act
        next_s, next_player = self.game.getNextState(board, player, a)
        # next_s = self.game.getCanonicalForm(next_s, next_player)

        scores = self.search(next_s, next_player, depth + 1)


        if (s, a, player) in self.Qsa:
            if scores is not None:
                self.Qsa[(s, a, player)] = (self.Nsa[(s, a, player)] * self.Qsa[(s, a, player)] + scores[player-1]) / (self.Nsa[(s, a, player)] + 1)
            self.Nsa[(s, a, player)] += 1

        else:
            if scores is not None:
                self.Qsa[(s, a, player)] = scores[player-1]
            self.Nsa[(s, a, player)] = 1

        self.Ns[(s, player)] += 1

        return scores

    def reset(self):
        self.Qsa = {}  # stores Q values for s,a (as defined in the paper)
        self.Nsa = {}  # stores #times edge s,a was visited
        self.Ns = {}  # stores #times board s was visited
        self.Ps = {}  # stores initial policy (returned by neural net)

        self.Es = {}  # stores game.getGameEnded ended for board s
        self.Vs = {}  # stores game.getValidMoves for board s
        self.C = {}
