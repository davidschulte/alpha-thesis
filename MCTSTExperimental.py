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
        self.Next_State = {}

        self.Es = {}  # stores game.getGameEnded ended for board s
        self.Vs = {}  # stores game.getValidMoves for board s

    def getActionProb(self, canonicalBoard, temp=1):
        """
        This function performs numMCTSSims simulations of MCTS starting from
        canonicalBoard.

        Returns:
            probs: a policy vector where the probability of the ith action is
                   proportional to Nsa[(s,a)]**(1./temp)
        """
        for i in range(self.args.numMCTSSims):
            self.search(canonicalBoard, 0)

        s = self.game.stringRepresentation(canonicalBoard)
        counts = [self.Nsa[(s, a)] if (s, a) in self.Nsa else 0 for a in range(self.game.getActionSize())]
        # test = sum(counts)

        if temp == 0:
            bestA = np.argmax(counts)
            probs = [0] * len(counts)
            probs[bestA] = 1
            return probs

        # counts = [x**(1./temp) for x in counts]
        if sum(counts) == 0:
            print("Random Move!")
            counts = self.game.getValidMoves(canonicalBoard, 1)

        probs = [x / float(sum(counts)) for x in counts]
        # test = sum(probs)
        return probs

    def search(self, canonicalBoard, depth):
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

        s = self.game.stringRepresentation(canonicalBoard)

        scores = self.game.getGameEnded(canonicalBoard, True).astype('float16')

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
            return np.array([scores[2], scores[0], scores[1]])

        if depth == 0:
            self.Loop = {s: 1}
        else:
            if s in self.Visited:
                return np.array([0, scores[0], scores[1]])

            if s in self.Loop:
                print("Pevented Loop! Depth: " + str(depth))
                return np.array([0, scores[0], scores[1]])
            else:
                self.Loop[s] = 1

        if s not in self.Ps or depth > DEPTHMAX:
            if depth > DEPTHMAX:
                print("CUT LEAF")
            # leaf node
            self.Ps[s], scores_nn = self.nnet.predict(canonicalBoard)
            valids = self.game.getValidMoves(canonicalBoard, 1)
            self.Ps[s] = self.Ps[s] * valids  # masking invalid moves
            sum_Ps_s = np.sum(self.Ps[s])
            if sum_Ps_s > 0:
                self.Ps[s] /= sum_Ps_s  # renormalize
            else:
                # if all valid moves were masked make all valid moves equally probable

                # NB! All valid moves may be masked if either your NNet architecture is insufficient or you've get overfitting or something else.
                # If you have got dozens or hundreds of these messages you should pay attention to your NNet and/or training process.   
                print("All valid moves were masked, do workaround.")
                self.Ps[s] = self.Ps[s] + valids
                self.Ps[s] /= np.sum(self.Ps[s])

            self.Vs[s] = valids
            self.Ns[s] = 0

            for i in range(3):
                if scores[i] == 0:
                    scores[i] = scores_nn[i]

            return np.array([scores[2], scores[0], scores[1]])

        valids = self.Vs[s]
        cur_best = -float('inf')
        best_act = -1

        # pick the action with the highest upper confidence bound
        for a in range(self.game.getActionSize()):
            if valids[a]:
                if (s, a) in self.Qsa:
                    qsa = self.Qsa[(s, a)]
                    pssa = self.Ps[s][a]
                    ns = self.Ns[s]
                    nsa = self.Nsa[(s, a)]
                    exploitation = self.Qsa[(s, a)]
                    exploration = self.Ps[s][a] * math.sqrt(self.Ns[s]) / (1 + self.Ns[self.Next_State[s,a]])
                    u = self.Qsa[(s, a)] + self.args.cpuct * self.Ps[s][a] * math.sqrt(self.Ns[s]) / (
                                1 + self.Ns[self.Next_State[s,a]])
                    u = u
                else:
                    pssa = self.Ps[s][a]
                    nsa = self.Ns[s]
                    exploitation = 0
                    exploration = self.Ps[s][a] * math.sqrt(self.Ns[s] + EPS)
                    u = self.args.cpuct * self.Ps[s][a] * math.sqrt(self.Ns[s] + EPS)  # Q = 0 ?
                    u = u

                if u > cur_best:
                    cur_best = u
                    best_act = a

        a = best_act
        next_s, next_player = self.game.getNextState(canonicalBoard, 1, a)
        next_s = self.game.getCanonicalForm(next_s, next_player)
        next_s_string = self.game.stringRepresentation(next_s)
        self.Next_State[(s,a)] = next_s_string

        scores = self.search(next_s, depth + 1)

        if (s, a) in self.Qsa:
            self.Qsa[(s, a)] = (self.Nsa[(s, a)] * self.Qsa[(s, a)] + scores[0]) / (self.Nsa[(s, a)] + 1)
            self.Nsa[(s, a)] += 1

        else:
            self.Qsa[(s, a)] = scores[0]
            self.Nsa[(s, a)] = 1

        self.Ns[s] += 1

        return np.array([scores[2], scores[0], scores[1]])

    def reset(self):
        self.Qsa = {}  # stores Q values for s,a (as defined in the paper)
        self.Nsa = {}  # stores #times edge s,a was visited
        self.Ns = {}  # stores #times board s was visited
        self.Ps = {}  # stores initial policy (returned by neural net)

        self.Es = {}  # stores game.getGameEnded ended for board s
        self.Vs = {}  # stores game.getValidMoves for board s
        self.C = {}
