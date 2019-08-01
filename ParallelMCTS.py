import math
import numpy as np
import cProfile, pstats, io

def profile(fnc):
    """A decorator that uses cProfile to profile a function"""

    def inner(*args, **kwargs):
        pr = cProfile.Profile()
        pr.enable()
        retval = fnc(*args, **kwargs)
        pr.disable()
        s = io.StringIO()
        sortby = 'cumulative'
        ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
        ps.print_stats()
        print(s.getvalue())
        return retval

    return inner


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
        self.Loop = []
        self.Loop_Scout = []
        self.Visited = []
        self.Es = {}  # stores game.getGameEnded ended for board s
        self.Vs = {}  # stores game.getValidMoves for board s
        self.iter = 0
        self.counts = [0]*game.getActionSize()
        self.prediction_pi = None
        self.prediction_v = None

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
            self.search(board, player, 0)

        if temp == 0:
            bestA = np.argmax(counts)
            probs = [0] * len(counts)
            probs[bestA] = 1
            return probs

        sum_counts = sum(counts)
        if sum_counts == 0:
            print(board)
            print("Random Move by " + str(player) + "!")
            counts = self.game.getValidMoves(board, player)

        probs = [x / float(sum_counts) for x in counts]
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
        # if depth > self.max_depth:
        #     self.max_depth = depth

        s = self.game.stringRepresentation(board)

        scores = self.game.getGameEnded(board, True).astype('float16')

        canonicalBoard = self.game.getCanonicalForm(board, player)

        if s not in self.Es:
            self.Es[s] = np.copy(scores)
        if np.count_nonzero(self.Es[s]) == 2:
            # terminal node
            return scores

        if depth == 0:
            self.Loop = [(s, player)]
        else:
            if s in self.Visited:
                # print("VISITED")
                scores[self.game.get_board().get_previous_player(player) - 1] = 0
                return scores
            if (s, player) in self.Loop:
                print("Prevented Loop! Depth: " + str(depth))
                scores[self.game.get_board().get_previous_player(player)-1] = 0
                return scores
            else:
                self.Loop.append((s, player))



        if (s, player) not in self.Ps or depth > DEPTHMAX:
            if depth > DEPTHMAX:

                print("CUT LEAF")
            # leaf node
            valids = self.Vs[(s, player)]

            self.Ps[(s, player)], scores_nn = self.prediction_pi, self.prediction_v
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
                    # qsa = self.Qsa[(s, a, player)]
                    # pssa = self.Ps[(s, player)][a]
                    # ns = self.Ns[(s, player)]
                    # nsa = self.Nsa[(s, a, player)]
                    # exploitation = self.Qsa[(s, a, player)]
                    # exploration = self.Ps[(s, player)][a] * math.sqrt(self.Ns[s, player]) / (1 + self.Nsa[(s, a, player)])
                    u = self.Qsa[(s, a, player)] + self.args.cpuct * self.Ps[(s, player)][a] * math.sqrt(self.Ns[s, player]) / (
                                1 + self.Nsa[(s, a, player)])
                    # u = u
                else:
                    # pssa = self.Ps[(s, player)][a]
                    # nsa = self.Ns[(s, player)]
                    # exploitation = 0
                    # exploration = self.Ps[(s, player)][a] * math.sqrt(self.Ns[(s, player)] + EPS)
                    u = self.args.cpuct * self.Ps[(s, player)][a] * math.sqrt(self.Ns[(s, player)] + EPS)  # Q = 0 ?
                    # u = u

                if u > cur_best:
                    # if exploitation > 0:
                    #     expl = True
                    # else:
                    #     expl = False
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
            self.Qsa[(s, a, player)] = (self.Nsa[(s, a, player)] * self.Qsa[(s, a, player)] + scores[player-1]) / (self.Nsa[(s, a, player)] + 1)
            self.Nsa[(s, a, player)] += 1

        else:
            self.Qsa[(s, a, player)] = scores[player-1]
            self.Nsa[(s, a, player)] = 1

        self.Ns[(s, player)] += 1

        return scores


    def get_next_leaf(self, board, player, depth):
        s = self.game.stringRepresentation(board)

        scores = self.game.getGameEnded(board, True).astype('float16')

        canonicalBoard = self.game.getCanonicalForm(board, player)

        if s not in self.Es:
            self.Es[s] = np.copy(scores)
        if np.count_nonzero(self.Es[s]) == 2:
            # terminal node
            return None

        if depth == 0:
            self.Loop_Scout = [(s, player)]
        else:
            if s in self.Visited:
                return None
            if (s, player) in self.Loop_Scout:
                return None
            else:
                self.Loop.append((s, player))

        if (s, player) not in self.Ps:
            valids = self.game.getValidMoves(canonicalBoard, 1)
            self.Vs[(s, player)] = valids

            return canonicalBoard
            # self.Ps[(s, player)], scores_nn = self.nnet.predict(canonicalBoard, player)
            # return scores

        valids = self.Vs[(s, player)]
        cur_best = -float('inf')
        best_act = -1

        # pick the action with the highest upper confidence bound
        for a in range(self.game.getActionSize()):
            if valids[a]:
                if (s, a, player) in self.Qsa:
                    # qsa = self.Qsa[(s, a, player)]
                    # pssa = self.Ps[(s, player)][a]
                    # ns = self.Ns[(s, player)]
                    # nsa = self.Nsa[(s, a, player)]
                    # exploitation = self.Qsa[(s, a, player)]
                    # exploration = self.Ps[(s, player)][a] * math.sqrt(self.Ns[s, player]) / (1 + self.Nsa[(s, a, player)])
                    u = self.Qsa[(s, a, player)] + self.args.cpuct * self.Ps[(s, player)][a] * math.sqrt(self.Ns[s, player]) / (
                                1 + self.Nsa[(s, a, player)])
                    # u = u
                else:
                    # pssa = self.Ps[(s, player)][a]
                    # nsa = self.Ns[(s, player)]
                    # exploitation = 0
                    # exploration = self.Ps[(s, player)][a] * math.sqrt(self.Ns[(s, player)] + EPS)
                    u = self.args.cpuct * self.Ps[(s, player)][a] * math.sqrt(self.Ns[(s, player)] + EPS)  # Q = 0 ?
                    # u = u

                if u > cur_best:
                    # if exploitation > 0:
                    #     expl = True
                    # else:
                    #     expl = False
                    cur_best = u
                    best_act = a

        a = best_act
        next_s, next_player = self.game.getNextState(board, player, a)

        state = self.get_next_leaf(next_s, next_player, depth + 1)

        return state

    # def back_propagate(self):


    def reset(self):
        self.Qsa = {}  # stores Q values for s,a (as defined in the paper)
        self.Nsa = {}  # stores #times edge s,a was visited
        self.Ns = {}  # stores #times board s was visited
        self.Ps = {}  # stores initial policy (returned by neural net)

        self.Es = {}  # stores game.getGameEnded ended for board s
        self.Vs = {}  # stores game.getValidMoves for board s
        self.iter = 0

    def get_counts(self, board, player):
        if self.iter < self.args.numMCTSSims:
            return None
        s = self.game.stringRepresentation(board)
        counts = [self.Nsa[(s, a, player)] if (s, a, player) in self.Nsa else 0 for a in range(self.game.getActionSize())]
        sum_counts = sum(counts)
        if sum_counts == 0:
            print("SUM COUNTS 0!")
            canonical_board = self.game.getCanonicalForm(board, player)
            counts = self.game.getValidMoves(canonical_board, 1)
            sum_counts = sum(counts)

        probs = [x / float(sum_counts) for x in counts]
        return probs

    def add_iter(self):
        self.iter += 1

    def get_done(self):
        return self.iter == self.args.numMCTSSims

    def update_predictions(self, pi, v):
        self.prediction_pi = pi
        self.prediction_v = v
