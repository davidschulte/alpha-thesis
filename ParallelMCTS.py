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
        self.Visited = []
        self.Es = {}  # stores game.getGameEnded ended for board s
        self.Vs = {}  # stores game.getValidMoves for board s
        self.B = {}
        self.iter = 0
        self.counts = [0]*game.getActionSize()
        self.trace = []

    def search(self, board, player, depth):
        while True:
            s = self.game.stringRepresentation(board)
            scores = self.game.getGameEnded(board, True).astype('float16')

            if s not in self.Es:
                self.Es[s] = np.copy(scores)
            if np.count_nonzero(self.Es[s]) > 1:
                # terminal node
                self.back_propagate(self.Es[s])
                # print("GAME OVER")
                return None

            if depth == 0:
                self.Loop = [(s, player)]
            else:
                if s in self.Visited:
                    self.back_propagate(self.Es[s])
                    print("VISITED")
                    return None
                if (s, player) in self.Loop:
                    print("LOOP")
                    self.back_propagate(self.Es[s])
                    return None
                else:
                    self.Loop.append((s, player))

            if (s, player) not in self.Ps:
                canonicalBoard = self.game.getCanonicalForm(board, player)
                valids = self.game.getValidMoves(canonicalBoard, 1)
                self.Vs[(s, player)] = valids
                self.trace.append([s, player])

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
                        u = self.Qsa[(s, a, player)] + self.args.cpuct * self.Ps[(s, player)][a] * math.sqrt(self.Ns[s, player]) / (
                                    1 + self.Nsa[(s, a, player)])
                    else:
                        u = self.args.cpuct * self.Ps[(s, player)][a] * math.sqrt(self.Ns[(s, player)] + EPS)  # Q = 0 ?

                    if u > cur_best:
                        cur_best = u
                        best_act = a

            a = best_act

            if a < 0:
                print(a)
                print(valids)
                print(board)
                print(player)
                print(depth)

            self.trace.append([s, a, player])
            # if (s, a, player) in self.B:
            #     board = self.B[(s, a, player)]
            #     player = self.game.get_next_player(player)
            # else:
            #     board, next_player = self.game.getNextState(board, player, a)
            #     self.B[(s, a, player)] = board
            #     player = next_player

            board, player = self.game.getNextState(board, player, a)
            depth += 1



    # def back_propagate(self):


    def reset(self):
        self.Qsa = {}  # stores Q values for s,a (as defined in the paper)
        self.Nsa = {}  # stores #times edge s,a was visited
        self.Ns = {}  # stores #times board s was visited
        self.Ps = {}  # stores initial policy (returned by neural net)

        self.Es = {}  # stores game.getGameEnded ended for board s
        self.Vs = {}  # stores game.getValidMoves for board s
        self.iter = 0
        self.B = {}

    def get_counts(self, board, player, best=False):
        if self.iter < self.args.numMCTSSims:
            return None
        s = self.game.stringRepresentation(board)
        counts = [self.Nsa[(s, a, player)] if (s, a, player) in self.Nsa else 0 for a in range(self.game.getActionSize())]

        if best:
            bestA = np.argmax(counts)
            probs = [0] * len(counts)
            probs[bestA] = 1
            return probs

        sum_counts = sum(counts)
        if sum_counts == 0:
            print("SUM COUNTS 0!")
            counts = self.game.getValidMoves(board, player)
            sum_counts = sum(counts)

        probs = [x / float(sum_counts) for x in counts]

        self.reset()

        return probs

    def get_done(self):
        return self.iter == self.args.numMCTSSims

    def update_predictions(self, pi, v):
        s, player = self.trace.pop()
        self.Ps[(s, player)] = pi

        if player == 1:
            scores_nn = v
        elif player == 2:
            scores_nn = np.array([v[2], v[0], v[1]])
        else:
            scores_nn = np.array([v[1], v[2], v[0]])

        sum_Ps_s = np.sum(self.Ps[(s, player)])
        if sum_Ps_s > 0:
            self.Ps[(s, player)] /= sum_Ps_s  # renormalize
        else:
            print("All valid moves were masked, do workaround.")
            self.Ps[(s, player)] = self.Ps[(s, player)] + self.Vs[(s, player)]
            self.Ps[(s, player)] /= np.sum(self.Ps[(s, player)])

        self.Ns[(s, player)] = 0

        scores = np.copy(self.Es[s])
        for i in range(3):
            if scores[i] == 0:
                scores[i] = scores_nn[i]

        self.back_propagate(scores)

    def back_propagate(self, scores):
        self.iter += 1
        while self.trace:
            s, a, player = self.trace.pop()
            if (s, a, player) in self.Qsa:
                self.Qsa[(s, a, player)] = (self.Nsa[(s, a, player)] * self.Qsa[(s, a, player)] + scores[
                    player - 1]) / (self.Nsa[(s, a, player)] + 1)
                self.Nsa[(s, a, player)] += 1
            else:
                self.Qsa[(s, a, player)] = scores[player - 1]
                self.Nsa[(s, a, player)] = 1
            self.Ns[(s, player)] += 1
