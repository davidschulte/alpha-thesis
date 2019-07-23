import numpy as np
import itertools
from pytorch_classification.utils import Bar, AverageMeter
from MCTST import MCTS
import time
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

class Arena():
    """
    An Arena class where any 2 agents can be pit against each other.
    """
    def __init__(self, nnet1, nnet2, game, args, display=None):
        """
        Input:
            player 1,2: two functions that takes board as input, return action
            game: Game object
            display: a function that takes board as input and prints it (e.g.
                     display in othello/OthelloGame). Is necessary for verbose
                     mode.

        see othello/OthelloPlayers.py for an example. See pit.py for pitting
        human players/other baselines with each other.
        """
        self.nnet1 = nnet1
        self.nnet2 = nnet2
        # self.player1 = player1
        # self.player2 = player2
        self.game = game
        self.args = args
        self.display = display

    @profile
    def playGame(self, lonely_player, turn_lonely, verbose=False):
        """
        Executes one episode of a game.

        Returns:
            either
                winner: player who won the game (1 if player1, -1 if player2)
            or
                draw result returned from the game that is neither 1, -1, nor 0.
        """
        if lonely_player == 1:
            players = [MCTS(self.game, self.nnet2, self.args), MCTS(self.game, self.nnet2, self.args), MCTS(self.game, self.nnet2, self.args)]
            players[turn_lonely] = MCTS(self.game, self.nnet1, self.args)
        else:
            players = [MCTS(self.game, self.nnet1, self.args), MCTS(self.game, self.nnet1, self.args), MCTS(self.game, self.nnet1, self.args)]
            players[turn_lonely] = MCTS(self.game, self.nnet2, self.args)

        curPlayer = 1
        board = self.game.getInitBoard()
        it = 0

        scores = np.array([0, 0, 0])
        self.game.reset_board()
        while np.count_nonzero(scores) < 2:
            it += 1

            if it % 100 == 0:
                print(it)

                if it % 1000 == 0:
                    print(board)

            if verbose:
                assert(self.display)
                print("Turn ", str(it), "Player ", str(curPlayer))
                self.display(board)

            if scores[curPlayer-1] == 0:
                canonicalBoard = self.game.getCanonicalForm(board, curPlayer)
                s = self.game.stringRepresentation(canonicalBoard)
                players[curPlayer-1].Visited.append(s)
                pi = players[curPlayer-1].getActionProb(canonicalBoard, temp=1)
                action = np.random.choice(self.game.getActionSize(), p=pi)
            else:
                action = self.game.getActionSize()-1

            valids = self.game.getValidMoves(self.game.getCanonicalForm(board, curPlayer),1)

            if valids[action]==0:
                print(action)
                assert valids[action] >0
            board, curPlayer = self.game.getNextState(board, curPlayer, action)

            scores = self.game.getGameEnded(board, False)

        if verbose:
            assert(self.display)
            print("Game over: Turn ", str(it), "Result ", str(self.game.getGameEnded(board, 1)))
            self.display(board)

        return scores

    def playGames(self, num, verbose=False):
        """
        Plays num games in which player1 starts num/2 games and player2 starts
        num/2 games.

        Returns:
            oneWon: games won by player1
            twoWon: games won by player2
            draws:  games won by nobody
        """
        eps_time = AverageMeter()
        bar = Bar('Arena.playGames', max=num)
        end = time.time()
        eps = 0
        maxeps = int(num)

        num = int(num / 6)
        # oneWon = 0
        # twoWon = 0
        # draws = 0
        scores = [0, 0]
        for lonely_player in [1, 2]:
            for lonely_turn in range(3):
                for _ in range(num):
                    self.game.reset_board()
                    print("New Game")
                    print("Lonely Player: " + str(lonely_player))
                    print("Lonely Turn: " + str(lonely_turn+1))
                    gameResult = self.playGame(lonely_player, lonely_turn, verbose=verbose)
                    print("RESULTS")
                    print(gameResult)
                    for t in range(3):
                        # if bool(p == lonely_player) != bool(t != lonely_turn):
                        if t == lonely_turn:
                            scores[lonely_player-1] += gameResult[t]
                        else:
                            scores[2-lonely_player] += gameResult[t]

                    print("CUMMULATED RESULTS:")
                    print(scores)
                    # bookkeeping + plot progress
                    eps += 1
                    eps_time.update(time.time() - end)
                    end = time.time()
                    bar.suffix  = '({eps}/{maxeps}) Eps Time: {et:.3f}s | Total: {total:} | ETA: {eta:}'.format(eps=eps+1, maxeps=maxeps, et=eps_time.avg,
                                                                                                               total=bar.elapsed_td, eta=bar.eta_td)
                    bar.next()



        bar.finish()

        return scores
