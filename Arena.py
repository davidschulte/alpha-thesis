import numpy as np
import itertools
from pytorch_classification.utils import Bar, AverageMeter
import time

class Arena():
    """
    An Arena class where any 2 agents can be pit against each other.
    """
    def __init__(self, player1, player2, game, display=None):
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
        self.player1 = player1
        self.player2 = player2
        self.game = game
        self.display = display

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
            players = [self.player2, self.player2, self.player2]
            players[turn_lonely] = self.player1
        else:
            players = [self.player1, self.player1, self.player1]
            players[turn_lonely] = self.player2

        curPlayer = 1
        board = self.game.getInitBoard()
        it = 0
        while np.count_nonzero(self.game.getGameEnded(board, False)) < 2:
            it += 1
            if verbose:
                assert(self.display)
                print("Turn ", str(it), "Player ", str(curPlayer))
                self.display(board)
            action = players[curPlayer-1](self.game.getCanonicalForm(board, curPlayer))

            valids = self.game.getValidMoves(self.game.getCanonicalForm(board, curPlayer),1)

            if valids[action]==0:
                print(action)
                assert valids[action] >0
            board, curPlayer = self.game.getNextState(board, curPlayer, action)
        if verbose:
            assert(self.display)
            print("Game over: Turn ", str(it), "Result ", str(self.game.getGameEnded(board, 1)))
            self.display(board)
        return self.game.getGameEnded(board, False)

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

        num = int(num/6)
        # oneWon = 0
        # twoWon = 0
        # draws = 0
        scores = [0, 0]
        for lonely_player in [1,2]:
            for lonely_turn in range(3):
                for _ in range(num):
                    self.game.reset_board()
                    gameResult = self.playGame(lonely_player, lonely_turn, verbose=verbose)
                    for t in range(3):
                        # if bool(p == lonely_player) != bool(t != lonely_turn):
                        if t == lonely_turn:
                            scores[lonely_player-1] += gameResult[t]
                        else:
                            scores[-lonely_player+1] += gameResult[t]

                    # bookkeeping + plot progress
                    eps += 1
                    eps_time.update(time.time() - end)
                    end = time.time()
                    bar.suffix  = '({eps}/{maxeps}) Eps Time: {et:.3f}s | Total: {total:} | ETA: {eta:}'.format(eps=eps+1, maxeps=maxeps, et=eps_time.avg,
                                                                                                               total=bar.elapsed_td, eta=bar.eta_td)
                    bar.next()



        bar.finish()

        return scores
