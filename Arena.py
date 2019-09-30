import numpy as np
from pytorch_classification.utils import Bar, AverageMeter
import time


class Arena:
    """
    An Arena class where any 2 agents can be pit against each other.
    """
    def __init__(self, mcts1, mcts2, game, args):
        """

        :param mcts1: agent 1
        :param mcts2: agent 2
        :param game: game
        :param args: arguments
        """
        self.mcts1 = mcts1
        self.mcts2 = mcts2
        self.game = game
        self.args = args

    def playGame(self, lonely_player, turn_lonely):
        """
        plays one game and returns the scores
        :param lonely_player: denotes which agent will play with one instance
        :param turn_lonely: denotes the player that the single instance agent is assigned to
        """
        if lonely_player == 1:
            players = [self.mcts2, self.mcts2, self.mcts2]
            players[turn_lonely] = self.mcts1
        else:
            players = [self.mcts1, self.mcts1, self.mcts1]
            players[turn_lonely] = self.mcts2

        cur_player = 1
        board = self.game.getInitBoard()
        it = 1

        scores = np.array([0, 0, 0])
        self.game.reset_logic()
        while np.count_nonzero(scores) < 2 and it < self.args.max_steps:

            if it % 100 == 0:
                print(it)
                print(board)

            if scores[cur_player-1] == 0:
                pi = players[cur_player-1].get_action_prob(board, cur_player, it > 15)
                action = np.random.choice(self.game.getActionSize(), p=pi)
                it += 1
            else:
                action = self.game.getActionSize()-1

            valids = self.game.getValidMoves(board, cur_player)

            if valids[action]==0:
                print(action)
                assert valids[action] >0
            board, cur_player = self.game.getNextState(board, cur_player, action)

            scores = self.game.getGameEnded(board, False)

        print(board)

        return scores

    def playGames(self, num):
        """
        plays a number of games
        :param num: number of games, has to be divisible by 6 for fair games
        :return: the summed scores of each agent
        """
        eps_time = AverageMeter()
        bar = Bar('Arena.playGames', max=num)
        end = time.time()
        eps = 0
        maxeps = int(num)

        max_scores = num * 4

        num = int(num / 6)
        # oneWon = 0
        # twoWon = 0
        # draws = 0
        scores = [0, 0]
        for lonely_player in [1, 2]:
            for lonely_turn in range(3):
                for _ in range(num):
                    if scores[0] < self.args.updateThreshold * max_scores and scores[1] < self.args.updateThreshold * max_scores:
                        self.game.reset_logic()
                        print("New Game")
                        print("Lonely Player: " + str(lonely_player))
                        print("Lonely Turn: " + str(lonely_turn+1))
                        gameResult = self.playGame(lonely_player, lonely_turn)
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

