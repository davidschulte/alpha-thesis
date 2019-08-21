import numpy as np
import pandas as pd


class Arena():
    """
    An Arena class where any 2 agents can be pit against each other.
    """
    def __init__(self, mcts1, mcts2, game, args):
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
        self.actors = [mcts1, mcts2]
        self.game = game
        self.args = args

    def play_game(self, playing_actors, start_best, start_not_random):

        curPlayer = 1
        board = self.game.getInitBoard()
        iter_step = 1

        scores = np.array([0, 0, 0])
        self.game.reset_board()
        while np.count_nonzero(scores) < 2 and iter_step < self.args.max_steps:

            if scores[curPlayer - 1] == 0:
                if iter_step >= start_not_random:
                    pi = self.actors[playing_actors[curPlayer - 1]-1].getActionProb(board, curPlayer, iter_step >= start_best)
                else:
                    valids = self.game.getValidMoves(board, curPlayer)
                    sum_valids = sum(valids)
                    pi = [x / float(sum_valids) for x in valids]
                action = np.random.choice(len(pi), p=pi)
                iter_step += 1
            else:
                action = self.game.getActionSize() - 1

            board, curPlayer = self.game.getNextState(board, curPlayer, action)
            scores = self.game.getGameEnded(board, False)

        return scores, iter_step

    def play_games(self, num, start_best, start_not_random):

        results_list = []

        num = int(num / 6)

        for lonely_player in [1, 2]:
            for turn_lonely in range(3):
                if lonely_player == 1:
                    playing_actors = [2] * 3
                    playing_actors[turn_lonely] = 1
                else:
                    playing_actors = [1] * 3
                    playing_actors[turn_lonely] = 2
                for _ in range(num):
                    self.game.reset_board()
                    scores, iter_steps = self.play_game(playing_actors, start_best, start_not_random)

                    results_list.append((playing_actors[0], playing_actors[1], playing_actors[2],
                                         scores[0], scores[1], scores[2], iter_steps))

        return pd.DataFrame(results_list, columns=["Player 1", "Player 2", "Player 3",
                                                   "Score 1", "Score 2", "Score 3", "Steps"])
