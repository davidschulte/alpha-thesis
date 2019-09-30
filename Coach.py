from collections import deque
from Arena import Arena
from MCTS import MCTS
import numpy as np
from pytorch_classification.utils import Bar, AverageMeter
import time, os, sys
from pickle import Pickler, Unpickler
from random import shuffle
from chinese_checkers.InitializeAgent import InitializeAgent


class Coach:
    """
    This class executes the self-play + learning. It uses the functions defined
    in Game and NeuralNet. args are specified in main.py.
    """
    def __init__(self, game, nnet, args):
        self.game = game
        self.board = self.game.getInitBoard()
        self.nnet = nnet
        self.pnet = self.nnet.__class__(self.game)  # the competitor network
        self.args = args
        self.mcts = MCTS(self.game, self.nnet, self.args)
        self.trainExamplesHistory = []    # history of examples from args.numItersForTrainExamplesHistory latest iterations
        self.skipFirstSelfPlay = False # can be overriden in loadTrainExamples()
        self.curPlayer = 1

        self.greedy_actor = InitializeAgent(game)

    def execute_episode(self, first):
        """
        """
        train_examples = []
        self.board = self.game.getInitBoard()
        self.curPlayer = 1
        episodeStep = 0
        scores = [0, 0, 0]
        self.game.reset_logic()
        self.mcts.Visited = []

        start_time = time.time()
        while np.count_nonzero(scores) < 2 and episodeStep < 1000:
            if episodeStep % 100 == 0:
                end_time = time.time()
                print("Step " + str(episodeStep) + ": " + str(end_time-start_time) + "s")
                start_time = end_time
                print(self.board)
                # _, new_greedy = greedy_actor.predict(self.board, 1)
                # print(greedy_scores)

            s = self.game.stringRepresentation(self.board)
            self.mcts.Visited.append(s)

            if scores[self.curPlayer - 1] == 0:
                episodeStep += 1
                canonicalBoard = self.game.getCanonicalForm(self.board, self.curPlayer)
                # temp = int(episodeStep < self.args.tempThreshold)
                # temp = 1
                if first and not self.args.load_model:
                    pi = self.greedy_actor.get_action_prob(canonicalBoard, episodeStep < 10)
                else:
                    pi = self.mcts.get_action_prob(self.board, self.curPlayer)
                # print(max(pi))
                train_examples.append([canonicalBoard, self.curPlayer, pi, None])

                action = np.random.choice(len(pi), p=pi)
            else:
                action = self.game.getActionSize() - 1

            self.board, self.curPlayer = self.game.getNextState(self.board, self.curPlayer, action)
            scores = self.game.getGameEnded(self.board, False)

        scores_player_two = np.array([scores[1], scores[2], scores[0]])
        scores_player_three = np.array([scores[2], scores[0], scores[1]])
        scores_all = [scores, scores_player_two, scores_player_three]
        if not first:
            print("GAME OVER! Step" + str(episodeStep))
            print(self.board)
        return [(x[0], x[2], scores_all[x[1]-1]) for x in train_examples]

    def learn(self):
        """
        Performs numIters iterations with numEps episodes of self-play in each
        iteration. After every iteration, it retrains neural network with
        examples in trainExamples (which has a maximium length of maxlenofQueue).
        It then pits the new neural network against the old one and accepts it
        only if it wins >= updateThreshold fraction of games.
        """

        for i in range(1, self.args.numIters+1):
            # bookkeeping
            print('------ITER ' + str(i) + '------')
            # examples of the iteration
            if not self.skipFirstSelfPlay or i>1:
                if i == 2 and not self.skipFirstSelfPlay:
                    iteration_train_examples = []
                else:
                    iteration_train_examples = deque([], maxlen=self.args.maxlenOfQueue)

                num_eps = self.args.numEps
                if i == 1 and not self.args.load_model:
                    num_eps = 1000
                eps_time = AverageMeter()
                bar = Bar('Self Play', max=num_eps)
                end = time.time()

                for eps in range(num_eps):
                    self.mcts = MCTS(self.game, self.nnet, self.args)   # reset search tree
                    iteration_train_examples += self.execute_episode(i == 1)
    
                    # bookkeeping + plot progress
                    eps_time.update(time.time() - end)
                    end = time.time()
                    bar.suffix  = '({eps}/{maxeps}) Eps Time: {et:.3f}s | Total: {total:} | ETA: {eta:}'.format(eps=eps+1, maxeps=num_eps, et=eps_time.avg,
                                                                                                               total=bar.elapsed_td, eta=bar.eta_td)
                    bar.next()
                bar.finish()

                # save the iteration examples to the history 
                self.trainExamplesHistory.append(iteration_train_examples)
                
            if len(self.trainExamplesHistory) > self.args.numItersForTrainExamplesHistory:
                print("len(trainExamplesHistory) =", len(self.trainExamplesHistory), " => remove the oldest trainExamples")
                self.trainExamplesHistory.pop(0)
            # backup history to a file
            # NB! the examples were collected using the model from the previous iteration, so (i-1)  
            self.save_train_examples(i)
            
            # shuffle examlpes before training
            trainExamples = []
            for e in self.trainExamplesHistory:
                trainExamples.extend(e)
            shuffle(trainExamples)

            # training new network, keeping a copy of the old one
            self.nnet.save_checkpoint(folder=self.args.checkpoint, filename='temp.h5')
            self.pnet.load_checkpoint(folder=self.args.checkpoint, filename='temp.h5')

            self.nnet.train(trainExamples)

            print('PITTING AGAINST PREVIOUS VERSION')

            arena = Arena(self.pnet, self.nnet, self.game, self.args)

            scores = arena.playGames(self.args.arenaCompare)
            if scores[1] == 0 or float(scores[1]) / sum(scores) < self.args.updateThreshold:
                print('REJECTING NEW MODEL')
                self.nnet.load_checkpoint(folder=self.args.checkpoint, filename='temp.h5')
            else:
                print('ACCEPTING NEW MODEL')
                self.nnet.save_checkpoint(folder=self.args.checkpoint, filename=self.get_checkpoint_file(i))
                self.nnet.save_checkpoint(folder=self.args.checkpoint, filename='temp.h5')

    def get_checkpoint_file(self, iteration):
        return 'checkpoint_' + str(iteration) + '.h5'

    def save_train_examples(self, iteration):
        folder = self.args.checkpoint
        if not os.path.exists(folder):
            os.makedirs(folder)
        filename = os.path.join(folder, self.get_checkpoint_file(iteration) + ".examples")
        with open(filename, "wb+") as f:
            Pickler(f).dump(self.trainExamplesHistory)
        f.closed

    def load_train_examples(self):
        model_file = os.path.join(self.args.load_folder_file[0], self.args.load_folder_file[1])
        examples_file = model_file+".examples"
        if not os.path.isfile(examples_file):
            print(examples_file)
            r = input("File with trainExamples not found. Continue? [y|n]")
            if r != "y":
                sys.exit()
        else:
            print("File with trainExamples found. Read it.")
            with open(examples_file, "rb") as f:
                self.trainExamplesHistory = Unpickler(f).load()
            f.closed
            # examples based on the model were already collected (loaded)
            self.skipFirstSelfPlay = True


