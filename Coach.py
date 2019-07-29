from collections import deque
from Arena import Arena
from MCTSTExperimental import MCTS
import numpy as np
from pytorch_classification.utils import Bar, AverageMeter
import time, os, sys
from pickle import Pickler, Unpickler
from random import shuffle
import time
import cProfile, pstats, io
from chinese_checkers.VeryGreedyActor import VeryGreedyActor
# from chinese_checkers.GreedyActorExperimental import GreedyActor

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

class Coach():
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

        self.greedy_actor = VeryGreedyActor(game)

    def executeEpisode(self, first):
        """
        This function executes one episode of self-play, starting with player 1.
        As the game is played, each turn is added as a training example to
        trainExamples. The game is played till the game ends. After the game
        ends, the outcome of the game is used to assign values to each example
        in trainExamples.

        It uses a temp=1 if episodeStep < tempThreshold, and thereafter
        uses temp=0.

        Returns:
            trainExamples: a list of examples of the form (canonicalBoard,pi,v)
                           pi is the MCTS informed policy vector, v is +1 if
                           the player eventually won the game, else -1.
        """
        trainExamples = []
        self.board = self.game.getInitBoard()
        self.curPlayer = 1
        episodeStep = 0
        scores = [0, 0, 0]
        self.game.reset_board()
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
                    pi = self.greedy_actor.getActionProb(canonicalBoard, episodeStep < 10)
                else:
                    pi = self.mcts.getActionProb(self.board, self.curPlayer, temp=1)
                # print(max(pi))
                trainExamples.append([canonicalBoard, self.curPlayer, pi, None])

                action = np.random.choice(len(pi), p=pi)
            else:
                action = self.game.getActionSize() - 1

            self.board, self.curPlayer = self.game.getNextState(self.board, self.curPlayer, action)


            # if action != self.game.getActionSize() - 1:
            #     if s not in self.mcts.C:
            #         self.mcts.C[s] = 1
            #     else:
            #         self.mcts.C[s] += 1

            scores = self.game.getGameEnded(self.board, False)
            # _, new_greedy = greedy_actor.predict(self.board, 1)
            # for i in range(3):
            #     if new_greedy[i] < greedy_scores[i]:
            #         print('-')
            #     elif new_greedy[i] > greedy_scores[i]:
            #         print('+')
            # greedy_scores = new_greedy

        scores_player_two = np.array([scores[1], scores[2], scores[0]])
        scores_player_three = np.array([scores[2], scores[0], scores[1]])
        scores_all = [scores, scores_player_two, scores_player_three]
        if not first:
            print("GAME OVER! Step" + str(episodeStep))
            print(self.board)
        return [(x[0], x[2], scores_all[x[1]-1]) for x in trainExamples]

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
                iterationTrainExamples = deque([], maxlen=self.args.maxlenOfQueue)

                num_eps = self.args.numEps
                if i == 1:
                    num_eps = 500
                eps_time = AverageMeter()
                bar = Bar('Self Play', max=num_eps)
                end = time.time()

                for eps in range(num_eps):
                    self.mcts = MCTS(self.game, self.nnet, self.args)   # reset search tree
                    iterationTrainExamples += self.executeEpisode(i == 1)
    
                    # bookkeeping + plot progress
                    eps_time.update(time.time() - end)
                    end = time.time()
                    bar.suffix  = '({eps}/{maxeps}) Eps Time: {et:.3f}s | Total: {total:} | ETA: {eta:}'.format(eps=eps+1, maxeps=num_eps, et=eps_time.avg,
                                                                                                               total=bar.elapsed_td, eta=bar.eta_td)
                    bar.next()
                bar.finish()

                # save the iteration examples to the history 
                self.trainExamplesHistory.append(iterationTrainExamples)
                
            if len(self.trainExamplesHistory) > self.args.numItersForTrainExamplesHistory:
                print("len(trainExamplesHistory) =", len(self.trainExamplesHistory), " => remove the oldest trainExamples")
                self.trainExamplesHistory.pop(0)
            # backup history to a file
            # NB! the examples were collected using the model from the previous iteration, so (i-1)  
            self.saveTrainExamples(i-1)
            
            # shuffle examlpes before training
            trainExamples = []
            for e in self.trainExamplesHistory:
                trainExamples.extend(e)
            shuffle(trainExamples)

            # training new network, keeping a copy of the old one
            self.nnet.save_checkpoint(folder=self.args.checkpoint, filename='temp.h5')
            self.pnet.load_checkpoint(folder=self.args.checkpoint, filename='temp.h5')
            pmcts = MCTS(self.game, self.pnet, self.args)
            
            self.nnet.train(trainExamples)
            nmcts = MCTS(self.game, self.nnet, self.args)

            print('PITTING AGAINST PREVIOUS VERSION')
            # arena = Arena(lambda x: np.argmax(pmcts.getActionProb(x, temp=1)),
            #               lambda x: np.argmax(nmcts.getActionProb(x, temp=1)), self.game)
            # arena = Arena(lambda x: np.random.choice(self.game.getActionSize(), p=pmcts.getActionProb(x, temp=1)),
            #               lambda x: np.random.choice(self.game.getActionSize(), p=nmcts.getActionProb(x, temp=1)), self.game, self.args)

            arena = Arena(self.pnet, self.nnet, self.game, self.args)

            scores = arena.playGames(self.args.arenaCompare)

            # print('NEW/PREV WINS : %d / %d ; DRAWS : %d' % (nwins, pwins, draws))
            # if pwins+nwins == 0 or float(nwins)/(pwins+nwins) < self.args.updateThreshold:
            #     print('REJECTING NEW MODEL')
            #     self.nnet.load_checkpoint(folder=self.args.checkpoint, filename='temp.pth.tar')
            # else:
            #     print('ACCEPTING NEW MODEL')
            #     self.nnet.save_checkpoint(folder=self.args.checkpoint, filename=self.getCheckpointFile(i))
            #     self.nnet.save_checkpoint(folder=self.args.checkpoint, filename='best.pth.tar')

            if scores[1] == 0 or float(scores[1]) / sum(scores) < self.args.updateThreshold:
                print('REJECTING NEW MODEL')
                self.nnet.load_checkpoint(folder=self.args.checkpoint, filename='temp.h5')
            else:
                print('ACCEPTING NEW MODEL')
                self.nnet.save_checkpoint(folder=self.args.checkpoint, filename=self.getCheckpointFile(i))
                self.nnet.save_checkpoint(folder=self.args.checkpoint, filename='temp.h5')

    def getCheckpointFile(self, iteration):
        return 'checkpoint_' + str(iteration) + '.h5'

    def saveTrainExamples(self, iteration):
        folder = self.args.checkpoint
        if not os.path.exists(folder):
            os.makedirs(folder)
        filename = os.path.join(folder, self.getCheckpointFile(iteration)+".examples")
        with open(filename, "wb+") as f:
            Pickler(f).dump(self.trainExamplesHistory)
        f.closed

    def loadTrainExamples(self):
        modelFile = os.path.join(self.args.load_folder_file[0], self.args.load_folder_file[1])
        examplesFile = modelFile+".examples"
        if not os.path.isfile(examplesFile):
            print(examplesFile)
            r = input("File with trainExamples not found. Continue? [y|n]")
            if r != "y":
                sys.exit()
        else:
            print("File with trainExamples found. Read it.")
            with open(examplesFile, "rb") as f:
                self.trainExamplesHistory = Unpickler(f).load()
            f.closed
            # examples based on the model were already collected (loaded)
            self.skipFirstSelfPlay = True


