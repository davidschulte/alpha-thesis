from collections import deque
from Arena import Arena
from ParallelMCTS import MCTS
import numpy as np
from pytorch_classification.utils import Bar, AverageMeter
import time, os, sys
from pickle import Pickler, Unpickler
from random import shuffle
import time
import cProfile, pstats, io
from chinese_checkers.VeryGreedyActor import VeryGreedyActor as Actor
# from chinese_checkers.RandomActor import RandomActor as Actor
from chinese_checkers.TinyChineseCheckersGame import ChineseCheckersGame
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
        self.nnet = nnet
        self.pnet = self.nnet.__class__(self.game)  # the competitor network
        self.args = args
        self.mcts = MCTS(self.game, self.nnet, self.args)
        self.trainExamplesHistory = []    # history of examples from args.numItersForTrainExamplesHistory latest iterations
        self.skipFirstSelfPlay = False  # can be overriden in loadTrainExamples()
        self.curPlayer = 1

        self.all_train_examples = None
        self.all_boards = None
        self.all_episode_steps = None
        self.all_scores = None
        self.all_cur_players = None
        self.all_done = None
        self.games = None
        self.mctss = None

        self.greedy_actor = Actor(game)

    def initialize_parallel(self):
        self.all_train_examples = [[] for _ in range(self.args.parallel_block)]
        self.all_boards = [self.game.getInitBoard()] * self.args.parallel_block
        self.all_episode_steps = [0] * self.args.parallel_block
        self.all_scores = [[0, 0, 0] for _ in range(self.args.parallel_block)]
        self.all_cur_players = [1] * self.args.parallel_block
        self.all_done = [False] * self.args.parallel_block
        self.games = [ChineseCheckersGame() for _ in range(self.args.parallel_block)]
        self.mctss = [MCTS(self.games[i], self.nnet, self.args) for i in range(self.args.parallel_block)]

    def execute_greedy_episode(self):
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
        board = self.game.getInitBoard()
        curPlayer = 1
        episodeStep = 0
        scores = [0, 0, 0]
        self.game.reset_board()

        while np.count_nonzero(scores) < 2 and episodeStep < self.args.max_steps:

            if scores[curPlayer - 1] == 0:
                episodeStep += 1
                canonicalBoard = self.game.getCanonicalForm(board, curPlayer)
                pi = self.greedy_actor.getActionProb(canonicalBoard, False)
                # pi = self.greedy_actor.getActionProb(canonicalBoard)

                trainExamples.append([canonicalBoard, curPlayer, pi, None])
                action = np.random.choice(len(pi), p=pi)
            else:
                action = self.game.getActionSize() - 1

            board, curPlayer = self.game.getNextState(board, curPlayer, action)

            scores = self.game.getGameEnded(board, False)

        #     if episodeStep % 100 == 0:
        #         print(episodeStep)
        #         print(board)
        #
        # print(board)
        scores_player_two = np.array([scores[1], scores[2], scores[0]])
        scores_player_three = np.array([scores[2], scores[0], scores[1]])
        scores_all = [scores, scores_player_two, scores_player_three]
        return [(x[0], x[2], scores_all[x[1]-1]) for x in trainExamples]

    def execute_episodes(self):
        self.initialize_parallel()
        it = 1
        start_time = time.time()

        matches_over = 0
        while False in self.all_done:
            # requests = self.get_requests()
            #
            # update_indices = []
            # valid_requests = []
            #
            # for i in range(self.args.parallel_block):
            #     if requests[i] is not None:
            #         update_indices.append(i)
            #         valid_requests.append(requests[i])
            #
            # matches_over_new = self.args.parallel_block - len(update_indices)
            # if matches_over_new > matches_over:
            #     matches_over = matches_over_new
            #     print(str(matches_over) + " games decided!")
            #
            # if len(update_indices) > 0:
            #     pis, vs = self.nnet.predict_parallel(valid_requests)
            #     self.update_predictions(pis, vs, update_indices)
            # if it % self.args.numMCTSSims == 0:
            #     end_time = time.time()
            #     print(str(int(it/self.args.numMCTSSims)) + " steps: " + str(int(end_time-start_time)) + "s")
            #     start_time = end_time
            # it += 1
            self.one_iter()

        return self.compile_train_examples()

    @profile
    def one_iter(self):
        requests = self.get_requests()

        update_indices = []
        valid_requests = []

        for i in range(self.args.parallel_block):
            if requests[i] is not None:
                update_indices.append(i)
                valid_requests.append(requests[i])

        pis, vs = self.nnet.predict_parallel(valid_requests)
        self.update_predictions(pis, vs, update_indices)

    def get_requests(self):
        all_request_states = [None] * self.args.parallel_block
        for n in range(self.args.parallel_block):
            if not self.all_done[n]:
                request_state = None
                scores = self.all_scores[n]
                while not self.all_done[n] and request_state is None:
                    game = self.games[n]
                    mcts = self.mctss[n]
                    board = self.all_boards[n]
                    cur_player = self.all_cur_players[n]
                    # mcts.search(board, cur_player, 0)
                    # mcts.add_iter()
                    if mcts.get_done():
                        s = self.game.stringRepresentation(board)
                        mcts.Visited.append(s)
                        pi = mcts.get_counts(board, cur_player)
                        canonical_board = self.game.getCanonicalForm(board, cur_player)
                        self.all_train_examples[n].append([canonical_board, cur_player, pi, None])
                        action = np.random.choice(len(pi), p=pi)
                        board, cur_player = game.getNextState(board, cur_player, action)
                        scores = game.getGameEnded(board, False)
                        self.all_episode_steps[n] += 1
                        if self.all_episode_steps[n] > self.args.max_steps or np.count_nonzero(scores) > 1:
                            self.all_done[n] = True
                        else:
                            if scores[cur_player-1] != 0:
                                action = game.getActionSize()-1
                                _, cur_player = game.getNextState(board, cur_player, action)
                    if not self.all_done[n]:
                        request_state = mcts.search(board, cur_player, 0)

                self.all_boards[n] = board
                self.all_scores[n] = scores
                self.all_cur_players[n] = cur_player
                all_request_states[n] = request_state

        return all_request_states

    def update_predictions(self, pis, vs, update_indices):
        for i in range(len(update_indices)):
            index = update_indices[i]
            mcts = self.mctss[index]
            mcts.update_predictions(pis[i,:], vs[i,:])

    def compile_train_examples(self):
        train_examples = []
        for n in range(self.args.parallel_block):
            single_examples = self.all_train_examples[n]
            scores = self.all_scores[n]
            scores_player_two = np.array([scores[1], scores[2], scores[0]])
            scores_player_three = np.array([scores[2], scores[0], scores[1]])
            scores_list = [scores, scores_player_two, scores_player_three]
            single_examples_processed = [(x[0], x[2], scores_list[x[1] - 1]) for x in single_examples]
            train_examples += single_examples_processed

        return train_examples

    def learn(self):
        """
        Performs numIters iterations with numEps episodes of self-play in each
        iteration. After every iteration, it retrains neural network with
        examples in trainExamples (which has a maximium length of maxlenofQueue).
        It then pits the new neural network against the old one and accepts it
        only if it wins >= updateThreshold fraction of games.
        """
        if self.args.load_model:
            start = self.args.load_folder_file[1] + 1
        else:
            start = 1
        for i in range(start, self.args.numIters+1):
            # bookkeeping
            print('------ITER ' + str(i) + '------')
            # examples of the iteration
            greedy = i == 1 and not self.args.load_model

            if not self.skipFirstSelfPlay or i>1:
                iterationTrainExamples = deque([], maxlen=self.args.maxlenOfQueue)

                num_eps = self.args.numEps
                if greedy:
                    num_eps = self.args.greedy_eps
                eps_time = AverageMeter()
                bar = Bar('Self Play', max=num_eps)
                end = time.time()

                for eps in range(num_eps):
                    if greedy:
                        iterationTrainExamples += self.execute_greedy_episode()
                    else:
                        iterationTrainExamples += self.execute_episodes()

                    # bookkeeping + plot progress
                    eps_time.update(time.time() - end)
                    end = time.time()
                    bar.suffix  = '({eps}/{maxeps}) Eps Time: {et:.3f}s | Total: {total:} | ETA: {eta:}'.format(eps=eps+1, maxeps=num_eps, et=eps_time.avg,
                                                                                                               total=bar.elapsed_td, eta=bar.eta_td)
                    bar.next()
                bar.finish()

                # save the iteration examples to the history
                if not greedy:
                    self.trainExamplesHistory.append(iterationTrainExamples)
                
                    if len(self.trainExamplesHistory) > self.args.numItersForTrainExamplesHistory:
                        print("len(trainExamplesHistory) =", len(self.trainExamplesHistory), " => remove the oldest trainExamples")
                        self.trainExamplesHistory.pop(0)
                    # backup history to a file
                    # NB! the examples were collected using the model from the previous iteration, so (i-1)
                    self.saveTrainExamples(i)
            
                    # shuffle examples before training
                    trainExamples = []
                    for e in self.trainExamplesHistory:
                        trainExamples.extend(e)
                    shuffle(trainExamples)

                else:
                    trainExamples = iterationTrainExamples

            # training new network, keeping a copy of the old one
            self.nnet.save_checkpoint(folder=self.args.checkpoint, filename='temp.h5')
            self.pnet.load_checkpoint(folder=self.args.checkpoint, filename='temp.h5')

            self.nnet.train(trainExamples)

            if not greedy:
                print('PITTING AGAINST PREVIOUS VERSION')
                arena = Arena(self.pnet, self.nnet, self.game, self.args)
                scores = arena.playGames(self.args.arenaCompare)

                if scores[1] == 0 or float(scores[1]) / sum(scores) < self.args.updateThreshold:
                    print('REJECTING NEW MODEL')
                    self.nnet.load_checkpoint(folder=self.args.checkpoint, filename='temp.h5')
                else:
                    print('ACCEPTING NEW MODEL')
                    self.nnet.save_checkpoint(folder=self.args.checkpoint, filename=self.getCheckpointFile(i))
                    self.nnet.save_checkpoint(folder=self.args.checkpoint, filename='temp.h5')
            else:
                self.nnet.save_checkpoint(folder=self.args.checkpoint, filename='checkpoint_1.h5')

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
        file_name = "checkpoint_" + str(self.args.load_folder_file[1]) + ".h5"
        modelFile = os.path.join(self.args.load_folder_file[0], file_name)
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
            # self.skipFirstSelfPlay = True


