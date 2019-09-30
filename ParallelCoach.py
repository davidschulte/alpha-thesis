from collections import deque
from Arena import Arena
from ParallelMCTS import MCTS
from MCTS import MCTS as MCTSSingle
import numpy as np
from pytorch_classification.utils import Bar, AverageMeter
import time, os, sys
from pickle import Pickler, Unpickler
from random import shuffle
import gc
from chinese_checkers.InitializeAgent import InitializeAgent
from chinese_checkers.TinyChineseCheckersGame import ChineseCheckersGame


class Coach:

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

        self.initialize_agent = InitializeAgent(game)

    def initialize_parallel(self):
        """
        initializes instances before self-play phase
        """
        self.all_train_examples = [[] for _ in range(self.args.parallel_block)]
        self.all_boards = [self.game.getInitBoard()] * self.args.parallel_block
        self.all_episode_steps = [1] * self.args.parallel_block
        self.all_scores = [[0, 0, 0] for _ in range(self.args.parallel_block)]
        self.all_cur_players = [1] * self.args.parallel_block
        self.all_done = [False] * self.args.parallel_block
        self.games = [ChineseCheckersGame() for _ in range(self.args.parallel_block)]
        self.mctss = [MCTS(self.games[i], self.nnet, self.args) for i in range(self.args.parallel_block)]
        gc.collect()

    def execute_initialize_episode(self):
        """
        executes a game played by the Initiliaze Agent
        :return: train examples
        """
        train_examples = []
        board = self.game.getInitBoard()
        cur_player = 1
        episode_step = 0
        scores = [0, 0, 0]
        self.game.reset_logic()

        while np.count_nonzero(scores) < 2 and episode_step < self.args.max_steps:

            if scores[cur_player - 1] == 0:
                episode_step += 1
                canonical_board = self.game.getCanonicalForm(board, cur_player)
                pi = self.initialize_agent.get_action_prob(canonical_board, 1)
                train_examples.append([canonical_board, cur_player, pi, None])
                action = np.random.choice(len(pi), p=pi)
            else:
                action = self.game.getActionSize() - 1

            board, cur_player = self.game.getNextState(board, cur_player, action)

            scores = self.game.getGameEnded(board, False)

        scores_player_two = np.array([scores[1], scores[2], scores[0]])
        scores_player_three = np.array([scores[2], scores[0], scores[1]])
        scores_all = [scores, scores_player_two, scores_player_three]
        return [(x[0], x[2], scores_all[x[1]-1]) for x in train_examples]

    def execute_episodes(self):
        """
        executes several games played by the Main Agent
        :return: train examples
        """
        self.initialize_parallel()
        it = 1
        start_time = time.time()

        matches_over = 0
        while False in self.all_done:
            requests = self.get_requests()

            update_indices = []
            valid_requests = []

            for i in range(self.args.parallel_block):
                if requests[i] is not None:
                    update_indices.append(i)
                    valid_requests.append(requests[i])

            matches_over_new = self.args.parallel_block - len(update_indices)
            if matches_over_new > matches_over:
                matches_over = matches_over_new
                print(str(matches_over) + "/" + str(self.args.parallel_block) + " games decided!")

            if len(update_indices) > 0:
                pis, vs = self.nnet.predict_parallel(valid_requests)
                self.update_predictions(pis, vs, update_indices)
            if it % self.args.numMCTSSims == 0:
                end_time = time.time()
                print(str(int(it/self.args.numMCTSSims)) + " steps: " + str(int(end_time-start_time)) + "s")
                start_time = end_time
            it += 1
        return self.compile_train_examples()

    def get_requests(self):
        """
        collects states whose predictions are requested
        :return: list of states
        """
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
                    if mcts.get_done():
                        s = self.game.stringRepresentation(board)
                        mcts.Visited.append(s)
                        canonical_board = self.game.getCanonicalForm(board, cur_player)
                        pi = mcts.get_counts(board, cur_player)
                        self.all_train_examples[n].append([canonical_board, cur_player, pi])
                        if self.all_episode_steps[n] > 30:
                            best_actions = np.zeros(self.game.getActionSize())
                            best_actions[np.where(pi == np.amax(pi))] = 1
                            pi = best_actions / sum(best_actions)
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
        """
        updates the values inside the tree searches
        :param pis: predicted pi values
        :param vs: predicted v values
        :param update_indices: indices corresponding to the tree search instances
        """
        for i in range(len(update_indices)):
            index = update_indices[i]
            mcts = self.mctss[index]
            mcts.update_predictions(pis[i,:], vs[i,:])

    def compile_train_examples(self):
        """
        compiles the train examples through usage of r_v(v,p)
        :return: train examples
        """
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
        main loop of the training loop

        """
        if self.args.load_model:
            start = self.args.load_example[1] + 1
        else:
            start = 1
        for i in range(start, self.args.numIters+1):
            # bookkeeping
            print('------ITER ' + str(i) + '------')
            # examples of the iteration
            greedy = i == 1 and not self.args.load_model

            if not self.skipFirstSelfPlay or i>1:
                iteration_train_examples = deque([], maxlen=self.args.maxlenOfQueue)

                num_eps = self.args.numEps
                if greedy:
                    num_eps = self.args.greedy_eps
                eps_time = AverageMeter()
                bar = Bar('Self Play', max=num_eps)
                end = time.time()

                for eps in range(num_eps):
                    if greedy:
                        iteration_train_examples += self.execute_initialize_episode()
                    else:
                        iteration_train_examples += self.execute_episodes()

                    # bookkeeping + plot progress
                    eps_time.update(time.time() - end)
                    end = time.time()
                    bar.suffix  = '({eps}/{maxeps}) Eps Time: {et:.3f}s | Total: {total:} | ETA: {eta:}'.format(eps=eps+1, maxeps=num_eps, et=eps_time.avg,
                                                                                                               total=bar.elapsed_td, eta=bar.eta_td)
                    bar.next()
                bar.finish()

                # save the iteration examples to the history
                if not greedy:
                    self.trainExamplesHistory.append(iteration_train_examples)
                
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
                    trainExamples = iteration_train_examples

            # training new network, keeping a copy of the old one
            self.nnet.save_checkpoint(folder=self.args.checkpoint, filename='temp.h5')
            self.pnet.load_checkpoint(folder=self.args.checkpoint, filename='temp.h5')

            self.nnet.train(trainExamples)

            if not greedy:
                pmcts = MCTSSingle(self.game, self.pnet, self.args)
                nmcts = MCTSSingle(self.game, self.nnet, self.args)
                print('PITTING AGAINST PREVIOUS VERSION')
                arena = Arena(pmcts, nmcts, self.game, self.args)
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
        file_name = "checkpoint_" + str(self.args.load_example[1]) + ".h5"
        modelFile = os.path.join(self.args.load_example[0], file_name)
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


