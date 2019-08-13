#!/opt/python/python-3.6/i/bin/python

from ParallelCoach import Coach
from chinese_checkers.TinyChineseCheckersGame import ChineseCheckersGame as Game
from chinese_checkers.tensorflow.ParallelResNet import NNetWrapper as nn
from utils import *
import numpy as np
import sys

np.set_printoptions(threshold=sys.maxsize)


args = dotdict({
    'numIters': 1000,
    'numEps': 1,
    'tempThreshold': 15,
    'updateThreshold': 0.55,
    'maxlenOfQueue': 1000000,
    'numMCTSSims': 200,
    'arenaCompare': 12,
    'cpuct': 20,
    'max_steps': 600,
    'parallel_block': 500,
    'greedy_eps': 500,

    'checkpoint': 'checkpoint',
    'load_model': False,
    'load_folder_file': ('checkpoint', 1),
    'numItersForTrainExamplesHistory': 5,

})

if __name__=="__main__":
    g = Game()
    nnet = nn(g)

    if args.load_model:
        nnet.load_first_checkpoint(args.load_folder_file[0], args.load_folder_file[1])

    c = Coach(g, nnet, args)

    if args.load_model:
        print("Load trainExamples from file")
        c.loadTrainExamples()
    c.learn()
