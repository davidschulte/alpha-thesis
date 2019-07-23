#!/opt/python/python-3.6/i/bin/python

from Coach import Coach
from chinese_checkers.SmallChineseCheckersGame import ChineseCheckersGame as Game
from chinese_checkers.tensorflow.NNet import NNetWrapper as nn
from utils import *
import numpy as np
import sys

np.set_printoptions(threshold=sys.maxsize)


args = dotdict({
    'numIters': 1000,
    'numEps': 50,
    'tempThreshold': 15,
    'updateThreshold': 0.6,
    'maxlenOfQueue': 100000,
    'numMCTSSims': 30,
    'arenaCompare': 6,
    'cpuct': 1.5,

    'checkpoint': 'checkpoint',
    'load_model': False,
    'load_folder_file': ('checkpoints','best.pth.tar'),
    'numItersForTrainExamplesHistory': 20,

})

if __name__=="__main__":
    g = Game()
    nnet = nn(g)

    if args.load_model:
        nnet.load_checkpoint(args.load_folder_file[0], args.load_folder_file[1])

    c = Coach(g, nnet, args)
    if args.load_model:
        print("Load trainExamples from file")
        c.loadTrainExamples()
    c.learn()
