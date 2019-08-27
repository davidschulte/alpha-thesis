from utils import *
from chinese_checkers.TinyChineseCheckersGame import ChineseCheckersGame
from chinese_checkers.tensorflow.ResNet import NNetWrapper as nn
from EvalArena import Arena
from MCTSTExperimental import MCTS
import os
from chinese_checkers.ForwardActor import ForwardActor
from pickle import Pickler
from chinese_checkers.InitializeActor import VeryGreedyActor
from chinese_checkers.ForwardActor import ForwardActor

import numpy as np


args = dotdict({
    'numMCTSSims': 200,
    'cpuct': 15,
    'max_steps': 120,
})

args2 = dotdict({
    'numMCTSSims': 2,
    'cpuct': 15,
    'max_steps': 120,
})

versions = [1, 2, 3, 6, 8, 10, 11, 12, 13, 15, 17, 20, 22, 23, 27]

game = ChineseCheckersGame()
greedy = ForwardActor(game)
initialize = VeryGreedyActor(game)

# for new in range(10, len(versions)):
#     print(str(new) + "/" + str(len(versions)))
#     nn_new = nn(game)
#     nn_new.load_first_checkpoint('checkpoint', versions[new])
#     mcts_new = MCTS(game, nn_new, args2)
#
#     # nn_old = nn(game)
#     # nn_old.load_first_checkpoint('checkpoint', versions[new-1])
#     # mcts_old = MCTS(game, nn_old, args)
#
#     arena = Arena(mcts_new, initialize, game, args2)
#     results = arena.play_games(120, 1, 7)
#
#     print(results)
#
#     folder = "nnet vs initialize"
#     if not os.path.exists(folder):
#         os.makedirs(folder)
#     filename = os.path.join(folder, str(versions[new]) + ".pkl")
#     results.to_pickle(filename)


for new in range(14,len(versions)):
    print(str(new) + "/" + str(len(versions)))
    nn_new = nn(game)
    nn_new.load_first_checkpoint('checkpoint', versions[new])
    mcts_new = MCTS(game, nn_new, args)

    nn_old = nn(game)
    nn_old.load_first_checkpoint('checkpoint', versions[new-1])
    mcts_old = MCTS(game, nn_old, args)

    arena = Arena(mcts_new, mcts_old, game, args)
    results = arena.play_games(30, 1, 7)

    print(results)

    folder = "mcts vs mcts old"
    if not os.path.exists(folder):
        os.makedirs(folder)
    filename = os.path.join(folder, str(versions[new]) + ".pkl")
    results.to_pickle(filename)

for new in range(len(versions)):
    print(str(new) + "/" + str(len(versions)))
    nn_new = nn(game)
    nn_new.load_first_checkpoint('checkpoint', versions[new])
    mcts_new = MCTS(game, nn_new, args)

    # nn_old = nn(game)
    # nn_old.load_first_checkpoint('checkpoint', versions[new-1])
    # mcts_old = MCTS(game, nn_old, args)

    arena = Arena(mcts_new, greedy, game, args)
    results = arena.play_games(30, 1, 7)

    print(results)

    folder = "mcts vs greedy"
    if not os.path.exists(folder):
        os.makedirs(folder)
    filename = os.path.join(folder, str(versions[new]) + ".pkl")
    results.to_pickle(filename)
