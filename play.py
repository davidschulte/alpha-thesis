from utils import *
from chinese_checkers.TinyChineseCheckersGame import ChineseCheckersGame
from chinese_checkers.tensorflow.ResNet import NNetWrapper as nn
from chinese_checkers.Evaluator import Evaluator
from MCTS import MCTS
from chinese_checkers.InitializeAgent import InitializeAgent
from chinese_checkers.GreedyAgent import GreedyAgent
from chinese_checkers.TinyGUI import GUI
import numpy as np

args = dotdict({
    'numMCTSSims': 2,
    'cpuct': 15,
    'max_steps': 600,

    'load_folder_file': ('checkpoint', 41),
})

args2 = dotdict({
    'numMCTSSims': 200,
    'cpuct': 15,
    'max_steps': 600,

    'load_folder_file': ('checkpoint', 12),
})
game = ChineseCheckersGame()
gui = GUI(1)
nn1 = nn(game)
nn1.load_first_checkpoint(args.load_folder_file[0], args.load_folder_file[1])
mcts1 = MCTS(game, nn1, args)

# nn2 = nn(game)
# nn2.load_first_checkpoint(args2.load_folder_file[0], args2.load_folder_file[1])

# mcts2 = MCTS(game, nn2, args2)
actor = InitializeAgent(game)
forward = GreedyAgent(game)

evaluator = Evaluator(None, mcts1, mcts1, game, gui, True)
scores_all = np.zeros((3, 3))
steps_all = 0
wrong_win_all = 0
for _ in range(20):
    scores, steps, wrong_win = evaluator.play_game(1, 1)
    for p in range(3):
        if scores[p] == 3:
            scores_all[p,0] += 1
        elif scores[p] == 1:
            scores_all[p,1] += 1
        else:
            scores_all[p,2] += 1
    steps_all += steps
    wrong_win_all += wrong_win
    print(scores_all)

