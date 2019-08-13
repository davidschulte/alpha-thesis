from utils import *
from chinese_checkers.TinyChineseCheckersGame import ChineseCheckersGame
from chinese_checkers.tensorflow.ParallelResNet import NNetWrapper as nn
from chinese_checkers.Evaluator import Evaluator
from MCTSTExperimental import MCTS
from chinese_checkers.VeryGreedyActor import VeryGreedyActor
from chinese_checkers.ForwardActor import ForwardActor

args = dotdict({
    'numIters': 1000,
    'numEps': 10,
    'tempThreshold': 15,
    'updateThreshold': 0.55,
    'maxlenOfQueue': 1000000,
    'numMCTSSims': 2,
    'arenaCompare': 12,
    'cpuct': 5,
    'max_steps': 600,
    'parallel_block': 500,
    'greedy_eps': 500,

    'checkpoint': 'checkpoint',
    'load_model': True,
    'load_folder_file': ('checkpoint', 1),
    'numItersForTrainExamplesHistory': 5,
})

game = ChineseCheckersGame()
nn1 = nn(game)
nn1.load_first_checkpoint(args.load_folder_file[0], args.load_folder_file[1])

mcts1 = MCTS(game, nn1, args)
# mcts2 = MCTS(game, nn1, args)
actor = ForwardActor(game)

evaluator = Evaluator(actor, actor, actor, False)
for _ in range(100):
    evaluator.play_game()
