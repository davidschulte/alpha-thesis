import numpy as np
from chinese_checkers.TinyGUI import GUI
from chinese_checkers.TinyChineseCheckersGame import ChineseCheckersGame as Game

game = Game()
gui = GUI(1)

# 0  1  2  3  4  5  6  7  8  9 10 11 12
board = np.array([[4, 4, 4, 4, 4, 4, 0, 4, 4],  # 0
                  [4, 4, 4, 4, 4, 0, 0, 4, 4],  # 1
                  [4, 4, 2, 2, 1, 0, 3, 3, 3],  # 2
                  [4, 4, 2, 2, 0, 0, 0, 3, 4],  # 3
                  [4, 4, 2, 0, 2, 3, 0, 4, 4],  # 4
                  [4, 0, 3, 1, 0, 0, 0, 4, 4],  # 5
                  [0, 0, 0, 1, 1, 0, 0, 4, 4],  # 6
                  [4, 4, 1, 1, 4, 4, 4, 4, 4],  # 7
                  [4, 4, 0, 4, 4, 4, 4, 4, 4]]).astype('int8')  # 8

gui.snapshot(board, "example.png")