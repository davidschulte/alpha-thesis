import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import os
from pickle import Pickler

def get_counts_player(file, games_played):
    places = np.zeros((3, 3))
    dataframe = pd.read_pickle(file)
    array = dataframe.to_numpy()
    for row in range(games_played):
        for column in range(3):
            if array[row, column+3] == 3:
                places[column, 0] += 1
            elif array[row, column+3] == 1:
                places[column, 1] += 1
            else:
                places[column, 2] += 1
    print(places)
    sum_places = sum(places)
    normalized = [x / sum_places for x in places]
    print(normalized)

    vector = np.transpose(normalized)
    return vector

def get_counts_all(file, games_played):
    places = [0] * 3
    dataframe = pd.read_pickle(file)
    print(dataframe.to_string())
    array = dataframe.to_numpy()
    print(array)
    for row in range(games_played):
        for column in range(3):
            if array[row, column] == 1:
                if array[row, column + 3] == 3:
                    places[0] += 1
                elif array[row, column + 3] == 1:
                    places[1] += 1
                else:
                    places[2] += 1
    print(places)
    sum_places = sum(places)
    normalized = [x / sum_places for x in places]
    print(normalized)

    vector = np.transpose(normalized)
    return vector

def get_counts_separate(file, alone, games_played):
    places = [0] * 3
    dataframe = pd.read_pickle(file)
    print(dataframe.to_string())
    array = dataframe.to_numpy()
    if alone:
        sum_players = 5
    else:
        sum_players = 4

    for row in range(games_played):
        if sum(array[row, 0:3]) == sum_players:
            if alone:
                for column in range(3):
                    if array[row, column] == 1:
                        if array[row, column + 3] == 3:
                            places[0] += 1
                        elif array[row, column + 3] == 1:
                            places[1] += 1
                        else:
                            places[2] += 1
            else:
                sum_positions = 0
                for column in range(3):
                    if array[row, column] == 1:
                        sum_positions += array[row, column + 3]

                print(sum_positions)
                if sum_positions == 4:
                    places[0] += 1
                elif sum_positions == 3:
                    places[1] += 1
                else:
                    places[2] += 1

    print(places)
    sum_places = sum(places)
    normalized = [x / sum_places for x in places]
    print(normalized)

    vector = np.transpose(normalized)
    return vector

#
versions = [1, 2, 3, 6, 8, 10, 11, 12, 13, 15, 17, 20, 22, 23, 27, 32, 34, 35, 37, 41]
#
# results_mcts_alone = np.zeros((3, len(versions)))
# results_mcts_team = np.zeros((3, len(versions)))
# results_nnet_alone = np.zeros((3, len(versions)))
# results_nnet_team = np.zeros((3, len(versions)))
#
# folder_mcts = "mcts vs nnet"
# folder_nnet = "nnet vs nnet"
# games_played_mcts = 30
# games_played_nnet = 120
#
# for v in range(len(versions)):
#     filename = os.path.join(folder_nnet, str(versions[v]) + ".pkl")
#     # results_mcts_alone[:, v] = get_counts_separate(filename, True, games_played_nnet)
#     # results_mcts_team[:, v] = get_counts_separate(filename, False, games_played_nnet)
#     #
#     # filename = os.path.join(folder_nnet, str(versions[v]) + ".pkl")
#     # results_nnet_alone[:, v] = get_counts_separate(filename, True, games_played_nnet)
#     # results_nnet_team[:, v] = get_counts_separate(filename, False, games_played_nnet)




x = np.array([x for x in range(1, len(versions)+1)])


fig = plt.figure()
fig.suptitle("Main Agent vs. Neural Net", fontsize=16)

plt.subplots_adjust(hspace=0.4)

ax1 = fig.add_subplot(211)
plt.xticks(np.arange(min(x), max(x)+1, 1.0))
plt.yticks(np.arange(0, 1.1, 0.1))
plt.ylim((0, 1))
ax1.set_title("1 vs. 2")
plt.xlabel("Version")
plt.ylabel("Relative frequency of score")
plt.grid(True)
ax2 = fig.add_subplot(212)
plt.xticks(np.arange(min(x), max(x)+1, 1.0))
plt.yticks(np.arange(0, 1.1, 0.1))
plt.ylim((0, 1))
ax2.set_title("2 vs. 1")
plt.xlabel("Version")
plt.ylabel("Relative frequency of score")
plt.grid(True)



for p in range(3):
    ax1.plot(x, results_mcts_alone[p,:], marker="o")
    ax2.plot(x, results_mcts_team[p,:], marker="o")


# plt.title("Neural Net without MCTS agains previous version")
ax1.legend(["3", "1", "0"])
ax2.legend(["4", "3", "1"])

plt.show()


