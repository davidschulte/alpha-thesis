import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import os
from pickle import Pickler

versions_all = [1, 2, 3, 6, 8, 10, 11, 12, 13, 15, 17, 20, 22, 23, 27, 32, 34, 35, 37]
versions = [1, 2, 3, 6, 8, 10, 11, 12, 13, 15, 17, 20, 22, 23, 27]

results = np.zeros((3, len(versions)))

def get_counts_all(file):
    places = [0] * 3
    dataframe = pd.read_pickle(file)
    print(dataframe.to_string())
    array = dataframe.to_numpy()
    print(array)
    for row in range(120):
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

def get_counts_separate(file, alone):
    places = [0] * 3
    dataframe = pd.read_pickle(file)
    print(dataframe.to_string())
    array = dataframe.to_numpy()
    if alone:
        sum_players = 5
    else:
        sum_players = 4

    for row in range(120):
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



folder = "nnet vs greedy"

for v in range(0,len(versions)):
    filename = os.path.join(folder, str(versions[v]) + ".pkl")
    results[:, v] = get_counts_separate(filename, True)

print(results)

x = np.array([x for x in range(1,len(versions)+1)])
for p in range(3):
    plt.plot(x, results[p,:], marker="o")

axes = plt.gca()
axes.set_ylim([0, 1])
plt.xticks(np.arange(min(x), max(x)+1, 1.0))


plt.legend(["Winner", "Second", "Loser"])
plt.title("Neural Net without MCTS against Greedy Actor")
# plt.title("Neural Net without MCTS agains previous version")
plt.xlabel("Version")
plt.ylabel("Relative Frequency of position")
plt.show()


