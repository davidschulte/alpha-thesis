import numpy as np
import pandas as pd
import os
from pickle import Pickler

versions = [2, 3, 6, 8, 10, 11, 12, 13, 15]

results = np.zeros((3, len(versions)))

def get_counts(file, player):
    places = [0] * 3
    dataframe = pd.read_pickle(file)
    print(dataframe)
    array = dataframe.to_numpy()
    for row in range(120):
        for column in range(3):
            if array[row, column] == player:
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

    vector = np.array(normalized)
    vector = np.transpose(normalized)
    return vector


folder = "tests nnet vs old"

for v in range(len(versions)):
    filename = os.path.join(folder, str(versions[v]) + ".pkl")
    results[:, v] = get_counts(filename, 1)

print(results)
