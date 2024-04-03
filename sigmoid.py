import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


values = [-3.5, -1.2, 0, 2.8, -4.1, 1.5, -0.7, 3.2, -2.4, 4.6]

for val in values:
    print(sigmoid(val))
