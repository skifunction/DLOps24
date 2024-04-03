import numpy as np


def relu(x):
    return np.maximum(0, x)


def leaky_relu(x, alpha=0.01):
    return np.where(x > 0, x, alpha * x)


def tanh(x):
    return np.tanh(x)


random_values = [-4.5, -1.2, 0, 2.8, -4.1, 1.5, -0.7, 3.2, -2.4, 4.6]

print("ReLU:")
for val in random_values:
    print(relu(val))

print("\nLeaky ReLU:")
for val in random_values:
    print(leaky_relu(val))

print("\nTanh:")
for val in random_values:
    print(tanh(val))
