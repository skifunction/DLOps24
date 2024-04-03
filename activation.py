import numpy as np
import matplotlib.pyplot as plt

x = np.linespace(-5, 5, 100)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def relu(x):
    return np.maximum(0, x)


def leaky_relu(x, alpha=0.01):
    return np.where(x >= 0, x, alpha * x)  # Changed, so 0 is also considered


def tanh(x):
    return np.tanh(x)


y_sig = sigmoid(x)
y_relu = relu(x)
y_l_relu = leaky_relu(x)
y_tanh = tanh(x)

plt.figure(figsize=(10, 6))
plt.plot(x, y_sigmoid, label="Sigmoid")
plt.plot(x, y_relu, label="ReLU")
plt.plot(x, y_leaky_relu, label="Leaky ReLU")
plt.plot(x, y_tanh, label="Tanh")
plt.xlabel("Input")
plt.ylabel("Output")
plt.title("Activation Functions")
plt.legend()
plt.grid(True)
plt.show()
