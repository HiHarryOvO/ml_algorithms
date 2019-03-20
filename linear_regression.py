import numpy as np
import matplotlib.pyplot as plt


def sum_partial(X, y, _theta, attr):
    partial = 0
    for i in range(len(X)):
        partial += (np.dot(_theta, X[i]) - y[i]) * X[i][attr]
    return partial


def loss_function(X, y, _theta):
    loss = 0
    for i in range(len(X)):
        loss += (np.dot(_theta, X[i]) - y[i]) ** 2
    return loss / (2 * len(X))


X = [[1, -0.9, -0.8], [1, -0.78, -0.56], [1, 0, 0.2]]
y = [-0.7, -0.23, 0.5]

alpha = 0.0001  # typically we set it to be O(1 / sqrt(t))
m = 3
theta = [0 for i in range(3)]

loss_all = []
while True:
    for theta_i in theta:
        theta_i -= alpha * (1 / 3) * sum_partial(X, y, theta, theta.index(theta_i))
    loss_all.append(loss_function(X, y, theta))

    break

plt.scatter(ra)