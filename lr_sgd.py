"""
Build linear regression with Stochastic Gradient Descent method.
"""
import numpy as np
import random
import matplotlib.pyplot as plt


def single_partial(X, y, _theta, attr):
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

alpha = 0.01  # typically we set it to be O(1 / sqrt(t))
m = 3  # number of instances
n = 3  # number of attributes
theta = [0 for i in range(3)]

loss_all = []
for i in range(1000):
    ind = random.randint(0, m - 1)
    theta_1 = theta[:]
    for j in range(m):
        theta[j] -= alpha * (np.dot(theta_1, X[ind]) - y[ind]) * X[ind][j]
    loss_all.append(loss_function(X, y, theta))
print(theta)

plt.scatter(range(1000), loss_all, s=1)
plt.show()
