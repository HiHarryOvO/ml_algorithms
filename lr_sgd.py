"""
Build linear regression with Stochastic Gradient Descent method.
"""
import numpy as np
import random
import math
import pandas as pd
import new_algorithms
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


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


def sgd(X, y, theta, alpha=1, iter_times=100):
    for i in range(iter_times):
        ind = random.randint(0, len(X) - 1)
        theta_1 = theta[:]
        for j in range(len(theta)):
            theta[j] -= alpha * (np.dot(theta_1, X[ind]) - y[ind]) * X[ind][j]
        # loss_all.append(loss_function(X_train, y_train, theta))


def duchi_sgd(X, y, theta, epsilon, alpha=1, iter_times=100):
    for i in range(iter_times):
        ind = random.randint(0, len(X) - 1)
        theta_1 = theta[:]
        gradients = []
        for j in range(len(theta)):
            gradients.append(alpha * (np.dot(theta_1, X[ind]) - y[ind]) * X[ind][j])
        private_gradients = new_algorithms.duchi_method(gradients, epsilon)
        theta = np.subtract(theta, private_gradients)
        # loss_all.append(loss_function(X_train, y_train, theta))


if __name__ == "__main__":
    df = pd.read_csv('kc_house_data.csv')
    y = df['price'][:]
    y = np.divide(np.subtract(y, np.mean(y)), max(y) - min(y))

    df = df.drop(['id', 'date', 'zipcode', 'lat', 'long', 'price'], axis=1)

    for index in df.columns:
        df[index] = np.divide(np.subtract(df[index], np.mean(df[index])), max(df[index]) - min(df[index]))

    df['x_0'] = pd.Series([1 for i in range(0, df.shape[0])])

    X = df.values.tolist()
    y = y.tolist()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

    alpha = 0.01  # typically we set it to be O(1 / sqrt(t))
    m = len(X_train)  # number of instances
    n = df.shape[1]  # number of attributes
    theta = [0 for i in range(n)]

    print(type(X_train))
    print(type(y_train))

    sgd(X_train, y_train, theta, alpha=0.001, iter_times=100)

    print(theta)

    mse = 0
    test_n = len(X_test)
    for data, data_y in zip(X_test, y_test):
        mse += (1 / test_n) * (np.dot(theta, data) - data_y) ** 2
    rmse = math.sqrt(mse)
    print("RMSE =", rmse)
    # plt.scatter(range(200), loss_all, s=1)
    # plt.show()
