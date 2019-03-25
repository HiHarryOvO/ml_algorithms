import pandas as pd
import numpy as np
import lr_sgd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


df = pd.read_csv('kc_house_data.csv')
y = df['price'][:]
y = np.divide(np.subtract(y, np.mean(y)), max(y) - min(y))

df = df.drop(['id', 'date', 'zipcode', 'lat', 'long', 'price', 'yr_renovated'], axis=1)

for index in df.columns:
    df[index] = np.divide(np.subtract(df[index], np.mean(df[index])), max(df[index]) - min(df[index]))

df['x_0'] = pd.Series([1 for i in range(0, df.shape[0])])

X = df.values.tolist()
y = y.tolist()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

m = len(X_train)  # number of instances
n = df.shape[1]  # number of attributes
theta = [0 for i in range(n)]

loss_all = []
print(type(X_train))
print(type(y_train))

print(theta)

list_mse = []
alphas = [0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1]
alpha = 0.1
test_n = len(X_test)

for epsilon in [0.5, 1, 2, 4]:
    # lr_sgd.sgd(X_train, y_train, theta, alpha, iter_times=700)
    lr_sgd.duchi_sgd(X_train, y_train, theta, epsilon=epsilon, alpha=0.3, iter_times=700)
    mse = 0
    for data, data_y in zip(X_test, y_test):
        mse += (1 / test_n) * (np.dot(theta, data) - data_y) ** 2
    list_mse.append(mse)

rmse = np.sqrt(list_mse)
print("RMSE =", rmse)
plt.plot([0.5, 1, 2, 4], rmse)
plt.show()
