import numpy as np
from Regression import LinearRegression
from numpy import genfromtxt
import matplotlib.pyplot as plt

dataset = genfromtxt('datasets/datasetENEI.csv', delimiter=',')

X = dataset[:, 3].reshape(-1, 1)
Y = dataset[:, 4].reshape(-1, 1)

plt.scatter(X, Y,  color='black')
plt.xticks(())
plt.yticks(())
plt.show()

lr = LinearRegression()
theta, costs = lr.train(X, Y)

print theta

plt.plot(costs, color='blue')
plt.show()

print lr.predict(np.array([0]))


plt.scatter(X, Y,  color='black')
plt.plot(X, lr.predict(X), color='blue', linewidth=3)
plt.xticks(())
plt.yticks(())

plt.show()