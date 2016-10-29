# coding=utf-8

import Optimization as opt
import numpy as np
import random
from numpy import linalg as la


__author__ = "Mário Antunes"
__copyright__ = "Copyright 2016, ENEI Workshop"
__license__ = "MIT"
__version__ = "1.0"
__email__ = "mariolpantunes@gmail.com"


class LinearRegression:

    def __init__(self):
        self.theta = []
        self.X = 0
        self.Y = 0

    def cost_function(self, theta):
        predictions = np.dot(self.X, theta).reshape(-1, 1)
        diff = np.subtract(predictions, self.Y) ** 2.
        cost = np.sum(diff) / (2. * self.X.shape[0])
        return cost

    def gradient(self, theta):
        predictions = np.dot(self.X, theta).reshape(-1, 1)
        diff = np.subtract(predictions, self.Y)
        gradient = (np.dot(np.transpose(self.X), diff)).reshape(1, -1)/self.X.shape[0]
        return gradient[0]

    def train(self, x, y, alpha=0.01, max_it=100):
        self.X = np.insert(x, 0, 1., axis=1)
        self.Y = y
        self.theta = np.zeros(self.X.shape[1])
        theta, costs = opt.gradient_descent(self.theta, self.gradient, self.cost_function, alpha=alpha, max_it=max_it)
        self.theta = theta
        return theta, costs

    def predict(self, points):
        points = points.reshape(-1, 1)
        points = np.insert(points, 0, 1., axis=1)
        return np.dot(points, self.theta)
