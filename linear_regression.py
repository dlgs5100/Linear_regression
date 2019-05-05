import numpy as np
import copy

class Linear_regression():
    def __init__(self, sourceData):
        self.sourceData = sourceData

    def train_test_split(self, seed = None):
        train = self.sourceData.sample(frac = 0.8, random_state = seed)
        test = self.sourceData.drop(train.index, axis = 0)

        y_train = train['PRICE'].values
        X_train = train['RM'].values
        # X_train = train.drop('PRICE', axis = 1)
        
        y_test = test['PRICE'].values
        X_test = test['RM'].values
        # X_test = test.drop('PRICE', axis = 1)
        return X_train, y_train, X_test, y_test

    def costComputing(self, X_train, y_train, theta):
        m = np.size(y_train)
        return (1 / (2 * m)) * sum(np.power(X_train.dot(theta) - y_train, 2));

    def gradientDescent(self, X_train, y_train, theta, alpha, iterations):
        temp = np.zeros((np.size(theta), 1))
        iterTheta = []

        m = np.size(y_train)
        for _ in range(iterations):
            for indexTheta in range(np.size(theta)):
                #--Gradient Descent formula--#
                # Theta i = Theta i - alpha * 1/m * sigma(1->m)[(h(xi) - yi) * xi]
                # Nxm dot mx1 = Nx1 (solve the h(xi) equation: y = theta 0 + theta i * xi)
                temp[indexTheta] = theta[indexTheta] - (1/m) * alpha * np.sum(np.transpose(X_train.dot(theta) - y_train) * X_train[: , indexTheta])
            theta = temp
            iterTheta.append(theta.copy())
        return theta, iterTheta
