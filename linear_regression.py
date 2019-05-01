from datetime import datetime
import random

class Linear_regression():
    def __init__(self, sourceData):
        self.sourceData = sourceData

    def train_test_split(self, seed = None):
        train = self.sourceData.sample(frac = 0.8, random_state = seed)
        test = self.sourceData.drop(train.index, axis = 0)

        # train = train.reset_index()
        y_train = train['PRICE']
        X_train = train['RM']
        
        # self.X_train = train.drop('PRICE', axis = 1)
        y_test = test['PRICE']
        X_test = test['RM']
        # self.X_test = test.drop('PRICE', axis = 1)
        return X_train, y_train, X_test, y_test
    
    def predict_y(self, X_predict, theta):
        return theta[1]*X_predict+theta[0]

    def costComputing(self, X_train, y_train, theta):
        sum = 0
        m = len(X_train)
        for i in range(m):
            sum += (self.predict_y(X_train[i], theta) - y_train[i])**2
        return (1/2*m) * sum

    def gradientDescent(self, X_train, y_train, theta, alpha, iterations):
        m = len(X_train)
        temp = [-1,-1]
        for _ in range(iterations):
            for i in range(2):
                sum = 0
                for j in range(m):
                    sum += (self.predict_y(X_train[j], theta) - y_train[j]) * X_train[j]
                temp[i] = theta[i] - alpha * (1/m) * sum
            theta = temp
        
        print(theta)