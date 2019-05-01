from datetime import datetime
import random

class Linear_regression():
    def __init__(self, sourceData):
        self.sourceData = sourceData

    def train_test_split(self, seed = None):
        train = self.sourceData.sample(frac = 0.8, random_state = seed)
        test = self.sourceData.drop(train.index, axis = 0)

        y_train = train['PRICE']
        X_train = train.drop('PRICE', axis = 1)
        y_test = test['PRICE']
        X_test = test.drop('PRICE', axis = 1)
        
        return train, test 
