from linear_regression import Linear_regression
from sklearn import datasets
import pandas as pd 
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib import animation

def main():
    
    alpha = 1.005
    theta = np.zeros((2,1))
    iterations = 10

    sourceData = inputDataset()
    linear_regression = Linear_regression(sourceData)
    X_train, y_train, X_test, y_test = linear_regression.train_test_split()

    X_train = X_train.reshape(len(X_train), 1)
    y_train = y_train.reshape(len(y_train), 1)
    oneX = np.concatenate((np.ones((np.size(X_train), 1)), X_train), axis = 1)
    resultTheta = linear_regression.gradientDescent(oneX, y_train, theta, alpha, iterations)

    print(resultTheta)
    # plt.plot(X_train, y_train, 'rx')
    # plt.plot(oneX[:,1], oneX.dot(theta), '-')
    # plt.xlabel('RM')
    # plt.ylabel('Price')
    # plt.show()
    fig, ax = plt.subplots()

    x = oneX[:,1]
    ax.plot(X_train, y_train, 'rx')
    line, = ax.plot(oneX[:,1], oneX.dot(resultTheta[-1]), '-')

    def animate(i):
        line.set_data(oneX[:,1], oneX.dot(resultTheta[i]))
        return line
    def init():
        line.set_data(oneX[:,1], oneX.dot(resultTheta[0]))
        return line

    ani = animation.FuncAnimation(fig=fig,  func=animate,  frames=9, init_func=init, interval=50, blit=False)
    plt.show()

    

    #--價位區間圖--#
    # sns.set(rc={'figure.figsize':(8.7,5.27)}) 
    # plt.hist(bos['PRICE'], bins=30) 
    # plt.xlabel("House prices in $1000") 
    # plt.show()

    #--相關性熱點圖--#
    # bos = pd.DataFrame(boston.data, columns = boston.feature_names) 
    # bos['PRICE'] = boston.target 
    # correlation_matrix = bos.corr().round(2) 
    # sns.set(rc={'figure.figsize':(8.7,5.27)}) 
    # sns.heatmap(data=correlation_matrix, annot=True)
    # plt.show()

def  inputDataset():
    boston = datasets.load_boston()
    bos = pd.DataFrame(boston.data, columns = boston.feature_names) 
    bos['PRICE'] = boston.target 
    return bos

if __name__ == '__main__':
    main()  