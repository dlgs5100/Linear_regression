from linear_regression import Linear_regression
from sklearn import datasets
import time
import os
import pandas as pd 
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib import animation

def main():
    sourceData = inputDataset()
    #--價位區間圖--#
    # plotPriceRange(sourceData)
    #--相關性熱點圖--#
    # plotPriceRelevance(sourceData)
    deletePreviousOutputFile()

    linear_regression = Linear_regression(sourceData)
    X_train, y_train, X_test, y_test = linear_regression.train_test_split()
    
    oneX_train = np.concatenate((np.ones((len(X_train), 1)), X_train), axis = 1)
    oneX_test = np.concatenate((np.ones((len(X_test), 1)), X_test), axis = 1)

    iterations = 500
    dfResult = pd.DataFrame(columns=['RMSE', 'R2_score'])
    
    theta = np.zeros((len(X_train[0])+1,1))
    alpha = 1.0

    print('Cost:', linear_regression.costComputing(oneX_train, y_train, theta))

    start = time.time()
    theta, iterTheta = linear_regression.gradientDescent(oneX_train, y_train, theta, alpha, iterations)
    end = time.time()

    RMSE = calcRMSE(theta, oneX_test, y_test)
    R2_score = calcR2_score(theta, oneX_test, y_test)
    dfResult.loc[str(alpha)] = [RMSE, R2_score]
    outputResult(alpha, RMSE, R2_score, end-start)
    #--單一屬性與價格(二維)收斂動畫--#
    # plotAnimation(iterTheta, X_train, y_train)
    plotResult(dfResult)

def calcRMSE(theta, X_test, y_test):
    N = np.size(y_test)
    return np.sqrt(sum(np.square(X_test.dot(theta) - y_test))/N)[0]

def calcR2_score(theta, X_test, y_test):
    return 1 - (sum(np.square(X_test.dot(theta) - y_test)) / sum(np.square(y_test - y_test.mean())))[0]

def deletePreviousOutputFile():
    try:
        os.remove('result.txt')
    except OSError as e:
        None

def outputResult(alpha, RMSE, R2_score, time):
    with open('result.txt', 'a') as file:
        file.write("Alpha: {}\n".format(alpha))
        file.write("RMSE: {}\n".format(RMSE))
        file.write("R2_score: {}\n".format(R2_score))
        file.write("Time: {} s\n".format(time))
        file.write('*--------------------------*\n')
        file.close()

def plotResult(dfResult):
    dfResult.plot(x=None, y=['RMSE', 'R2_score'], kind='bar')
    sns.set(rc={'figure.figsize':(8.7,5.27)}) 
    plt.xlabel('Alpha')
    plt.ylim(-50,50)
    plt.savefig("RMSE&R2_score.png")

def plotPriceRange(sourceData):
    sns.set(rc={'figure.figsize':(8.7,5.27)}) 
    plt.hist(sourceData['PRICE'], bins=30) 
    plt.xlabel("House prices in $1000")
    plt.ylabel("House amount")
    plt.savefig("Price range.png")
    plt.show()

def plotPriceRelevance(sourceData):
    correlation_matrix = sourceData.corr().round(2) 
    sns.set(rc={'figure.figsize':(10.7,7.27)}) 
    sns.heatmap(data=correlation_matrix, annot=True)
    plt.savefig("Price relevance.png")
    plt.show()

def plotAnimation(iterTheta, X_train, y_train):
    fig, ax = plt.subplots()
    ax.plot(X_train, y_train, 'rx')
    line, = ax.plot([], [], '-')

    startX = np.array([1,-1000])
    endX = np.array([1,1000])
    def animate(i):
        newx = [-1000, 1000]
        newy = [startX.dot(iterTheta[i])[0], endX.dot(iterTheta[i])[0]]
        line.set_data(newx, newy)
        return line
    def init():
        newx = [-1000, 1000]
        newy = [startX.dot(iterTheta[0])[0], endX.dot(iterTheta[0])[0]]
        line.set_data(newx, newy)
        return line

    ani = animation.FuncAnimation(fig=fig, func=animate, frames=20, init_func=init, interval=80, blit=False)
    ani.save("Animation.html",writer='pillow')
    # plt.show()

def inputDataset():
    boston = datasets.load_boston()
    bos = pd.DataFrame(boston.data, columns = boston.feature_names) 
    bos['PRICE'] = boston.target 
    return bos

if __name__ == '__main__':
    main()  