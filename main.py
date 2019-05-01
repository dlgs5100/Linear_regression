from linear_regression import Linear_regression
from sklearn import datasets
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt


def main():
    
    sourceData = inputDataset()
    linear_regression = Linear_regression(sourceData)
    train, test = linear_regression.train_test_split()

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