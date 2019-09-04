# PCA降维

# 分别导入numpy、pandas和matplotlib用于数学运算、作图以及数据分析
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# 导入KMeans模型
from sklearn.decomposition import PCA
# 导入度量函数库metrics
from sklearn import metrics

# 使用pandas分别读取训练集和测试集
digits_train = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/optdigits/optdigits.tra',
                           header=None)
digits_test = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/optdigits/optdigits.tes',
                          header=None)

# 从训练集和测试集中分离出64维度的像素特征与1维度的数字目标
X_digits = digits_train[np.arange(64)]
Y_digits = digits_train[64]

# 初始化一个可以将高纬度特征向量压缩至2维的PCA模型
estimator = PCA(n_components=2)
X_pca = estimator.fit_transform(X_digits)


def plot_pca_scatter():
    colors = ['black', 'blue', 'purple', 'yellow', 'white', 'red', 'lime', 'cyan', 'orange', 'gray']
    for i in range(len(colors)):
        px = X_pca[:, 0][Y_digits.as_matrix() == i]
        py = X_pca[:, 1][Y_digits.as_matrix() == i]
        plt.scatter(px, py, c=colors[i])
    plt.legend(np.arange(0, 10).astype(str))
    plt.xlabel('First Principal Component')
    plt.ylabel('Second Principal Component')
    plt.show()


plot_pca_scatter()
