# KMeans聚类并使用ARI评价指标

# 分别导入numpy、pandas和matplotlib用于数学运算、作图以及数据分析
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# 导入KMeans模型
from sklearn.cluster import KMeans
# 导入度量函数库metrics
from sklearn import metrics

# 使用pandas分别读取训练集和测试集
digits_train = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/optdigits/optdigits.tra', header=None)
digits_test = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/optdigits/optdigits.tes', header=None)

# 从训练集和测试集中分离出64维度的像素特征与1维度的数字目标
X_train = digits_train[np.arange(64)]
Y_train = digits_train[64]

X_test = digits_test[np.arange(64)]
Y_test = digits_test[64]

# 初始化KMeans模型，并设置聚类中心数量为10
kmeans = KMeans(n_clusters=10)
kmeans.fit(X_train)
# 逐条判断每个测试图像所属的聚类中心
Y_predict = kmeans.predict(X_test)

# 使用ARI进行KMeans聚类性能评估
print(metrics.adjusted_rand_score(Y_test, Y_predict))
