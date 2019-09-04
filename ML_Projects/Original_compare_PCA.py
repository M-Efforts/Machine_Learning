# 原始数据和经过PCA降维后的数据比较

# 分别导入numpy、pandas和matplotlib用于数学运算、作图以及数据分析
import numpy as np
import pandas as pd
from sklearn.svm import LinearSVC
from sklearn.decomposition import PCA
from sklearn.metrics import classification_report

# 使用pandas分别读取训练集和测试集
digits_train = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/optdigits/optdigits.tra', header=None)
digits_test = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/optdigits/optdigits.tes', header=None)

# 从训练集和测试集中分离出64维度的像素特征与1维度的数字目标
X_train = digits_train[np.arange(64)]
Y_train = digits_train[64]

X_test = digits_test[np.arange(64)]
Y_test = digits_test[64]

# 使用默认配置初始化LinearSVC，对原始64维像素特征的训练数据进行建模，并在测试数据上做出预测，存储在Y_predict中
svc = LinearSVC()
svc.fit(X_train, Y_train)
Y_predict = svc.predict(X_test)

# 使用PCA将原64维的图像数据压缩到20维
estimator = PCA(n_components=20)

# 利用训练特征决定(fit)20个正交维度的方向，并转化(transform)原训练特征
pca_X_train = estimator.fit_transform(X_train)
# 测试特征也按照上述的20个正交维度方向进行转化(transform)
pca_X_test = estimator.transform(X_test)

# 使用默认配置初始化LinearSVC，对压缩后的20维特征的训练数据进行建模，并在测试数据上做出预测，存储在pca_Y_predict中
pca_svc = LinearSVC()
pca_svc.fit(pca_X_train, Y_train)
pca_Y_predict = pca_svc.predict(pca_X_test)


# 性能评测
# 对使用原始图像高维像素特征训练的支持向量机分类器的性能做出评估
print(svc.score(X_test, Y_test))
print(classification_report(Y_test, Y_predict, target_names=np.arange(10).astype(str)))

# 对使用PCA压缩重建的低维图像特征训练的支持向量机分类器的性能做出评估
print(pca_svc.score(pca_X_test, Y_test))
print(classification_report(Y_test, pca_Y_predict, target_names=np.arange(10).astype(str)))
