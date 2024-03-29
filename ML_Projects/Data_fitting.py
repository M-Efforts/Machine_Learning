# 模型的欠拟合，拟合，过拟合

# 使用线性回归模型在披萨训练样本上进行拟合
# 导入线性回归模型
from sklearn.linear_model import LinearRegression
# 导入多项式特征产生器
from sklearn.preprocessing import PolynomialFeatures

import numpy as np
import matplotlib.pyplot as plt

# 输入训练样本的特征以及目标值，分别存储在变量X_train和Y_train中
X_train = [[6], [8], [10], [14], [18]]
Y_train = [[7], [9], [13], [17.5], [18]]


# 使用线性回归模型拟合数据
# 使用默认配置初始化线性回归模型
regressor = LinearRegression()
# 直接以披萨的直径作为特征训练模型
regressor.fit(X_train, Y_train)
# 在X轴上从0到25均匀采样100个数据点
xx = np.linspace(0, 26, 100)
xx = xx.reshape(xx.shape[0], 1)
# 以上述100个数据点作为基准，预测回归直线
yy = regressor.predict(xx)


# 使用2次多项式回归模型进行样本拟合
# 使用PolynominalFeatures(degree=2)映射出2次多项式特征，存储在变量X_train_poly2中
poly2 = PolynomialFeatures(degree=2)
X_train_poly2 = poly2.fit_transform(X_train)
# 以线性回归器为基础，初始化回归模型。尽管特征的维度有提升，当时模型基础仍然是线性模型
regressor_poly2 = LinearRegression()
# 对2次多项式回归模型进行训练
regressor_poly2.fit(X_train_poly2, Y_train)
# 从新映射绘图用x轴采样数据
xx_poly2 = poly2.transform(xx)
# 使用2次多项式回归模型对应x轴采样数据进行回归预测
yy_poly2 = regressor_poly2.predict(xx_poly2)


# 使用4次多项式回归模型进行样本拟合
# 使用PolynominalFeatures(degree=4)映射出4次多项式特征，存储在变量X_train_poly4中
poly4 = PolynomialFeatures(degree=4)
X_train_poly4 = poly4.fit_transform(X_train)
# 以线性回归器为基础，初始化回归模型。尽管特征的维度有提升，当时模型基础仍然是线性模型
regressor_poly4 = LinearRegression()
# 对4次多项式回归模型进行训练
regressor_poly4.fit(X_train_poly4, Y_train)
# 从新映射绘图用x轴采样数据
xx_poly4 = poly4.transform(xx)
# 使用4次多项式回归模型对应x轴采样数据进行回归预测
yy_poly4 = regressor_poly4.predict(xx_poly4)


# 分别对训练数据点、线性回归直线、2次多项式以及4次多项式回归曲线进行作图
plt.scatter(X_train, Y_train)
plt1, = plt.plot(xx, yy, label='degree=1')
plt2, = plt.plot(xx, yy_poly2, label='degree=2')
plt4, = plt.plot(xx, yy_poly4, label='degree=4')

plt.axis([0, 25, 0, 25])
plt.xlabel('Diameter of Pizza')
plt.ylabel('Price of Pizza')
plt.legend(handles=[plt1, plt2, plt4])
plt.show()


# 输出线性回归模型在训练样本上的R-squared值
print('The R-squared value of Linear Regressor performing on the training data is:', regressor.score(X_train, Y_train))

# 输出2次多项式回归模型在训练样本上的R-squared值
print('The R-squared value of Polynominal Regressor (degree=2) performing on the training data is',
      regressor_poly2.score(X_train_poly2, Y_train))

# 输出4次多项式回归模型在训练样本上的R-squared值
print('The R-squared value of Polynominal Regressor (degree=4) performing on the training data is',
      regressor_poly4.score(X_train_poly4, Y_train))

# 评估3种回归模型在测试数据上的性能表现
# 准备测试数据
X_test = [[6], [8], [11], [16]]
Y_test = [[8], [12], [15], [18]]

# 使用测试数据对线性回归模型的性能进行评估
print(regressor.score(X_test, Y_test))

# 使用测试数据对2次多项式回归模型的性能进行评估
X_test_poly2 = poly2.transform(X_test)
print(regressor_poly2.score(X_test_poly2, Y_test))

# 使用测试数据对4次多项式回归模型的性能进行评估
X_test_poly4 = poly4.transform(X_test)
print(regressor_poly4.score(X_test_poly4, Y_test))
