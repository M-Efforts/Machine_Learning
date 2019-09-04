# L2范数正则化

import numpy as np
# 导入L1范数模型
from sklearn.linear_model import Ridge, LinearRegression
# 导入多项式特征产生器
from sklearn.preprocessing import PolynomialFeatures

# 输入训练样本的特征以及目标值，分别存储在变量X_train和Y_train中
X_train = [[6], [8], [10], [14], [18]]
Y_train = [[7], [9], [13], [17.5], [18]]

# 准备测试数据
X_test = [[6], [8], [11], [16]]
Y_test = [[8], [12], [15], [18]]

# 使用4次多项式回归模型进行样本拟合
# 使用PolynominalFeatures(degree=4)映射出4次多项式特征，存储在变量X_train_poly4中
poly4 = PolynomialFeatures(degree=4)
X_train_poly4 = poly4.fit_transform(X_train)
X_test_poly4 = poly4.transform(X_test)
# 以线性回归器为基础，初始化回归模型。尽管特征的维度有提升，当时模型基础仍然是线性模型
regressor_poly4 = LinearRegression()
# 对4次多项式回归模型进行训练
regressor_poly4.fit(X_train_poly4, Y_train)
# 普通4次多项式回归模型过拟合之后的性能
print(regressor_poly4.score(X_test_poly4, Y_test))
print(regressor_poly4.coef_)
print('原始模型参数之间的差异：', np.sum(regressor_poly4.coef_**2))


# 使用默认配置初始化ridge
ridge_poly4 = Ridge()
# 使用ridge对4次多项式特征进行拟合
ridge_poly4.fit(X_train_poly4, Y_train)
# 对ridge模型在测试样本上的回归性能进行评估
print(ridge_poly4.score(X_test_poly4, Y_test))
# 输出ridge模型的参数列表
print(ridge_poly4.coef_)
# 输出上述参数的平方和，验证参数之间的巨大差异
print('使用L2范数正则化后模型参数之间的差异：', np.sum(ridge_poly4.coef_**2))
