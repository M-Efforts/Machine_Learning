# 回归预测

# 导入波士顿房价数据读取器
from sklearn.datasets import load_boston
# 导入numpy
import numpy as np
# 导入用于数据标准化模块
from sklearn.preprocessing import StandardScaler
# 导入数据分割工具
from sklearn.model_selection import train_test_split
# 导入LinearRegression
from sklearn.linear_model import LinearRegression
# 导入SGDRegressor
from sklearn.linear_model import SGDRegressor
# 导入回归性能评价指标
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
# 读取房价数据并存储在boston变量中
boston = load_boston()
print(boston.DESCR)

X = boston.data
Y = boston.target

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25, random_state=17)
print("The max target value is:", np.max(boston.target))
print("The min target value is:", np.min(boston.target))
print("The average target value is:", np.mean(boston.target))

ss_X = StandardScaler()

X_train = ss_X.fit_transform(X_train)
X_test = ss_X.transform(X_test)

lr = LinearRegression()
lr.fit(X_train, Y_train)
lr_Y_predict = lr.predict(X_test)

sgdr = SGDRegressor()
sgdr.fit(X_train, Y_train)
sgdr_Y_predict = sgdr.predict(X_test)

# 使用LinearRegression模型自带的评估模块，并输出评估结果
print("The value of default measurement of LinearRegression is:", lr.score(X_test, Y_test))
# 使用r2_score模块，并输出评估结果
print("The value of R-squared of LinearRegression is:", r2_score(Y_test, lr_Y_predict))
# 使用mean_squared_error模块，并输出评估结果
print("The mean squared error of LinearRegression is:", mean_squared_error(Y_test, lr_Y_predict))
# 使用mean_absolute_error模块，并输出评估结果
print("The mean absolute error of LinearRegression is:", mean_absolute_error(Y_test, lr_Y_predict))

# 使用SGDRegressor模型自带的评估模块，并输出评估结果
print("The value of default measurement of SGDRegressor is:", sgdr.score(X_test, Y_test))
# 使用r2_score模块，并输出评估结果
print("The value of R-squared of SGDRegressor is:", r2_score(Y_test, sgdr_Y_predict))
# 使用mean_squared_error模块，并输出评估结果
print("The mean squared error of SGDRegressor is:", mean_squared_error(Y_test, sgdr_Y_predict))
# 使用mean_absolute_error模块，并输出评估结果
print("The mean absolute error of SGDRegressor is:", mean_absolute_error(Y_test, sgdr_Y_predict))
