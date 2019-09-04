# 随机森立、梯度提升和极端随机森林回归树模型

import numpy as np
# 导入波士顿房价数据读取器
from sklearn.datasets import load_boston
# 导入数据分割工具
from sklearn.model_selection import train_test_split
# 导入随机森林、极端随机森林和梯度提升随机森林模型
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor
# 导入回归性能评价指标
from sklearn.metrics import mean_squared_error, mean_absolute_error

# 读取房价数据并存储在boston变量中
boston = load_boston()

X = boston.data
Y = boston.target

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25, random_state=17)

# 使用RandomForestRegressor训练模型，并对测试数据做出预测
rfr = RandomForestRegressor()
rfr.fit(X_train, Y_train)
rfr_Y_predict = rfr.predict(X_test)

# # 使用ExtraTreesRegressor训练模型，并对测试数据做出预测
etr = ExtraTreesRegressor()
etr.fit(X_train, Y_train)
etr_Y_predict = etr.predict(X_test)

# # 使用GradientBoostingRegressor训练模型，并对测试数据做出预测
gbr = GradientBoostingRegressor()
gbr.fit(X_train, Y_train)
gbr_Y_predict = gbr.predict(X_test)

print("R-squared value of RandomForestRegressor is:", rfr.score(X_test, Y_test))
# 使用mean_squared_error模块，并输出评估结果
print("The mean squared error of RandomForestRegressor is:", mean_squared_error(Y_test, rfr_Y_predict))
# 使用mean_absolute_error模块，并输出评估结果
print("The mean absolute error of RandomForestRegressor is:", mean_absolute_error(Y_test, rfr_Y_predict))

print("R-squared value of ExtraTreesRegressor is:", etr.score(X_test, Y_test))
# 使用mean_squared_error模块，并输出评估结果
print("The mean squared error of ExtraTreesRegressor is:", mean_squared_error(Y_test, etr_Y_predict))
# 使用mean_absolute_error模块，并输出评估结果
print("The mean absolute error of ExtraTreesRegressor is:", mean_absolute_error(Y_test, etr_Y_predict))
# 利用训练好的极端回归森林模型，输出每种特征对预测目标的贡献度(python3中zip需要强制转换为list才可以显示)
print(np.sort(list(zip(etr.feature_importances_, boston.feature_names)), axis=0))

print("R-squared value of GradientBoostingRegressor is:", gbr.score(X_test, Y_test))
# 使用mean_squared_error模块，并输出评估结果
print("The mean squared error of GradientBoostingRegressor is:", mean_squared_error(Y_test, gbr_Y_predict))
# 使用mean_absolute_error模块，并输出评估结果
print("The mean absolute error of GradientBoostingRegressor is:", mean_absolute_error(Y_test, gbr_Y_predict))
