# K 邻近（回归）

# 导入iris数据加载器
from sklearn.datasets import load_iris
# 导入数据分割工具
from sklearn.model_selection import train_test_split
# 导入用于数据标准化模块
from sklearn.preprocessing import StandardScaler
# 导入k邻近回归器
from sklearn.neighbors import KNeighborsRegressor
# 导入回归性能评价指标
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# 使用加载器读取数据并存入变量iris
iris = load_iris()

X_train, X_test, Y_train, Y_test = train_test_split(iris.data, iris.target, test_size=0.35, random_state=17)

ss = StandardScaler()
X_train = ss.fit_transform(X_train)
X_test = ss.transform(X_test)

# 初始化K近邻回归器，并且调整配置，使得预测的方式为平均回归
uni_knr = KNeighborsRegressor(weights='uniform')
uni_knr.fit(X_train, Y_train)
uni_knr_Y_predict = uni_knr.predict(X_test)

# 初始化K近邻回归器，并且调整配置，使得预测的方式为根据距离加权回归
dis_knr = KNeighborsRegressor(weights='distance')
dis_knr.fit(X_train, Y_train)
dis_knr_Y_predict = dis_knr.predict(X_test)

print('The R-squared value of uniform-weighted KNeighborsRegression is:', uni_knr.score(X_test, Y_test))
# 使用mean_squared_error模块，并输出评估结果
print("The mean squared error of uniform-weighted KNeighborsRegression is:", mean_squared_error(Y_test, uni_knr_Y_predict))
# 使用mean_absolute_error模块，并输出评估结果
print("The mean absolute error of uniform-weighted KNeighborsRegression is:", mean_absolute_error(Y_test, uni_knr_Y_predict))

print('The R-squared value of distance-weighted KNeighborsRegression is:', dis_knr.score(X_test, Y_test))
# 使用mean_squared_error模块，并输出评估结果
print("The mean squared error of distance-weighted KNeighborsRegression is:", mean_squared_error(Y_test, dis_knr_Y_predict))
# 使用mean_absolute_error模块，并输出评估结果
print("The mean absolute error of distance-weighted KNeighborsRegression is:", mean_absolute_error(Y_test, dis_knr_Y_predict))
