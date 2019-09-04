# 回归树模型

# 导入波士顿房价数据读取器
from sklearn.datasets import load_boston
# 导入用于数据标准化模块
from sklearn.preprocessing import StandardScaler
# 导入数据分割工具
from sklearn.model_selection import train_test_split
# 导入决策树回归器
from sklearn.tree import DecisionTreeRegressor
# 导入回归性能评价指标
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# 读取房价数据并存储在boston变量中
boston = load_boston()

X = boston.data
Y = boston.target

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25, random_state=17)

# 树模型不要求对特征标准化核统一量化，即数值型和类别型的特征都可以
# ss_X = StandardScaler()
#
# X_train = ss_X.fit_transform(X_train)
# X_test = ss_X.transform(X_test)

# 初始化决策树分类器
dtr = DecisionTreeRegressor()
dtr.fit(X_train, Y_train)
dtr_Y_predict = dtr.predict(X_test)

print("R-squared value of DecisionTreeRegressor is:", dtr.score(X_test, Y_test))
# 使用mean_squared_error模块，并输出评估结果
print("The mean squared error of DecisionTreeRegressor is:", mean_squared_error(Y_test, dtr_Y_predict))
# 使用mean_absolute_error模块，并输出评估结果
print("The mean absolute error of DecisionTreeRegressor is:", mean_absolute_error(Y_test, dtr_Y_predict))
