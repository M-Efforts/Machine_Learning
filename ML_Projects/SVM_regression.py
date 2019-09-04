# 支持向量机回归

# 导入手写体数字加载器
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
# 导入用于数据标准化模块
from sklearn.preprocessing import StandardScaler
# 导入支持向量机（回归）模型
from sklearn.svm import SVR
# 导入回归性能评价指标
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

digits = load_digits()
digits.data.shape

X_train, X_test, Y_train, Y_test = train_test_split(digits.data, digits.target, test_size=0.25, random_state=17)
ss = StandardScaler()
X_train = ss.fit_transform(X_train)
X_test = ss.transform(X_test)

# 使用线性核函数配置的支持向量机进行回归训练，并且对测试样本进行预测
linear_svr = SVR(kernel='linear')
linear_svr.fit(X_train, Y_train)
linear_svr_Y_predict = linear_svr.predict(X_test)

# 使用多项式核函数配置的支持向量机进行回归训练，并且对测试样本进行预测
poly_svr = SVR(kernel='poly')
poly_svr.fit(X_train, Y_train)
poly_svr_Y_predict = linear_svr.predict(X_test)

# 使用径向基核函数配置的支持向量机进行回归训练，并且对测试样本进行预测
rbf_svr = SVR(kernel='rbf')
rbf_svr.fit(X_train, Y_train)
rbf_svr_Y_predict = linear_svr.predict(X_test)

print('The R-squared value of Linear SVR is:', linear_svr.score(X_test, Y_test))
# 使用mean_squared_error模块，并输出评估结果
print("The mean squared error of Linear SVR is:", mean_squared_error(Y_test, linear_svr_Y_predict))
# 使用mean_absolute_error模块，并输出评估结果
print("The mean absolute error of Linear SVR is:", mean_absolute_error(Y_test, linear_svr_Y_predict))

print('The R-squared value of Poly SVR is:', poly_svr.score(X_test, Y_test))
# 使用mean_squared_error模块，并输出评估结果
print("The mean squared error of Poly SVR is:", mean_squared_error(Y_test, poly_svr_Y_predict))
# 使用mean_absolute_error模块，并输出评估结果
print("The mean absolute error of Poly SVR is:", mean_absolute_error(Y_test, poly_svr_Y_predict))

print('The R-squared value of RBF SVR is:', rbf_svr.score(X_test, Y_test))
# 使用mean_squared_error模块，并输出评估结果
print("The mean squared error of RBF SVR is:", mean_squared_error(Y_test, rbf_svr_Y_predict))
# 使用mean_absolute_error模块，并输出评估结果
print("The mean absolute error of RBF SVR is:", mean_absolute_error(Y_test, rbf_svr_Y_predict))
