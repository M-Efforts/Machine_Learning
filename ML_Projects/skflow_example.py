# 使用skflow中内置的LinearRegressor、DNN以及Scikit-learn中集成的回归模型对“波士顿房价”数据进行回归预测


# 导入sklearn中的多个模块
from sklearn import datasets, metrics, preprocessing, model_selection
# 导入skflow
import skflow
from sklearn.ensemble import RandomForestRegressor

# 读取波士顿房价数据
boston = datasets.load_boston()

# 获取房屋数据特征及对应的房价
X, Y = boston.data, boston.target

# 分割训练和测试数据
X_train, X_test, Y_train, Y_test = model_selection.train_test_split(X, Y, test_size=0.35, random_state=17)

# 对数据特征进行标准化处理
scaler = preprocessing.StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 使用skflow的LinearRegressor
tf_lr = skflow.TensorFlowLinearRegressor(steps=10000, learning_rate=0.01, batch_size=50)
tf_lr.fit(X_train, Y_train)
tf_lr_Y_predict = tf_lr.predict(X_test)

# 输出skflow中LinearRgeressor模型的回归性能
print('The mean absolute error of Tensorflow Linear Regressor on boston dataset is', metrics.mean_absolute_error(tf_lr_Y_predict, Y_test))
print('The mean squared error of Tensorflow Linear Regressor on boston dataset is', metrics.mean_squared_error(tf_lr_Y_predict, Y_test))
print('The mean R-squared value of Tensorflow Linear Regressor on boston dataset is', metrics.r2_score(tf_lr_Y_predict, Y_test))


# 使用skflow的DNNRegressor,并且注意其每个隐层特征数量的配置
tf_dnn_regressor = skflow.TensorFlowDNNRegressor(hidden_units=[100, 40], steps=10000, learning_rate=0.01, batch_size=50)
tf_dnn_regressor.fit(X_train, Y_train)
tf_dnn_Y_predict = tf_dnn_regressor.predict(X_test)

# 输出skflow中DNNRegressor模型的回归性能
print('The mean absolute error of Tensorflow DNN Regressor on boston dataset is', metrics.mean_absolute_error(tf_dnn_Y_predict, Y_test))
print('The mean squared error of Tensorflow DNN Regressor on boston dataset is', metrics.mean_squared_error(tf_dnn_Y_predict, Y_test))
print('The mean R-squared value of Tensorflow DNN Regressor on boston dataset is', metrics.r2_score(tf_dnn_Y_predict, Y_test))


# 使用sklearn中的RandomForestRegressor
rfc = RandomForestRegressor()
rfc.fit(X_train, Y_train)
rfc_Y_predict = rfc.predict(X_test)

# 输出sklearn中RandomForestRegressor模型的回归性能
print('The mean absolute error of Sklearn Random Forest Regressor on boston dataset is', metrics.mean_absolute_error(rfc_Y_predict, Y_test))
print('The mean squared error of Sklearn Random Forest Regressor on boston dataset is', metrics.mean_squared_error(rfc_Y_predict, Y_test))
print('The mean R-squared value of Sklearn Random Forest Regressor on boston dataset is', metrics.r2_score(rfc_Y_predict, Y_test))
