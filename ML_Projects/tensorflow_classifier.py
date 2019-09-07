# 使用tensorflow自定一个线性分类器用于“良性/恶性乳腺癌肿瘤”预测


import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 读取乳腺癌肿瘤的训练和测试数据
train = pd.read_csv('training_set')
test = pd.read_csv('testing_set')

# 分割特征与分类目标
X_train = np.float32(train[['Clump Thickness', 'Cell Size']].T)
Y_train = np.float32(train['Type'].T)
X_test = np.float32(test[['Clump Thickness', 'Cell Size']].T)
Y_test = np.float32(test['Type'].T)

# 定义一个tensorflow的变量b作为线性模型的截距，同时设置初始值为1.0
b = tf.Variable(tf.zeros([1]))
# 定义一个tensorflow的变量W作为线性模型的参数，并设置初始值为-1.0到1.0之间的均匀分布的随机数据
W = tf.Variable(tf.random_uniform([1, 2], -1.0, 1.0))

# 显示定义这个线性函数
Y = tf.matmul(W, X_train) + b

# 使用tensorflow中的reduce_mean取得训练集上的均方误差
loss = tf.reduce_mean(tf.square(Y-Y_train))

# 使用梯度下降法估计参数W,b，并且设置迭代步长为0.01(学习率)，这个与Scikit-learn中的SGDRegressor模型类似
optimizer = tf.train.GradientDescentOptimizer(0.01)

# 以最小二乘损失为优化目标
train = optimizer.minimize(loss)

# 初始化所有变量
init = tf.initialize_all_variables()

# 开启tensorflow中的Session会话
sess = tf.Session()

# 执行变量初始化
sess.run(init)

# 迭代1000次，训练参数
for step in range(0, 1000):
    sess.run(train)
    if step % 200 == 0:
        print(step, sess.run(W), sess.run(b))

# 准备测试样本
test_negative = test.loc[test['Type'] == 0][['Clump Thickness', 'Cell Size']]
test_positive = test.loc[test['Type'] == 1][['Clump Thickness', 'Cell Size']]

# 以最终更新的参数作图
plt.scatter(test_negative['Clump Thickness'], test_negative['Cell Size'], marker='o', s=200, c='red')
plt.scatter(test_positive['Clump Thickness'], test_positive['Cell Size'], marker='x', s=150, c='black')

plt.xlabel('Clump Thickness')
plt.ylabel('Cell Size')
lx = np.arange(1, 12)

# 此处0.5作为分界线，因此计算方式如下
ly = (0.5-sess.run(b)-lx*sess.run(W)[0][0])/sess.run(W)[0][1]

plt.plot(lx, ly, color='green')
plt.show()
