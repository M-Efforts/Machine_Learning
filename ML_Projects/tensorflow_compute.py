# 使用tensorflow工具进行线性函数计算


import tensorflow as tf

# 声明matrix1为tensorflow的一个1*2的行向量
matrix1 = tf.constant([[3., 3.]])
# 声明matrix2为tensorflow的一个2*1的列向量
matrix2 = tf.constant([[2.], [2.]])

# 将上述两个算子相乘，作为新算例
product = tf.matmul(matrix1, matrix2)

# 继续使用相乘结果product与常量2.0求和拼接
linear = tf.add(product, tf.constant(2.0))

# 直接在会话中执行linear算例，相当于将上述所有单独的算例拼接为一个流程
with tf.compat.v1.Session() as sess:
    result = sess.run(linear)
    print(result)
