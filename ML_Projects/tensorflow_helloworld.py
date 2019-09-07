# 使用tensorflow工具输出字符串（测试示例）

# 导入tensorflow工具包并重命名为tf
import tensorflow as tf
import numpy as np

# 初始化一个Tensorflow常量字符串，并命名为greeting作为一个计算模块
greeting = tf.constant('Hello world!')

# 启动一个会话
sess = tf.compat.v1.Session()
# 使用会话执行greeting计算模块
result = sess.run(greeting)
# 输出会话执行的结果
print(result)
# 关闭会话，这是一种显示关闭会话的方式
# with tf.Session() as sess:
#   code paragraph
# 这种方式会在代码段执行结束之后自动关闭会话
