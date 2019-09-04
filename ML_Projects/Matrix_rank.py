# 计算矩阵的秩

import numpy as np
# 初始化一个2*2的线性相关矩阵
M = np.array([[1, 2], [2, 4]])
# 计算2*2线性相关矩阵
rank = np.linalg.matrix_rank(M, tol=None)
print(rank)
