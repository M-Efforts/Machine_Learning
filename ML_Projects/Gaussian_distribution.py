# 高斯分布（正态分布）

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
import seaborn as sns

mu_params = [-1, 0, 1]
sd_params = [0.5, 1, 1.5]
x = np.linspace(-7, 7, 100)  # 返回一个等差数列，start-起始点，stop-结束点，num-元素个数，默认为50
# f表示整体画布，ax表示子图集合
f, ax = plt.subplots(len(mu_params), len(sd_params), sharex=True, sharey=True)
for i in range(3):
    for j in range(3):
        mu = mu_params[i]
        sd = sd_params[j]
        y = stats.norm(mu, sd).pdf(x)  # norm函数 可以实现正态分布；pdf : 概率密度函数
        ax[i, j].plot(x, y)  # 绘制正态分布曲线
        # 在第i行j列的子图上绘制mu和sigma的值（占3位，其中两位小数），透明度为0
        ax[i, j].plot(0, 0, label="$\\mu$={:3.2f}\n$\\sigma$={:3.2f}".format(mu, sd), alpha=0)
        # 设置子图中文字格式
        ax[i, j].legend(fontsize=12)
ax[2, 1].set_xlabel('$x$', fontsize=16)
ax[1, 0].set_ylabel('$pdf(x)$', fontsize=16)
plt.tight_layout()
plt.show()
