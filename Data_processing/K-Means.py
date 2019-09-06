from sklearn.cluster import KMeans
from sklearn import metrics
from sklearn.metrics.cluster import normalized_mutual_info_score
from sklearn.metrics import accuracy_score
# 导入数据分割工具
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt      # python画图包

X = np.genfromtxt(r"10维向量提取结果",
                  dtype=np.float32)
Y = np.genfromtxt(r'10维向量提取结果',
                  dtype=np.int32)
# label_txt = '乱序向量和标签'
# save_txt = '乱序向量和标签'
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3)

kmeans = KMeans(n_clusters=4)
kmeans.fit(X_train)
Y_pre = kmeans.predict(X_test)

# plt.scatter(X_test[:, 0], X_test[:, 1], c=Y_test)  # scatter绘制散点
# plt.title("Incorrect Number of Blobs")   # 加标题
# plt.show()  # 显示图
# save_folder = open(label_txt, "a+")
# for label in Y_pre:
#     save_folder.write(str(label) + '\n')
# save_folder.close()
# Y_pre = np.genfromtxt(label_txt, dtype=np.int32)
# save_folder1 = open(save_txt, "a+")
# save_folder1.write("AC:" + str(accuracy_score(Y, Y_pre)) + "\n")
# save_folder1.write("NMI:" + str(normalized_mutual_info_score(Y, Y_pre)) + "\n")
# save_folder1.close()
print(metrics.adjusted_rand_score(Y_test, Y_pre))
print(accuracy_score(Y_test, Y_pre))
print(normalized_mutual_info_score(Y_test, Y_pre))


