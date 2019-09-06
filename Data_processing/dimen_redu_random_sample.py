from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np
import random

train_txt = '作者标签'
test_txt = '作者向量'
label_folder = open(train_txt, "r")
data_folder = open(test_txt, "r")

save_folder = open("降维向量", "a+")
input_label = "标签"
input_data = "数据"

label = label_folder.read().splitlines()
X = data_folder.read().splitlines()

lines = []
for i in range(len(label)):
    temp_line = str(label[i]) + " " + str(X[i])
    lines.append(temp_line + '\n')
line = random.sample(lines, 400)
data_folder = open(input_data, "a+")
label_folder = open(input_label, "a+")
for j in line:
    temp = j.split(" ", 1)
    label_folder.write(str(temp[0]) + '\n')
    data_folder.write(str(temp[1]))
label_folder.close()
data_folder.close()

input_data1 = np.genfromtxt('数据', dtype=np.float32)
input_label1 = np.genfromtxt('标签', dtype=np.int32)

X_tsne = TSNE(n_components=2, init='pca').fit_transform(input_data1)
plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=input_label1)
plt.show()
for tsne in X_tsne:
    temp = str(tsne) + '\n'
    save_folder.write(temp)
save_folder.close()
# print(type(X_tsne))
# plt.colorbar()
# plt.show()
