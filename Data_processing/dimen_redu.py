from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np

X = np.genfromtxt('作者向量-有序',
                  dtype=np.float32)
label = np.genfromtxt('作者标签-有序',
                      dtype=np.int32)
save_folder = open("作者向量降维结果", "a+")


X_tsne = TSNE(n_components=2).fit_transform(X)
plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=label)
plt.show()
for tsne in X_tsne:
    temp = str(tsne) + '\n'
    save_folder.write(temp)
save_folder.close()
# print(type(X_tsne))
# plt.colorbar()
# plt.show()
