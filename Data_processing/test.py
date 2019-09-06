from sklearn.preprocessing import MultiLabelBinarizer
import numpy as np
label_path = '不含[0]标签和第2560位置的标签数据'
# save_path = '非空标签数据'
# read_path = '非空标签数据'
# save_path1 = '非空标签数据'
read_path1 = '边表数据'
save_path1 = '边表数据'

label = np.load(label_path)
# p = []
# for la in labels:
#     if la:
#         print(la)
#         p.append(la)
# np.save(save_path, p)
# t = MultiLabelBinarizer().fit_transform(p)
# np.save(save_path1, t)
# for te in t:
#     print(str(te))


# save_folder = open(save_path1, "a+")
# for te in labels:
#     save_folder.write(str(te) + '\n')
# save_folder.close()


read_folder = open(read_path1, "r")
save_folder = open(save_path1, "a+")
labels = read_folder.read().splitlines()  # 将标签文件按行分割
# temp = label
temp = []
for i in range(len(label)):
    if not label[i]:
        # label = np.delete(label, i, axis=0)
        temp.append(i + 1)
        # print(i+1)
# np.save(save_path, label)
# j = 0
# while j < len(labels):
#     if int(labels[j].split("\t", 1)[0]) not in temp:
#         save_folder.write(labels[j] + '\n')
#         save_folder.write(labels[j+1] + '\n')
#     j += 2
for movie in labels:
    if int(movie.split("\t", 1)[0]) not in temp:
        save_folder.write(str(movie) + "\n")
save_folder.close()
read_folder.close()


