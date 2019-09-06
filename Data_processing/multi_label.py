from sklearn.preprocessing import MultiLabelBinarizer
import numpy as np

file_path = '电影类型对应标签文件'
save_path = '独热编码的电影类型标签文件'

read_folder = open(file_path, "r")   # 读取原始标签文件
labels = read_folder.read().splitlines()  # 将标签文件按行分割
# 将标签和类别ID对应
label_type = {122714: 1, 122715: 2, 122716: 3, 122717: 4, 122718: 5,
              122719: 6, 122720: 7, 122721: 8, 122722: 9, 122723: 10,
              122724: 11, 122725: 12, 122726: 13, 122727: 14, 122728: 15,
              122729: 16, 122730: 17, 122731: 18, 122732: 19, 122733: 20}

movies = []   # 保存电影ID
types = []    # 保存类型ID
# 循环分割每一行，分别保存电影ID和类型ID
for lines in labels:
    movies.append(lines.split("\t", 1)[0])
    types.append(lines.split("\t", 1)[1])

flag = 1
label_types = []
temp = []
for i in range(len(movies)):
    if int(movies[i]) == flag:
        temp.append(label_type[int(types[i])])
    elif i == len(movies) - 1:
        label_types.append(temp)
        temp = [label_type[int(types[i])]]
        label_types.append(temp)
    else:
        label_types.append(temp)
        temp = [label_type[int(types[i])]]
    flag = int(movies[i])

print(flag)
t = MultiLabelBinarizer().fit_transform(label_types)
np.save(save_path, t)
read_folder.close()
# p = np.load(save_path)
