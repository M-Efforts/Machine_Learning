from random import shuffle

import numpy as np

train_txt = '所有节点向量抽取结果'
test_txt = '所有节点向量抽取结果的有序排列'
read_path = '所有标签的独热编码'
save_path = '所有标签的独热编码有序排列（npy格式）'
temp_txt = '所有标签的独热编码有序排列(txt格式)'


p = np.load(read_path)
label_folder = open(train_txt, "r")
save_folder = open(test_txt, "a+")
save_folder1 = open(temp_txt, "a+")

# trainlines = label_folder.read().splitlines()  # 返回每一行的数据
# trainlines1 = label_folder1.read().splitlines()  # 返回每一行的数据
# for i in range(len(trainlines1)):
#     temp = str(trainlines[i]) + " " + str(trainlines1[i]) + '\n'
#     save_folder.write(temp)
#
# label_folder.close()
# label_folder1.close()
# save_folder.close()

trainlines = label_folder.read().splitlines()  # 返回每一行的数据
data_dict = {}
for line in trainlines:
    line = line.split(" ", 1)  # 按照空格分割每一行里面的数据
    data_dict[float(line[0])] = line[1]

test = range(1, 10198)
print(p.shape)
f = 1
for test1 in test:
    if test1 in data_dict.keys():
        temp = str(data_dict[test1]) + '\n'
        temp1 = str(p[test1-f]) + "\n"
        save_folder.write(temp)
        save_folder1.write(temp1)
    else:
        f += 1
        p = np.delete(p, test1-1, axis=0)
        print(p.shape)
np.save(save_path, p)
label_folder.close()
save_folder.close()
save_folder1.close()
