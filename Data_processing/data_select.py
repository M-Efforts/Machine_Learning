from random import shuffle

import numpy as np

train_txt = r'保留10维的所有节点向量文件'
test_txt = r'保留10维的作者节点向量文件'


label_folder = open(train_txt, "r")
save_folder = open(test_txt, "a+")

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

t = 0
for test in range(25473, 53715):
    if test in data_dict.keys():
        temp = str(data_dict[test]) + '\n'
        save_folder.write(temp)
    else:
        print(test)
        t += 1
        pass
print(t)
label_folder.close()
save_folder.close()
