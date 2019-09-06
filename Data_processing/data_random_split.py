from random import shuffle
train_txt = '作者标签文件'
temp_txt = '作者向量文件（有序）'
test_txt = '作者向量-标签文件（有序）'


def load_label_set():
    label_folder = open(train_txt, "r")
    data_folder = open(temp_txt, "r")
    save_folder = open(test_txt, "a+")
    label_lines = label_folder.read().splitlines()  # 返回每一行的数据
    data_lines = data_folder.read().splitlines()  # 返回每一行的数据

    # data_dict = {}
    line = []
    for i in range(len(label_lines)):
        # print(str(i) + " " + label_lines[i] + " " + data_lines[i])
        line.append(label_lines[i] + " " + data_lines[i] + '\n')     # 按照空格分割每一行里面的数据
    shuffle(line)
    for temp in line:
        save_folder.write(temp)
        # data_dict[float(line[0])] = line[1]

    # test = range(25473, 53715)
    # for test1 in test:
    #     temp = str(test1) + " " + str(data_dict[test1]) + '\n'
    #     save_folder.write(temp)
        # box = [float(line[0]), float(line[1]), float(line[2]), float(line[3])]#box读取标签ground_truth
    label_folder.close()
    data_folder.close()
    save_folder.close()
    # return  train_box

# train_box = load_train_test_set(test_txt)


if __name__ == '__main__':
    load_label_set()
