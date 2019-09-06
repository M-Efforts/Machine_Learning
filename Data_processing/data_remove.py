train_txt = '所有作者对应标签文件'
test_txt = '所有作者对应标签文件（去重）'


def load_label_set(label_dir, data_dir):
    label_folder = open(label_dir, "r")
    save_folder = open(data_dir, "a+")
    trainlines = label_folder.read().splitlines()  # 返回每一行的数据
    data_dict = {}
    for line in trainlines:
        line = line.split("\t", 2)  # 按照空格键分割每一行里面的数据
        if line[1] not in data_dict or int(data_dict[line[1]]) < int(line[2]):
            data_dict[line[1]] = line[2]

    l_data = list(data_dict.keys())

    for i in l_data:
        temp = i + "\t" + data_dict[i] + '\n'
        save_folder.write(temp)
        # box = [float(line[0]), float(line[1]), float(line[2]), float(line[3])]#box读取标签ground_truth
    label_folder.close()
    save_folder.close()
    # return  train_box

# train_box = load_train_test_set(test_txt)


if __name__ == '__main__':
    load_label_set(train_txt, test_txt)
