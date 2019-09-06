train_txt = '所有节点向量映射-无序'
test_txt = '所有节点向量映射-有序'
temp_txt = '所有向量-有序'

read_txt = "作者向量降维结果（带[]符号的字符串）"
save_txt = "作者向量降维结果（去除[]符号）"


def load_label_set(label_dir, data_dir):
    label_folder = open(label_dir, "r")
    save_folder = open(data_dir, "a+")
    trainlines = label_folder.read().splitlines()  # 返回每一行的数据
    for line in trainlines:
        line = line.split(" ", 1)  # 按照空格分割每一行里面的数据
        temp = line[1] + '\n'
        save_folder.write(temp)
    label_folder.close()
    save_folder.close()


def array_To_string():
    read_folder = open(read_txt, 'r')
    save_folder = open(save_txt, 'a+')
    datas = read_folder.read().splitlines()  # 返回每一行的数据
    for data in datas:
        data = data.split("[", 1)[1]
        data = data.split("]", 1)[0]
        data = data.split(" ", 1)
        temp = data[0] + ' ' + data[1] + '\n'
        save_folder.write(temp)
    read_folder.close()
    save_folder.close()


if __name__ == '__main__':
    array_To_string()
