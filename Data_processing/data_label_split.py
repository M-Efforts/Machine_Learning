from random import shuffle
fuse_txt = '作者-向量-标签（有序）文件'
label_txt = '作者-标签文件'
data_txt = '作者-向量文件上'


def load_label_set():
    read_folder = open(fuse_txt, "r")
    label_folder = open(label_txt, "a+")
    data_folder = open(data_txt, "a+")

    data_lines = read_folder.read().splitlines()  # 返回每一行的数据
    for line in data_lines:
        line = line.split(" ", 1)  # 按照空格分割每一行里面的数据
        temp0 = line[0] + '\n'
        label_folder.write(temp0)
        temp1 = line[1] + '\n'
        data_folder.write(temp1)

    label_folder.close()
    data_folder.close()
    read_folder.close()
    # return  train_box


if __name__ == '__main__':
    load_label_set()
