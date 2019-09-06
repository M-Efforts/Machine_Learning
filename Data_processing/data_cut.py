train_txt = '原路径'
test_txt = '目标路径'


def dataCut():
    data_folder = open(train_txt, "r")
    save_folder = open(test_txt, "a+")
    trainlines = data_folder.read().splitlines()  # 返回每一行的数据

    for i in range(241892):
        save_folder.write(trainlines[i] + '\n')
    save_folder.close()
    data_folder.close()


if __name__ == "__main__":
    dataCut()
