# from time import time
from sklearn.model_selection import train_test_split
import sklearn.metrics
import numpy as np

X = np.genfromtxt(r'所有节点向量映射-有序',
                  dtype=np.float32)
Y = np.genfromtxt(r'期刊标签',
                  dtype=np.int32)
output_txt = r'分类预测'


def GBDT_classify(train_X, train_Y, test_X, test_Y, inte_temp):
    from sklearn.ensemble import GradientBoostingClassifier
    # t0 = time()
    clf = GradientBoostingClassifier(n_estimators=200)
    clf.fit(train_X, train_Y)
    # print("GBDT done in %0.3fs" % (time() - t0))
    pre_y_test = clf.predict(test_X)
    save_folder = open(output_txt, "a+")
    # 将测试结果和预测值保存到文件中
    # save_folder1 = open(target_txt, "a+")
    # for i in test_Y:
    #     save_folder1.write(str(i) + "\n")
    # save_folder2 = open(predict_txt, "a+")
    # for j in pre_y_test:
    #     save_folder2.write(str(j) + "\n")
    # save_folder1.close()
    # save_folder2.close()

    # print(sklearn.metrics.precision_score(test_Y, pre_y_test, average='macro'))
    # print(sklearn.metrics.recall_score(test_Y, pre_y_test, average='micro'))
    # print(sklearn.metrics.f1_score(test_Y, pre_y_test, average='macro'))
    # print(sklearn.metrics.f1_score(test_Y, pre_y_test, average='micro'))
    # print(sklearn.metrics.fbeta_score(test_Y, pre_y_test, beta=0.5, average='macro'))
    # print("GBDT Metrics : {0}".format(sklearn.metrics.precision_recall_fscore_support(test_Y, pre_y_test)))
    save_folder.write("Training data rate:" + str(inte_temp) + "\n")
    save_folder.write("Macro-F1:" + str(sklearn.metrics.f1_score(test_Y, pre_y_test, average='macro')) + "\n")
    save_folder.write("Micro-F1:" + str(sklearn.metrics.f1_score(test_Y, pre_y_test, average='micro')) + "\n")
    save_folder.close()


if __name__ == '__main__':
    interval = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    for in_temp in interval:
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=in_temp, random_state=17)
        GBDT_classify(X_train, Y_train, X_test, Y_test, in_temp)
