# from skmultilearn.problem_transform import ClassifierChain
# from skmultilearn.problem_transform import BinaryRelevance
# from skmultilearn.problem_transform import LabelPowerset
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier
# from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from skmultilearn.adapt import MLkNN
from sklearn.svm import SVC
import numpy as np

data_path = '节点向量抽取结果'
data = np.genfromtxt(data_path, dtype=np.float32)
label_path = '独热编码标签数据'
label = np.load(label_path)
save_path = '分类预测结果'


def MultiLabel_class(temp_interval):
    # 把数据集划为测试集和训练集
    X_train, X_test, y_train, y_test = train_test_split(data, label, test_size=temp_interval, random_state=17)
    save_folder = open(save_path, "a+")
    save_folder.write("测试集占比：" + str(temp_interval) + "\n")
    # # 方法1:x_train 对每一个单标签.
    # # with a gaussian naive bayes base classifier
    # classifier = BinaryRelevance(GaussianNB())
    # # train
    # classifier.fit(X_train, y_train)
    # # predict
    # predictions = classifier.predict(X_test)
    # print("方法一：", accuracy_score(y_test, predictions))
    # print("方法一：", np.mean(predictions == y_test))

    # 方法2：OneVsRest 想要分类的作为正类,其他的类作为反类。
    # 分类器使用1对多，SVM用linear kernel
    clf1 = OneVsRestClassifier(SVC(kernel='linear', gamma='auto'), n_jobs=-1)
    # clf1 = OneVsRestClassifier(SVC(kernel='poly', gamma='auto'), n_jobs=-1)
    # 训练
    clf1.fit(X_train, y_train)
    # 输出预测的标签结果
    predict_class = clf1.predict(X_test)
    # 准确率，预测的结果和实际的结果
    save_folder.write("OneVsRest(accuracy_score)：" + str(clf1.score(X_test, y_test)) + "\n")
    save_folder.write("OneVsRest(mean)：" + str(np.mean(predict_class == y_test)) + "\n")

    # # 方法3:powerset:随机抽取k个label,将这k类(有2^k种组合)转化为单标签.
    # classifier = LabelPowerset(GaussianNB())
    # # train
    # classifier.fit(X_train, y_train)
    # # predict
    # predictions = classifier.predict(X_test)
    # print("方法三(accuracy_score)：", accuracy_score(y_test, predictions))
    # print("方法三(mean)：", np.mean(predictions == y_test))

    # 方法4:Adapted Algorithm:多标签KNN算法MLKNN
    classifier = MLkNN(k=20)
    # train
    classifier.fit(X_train, y_train)
    # predict
    predictions = classifier.predict(X_test)
    save_folder.write("MLKNN(accuracy_score)：" + str(accuracy_score(y_test, predictions)) + "\n")
    save_folder.write("MLKNN(mean)：" + str(np.mean(predictions == y_test)) + "\n")

    # # 方法5:分类器链
    # classifier = ClassifierChain(GaussianNB())
    # # train
    # classifier.fit(X_train, y_train)
    # # predict
    # predictions = classifier.predict(X_test)
    # print("方法五：", accuracy_score(y_test, predictions))
    # print("方法五：", np.mean(predictions == y_test))

    # np.save(save_path, predict_class)
    # 准确率，预测的结果和实际的结果
    # print(np.mean(predict_class == y_test))
    save_folder.close()


if __name__ == '__main__':
    interval = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    for in_temp in interval:
        MultiLabel_class(in_temp)
