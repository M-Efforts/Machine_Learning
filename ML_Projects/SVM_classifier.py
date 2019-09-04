# 支持向量机分类

# 导入手写体数字加载器
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
# 导入用于数据标准化模块
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
# 导入用于详细的分类性能报告模块
from sklearn.metrics import classification_report

digits = load_digits()
digits.data.shape

X_train, X_test, Y_train, Y_test = train_test_split(digits.data, digits.target, test_size=0.25, random_state=17)
ss = StandardScaler()
X_train = ss.fit_transform(X_train)
X_test = ss.transform(X_test)

lsvc = LinearSVC()
lsvc.fit(X_train, Y_train)
Y_predict = lsvc.predict(X_test)
print('The accuracy of Linear SVC is:', lsvc.score(X_test, Y_test))
print(classification_report(Y_test, Y_predict, target_names=digits.target_names.astype(str)))
