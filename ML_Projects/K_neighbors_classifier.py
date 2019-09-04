# K 邻近（分类）

# 导入iris数据加载器
from sklearn.datasets import load_iris
# 导入数据分割工具
from sklearn.model_selection import train_test_split
# 导入用于数据标准化模块
from sklearn.preprocessing import StandardScaler
# 导入k邻近分类器
from sklearn.neighbors import KNeighborsClassifier
# 导入用于详细的分类性能报告模块
from sklearn.metrics import classification_report

# 使用加载器读取数据并存入变量iris
iris = load_iris()

# 查验数据规模
iris.data.shape
# 查看数据说明
print(iris.DESCR)

X_train, X_test, Y_train, Y_test = train_test_split(iris.data, iris.target, test_size=0.35, random_state=17)

ss = StandardScaler()
X_train = ss.fit_transform(X_train)
X_test = ss.transform(X_test)

# 使用K邻近分类器对测试数据进行类别预测，预测结果存储在Y_predict中
knc = KNeighborsClassifier()
knc.fit(X_train, Y_train)
Y_predict = knc.predict(X_test)
print('The accuracy of K-Nearest Neighbor Classifier is ', knc.score(X_test, Y_test))
print(classification_report(Y_test, Y_predict, target_names=iris.target_names))
