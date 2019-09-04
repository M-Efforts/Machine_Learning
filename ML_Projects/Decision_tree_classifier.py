# 决策树模型

# 导入pandas用于数据分析
import pandas as pd
# 导入数据分割工具
from sklearn.model_selection import train_test_split
# 使用特征转换器，抽取特征
from sklearn.feature_extraction import DictVectorizer
# 导入决策树分类器
from sklearn.tree import DecisionTreeClassifier
# 导入用于详细的分类性能报告模块
from sklearn.metrics import classification_report

# 利用pandas的read_csv模块直接从互联网中收集泰坦尼克号乘客数据
titanic = pd.read_csv('http://biostat.mc.vanderbilt.edu/wiki/pub/Main/DataSets/titanic.txt')
# 观察前几行数据，查看数据种类
titanic.head()
# 使用pandas，数据都转入pandas独有的dataframe格式（二维数据表格），直接使用info()，查看数据的统计特性
# titanic.info()
# 机器学习有一个不太被初学者重视并且耗时，但是十分重要的一环——特征的选择，这个需要基于一些背景知识。
# 根据对这场事故的了解，sex, age, pclass这些特征都很可能是决定幸免与否的关键因素
X = titanic[['pclass', 'age', 'sex']]
Y = titanic['survived']
# X.info()

# 借由上面的输出，我们设计如下几个数据处理任务：
# 1）age这个数据列，只有633个，需要补充完整
# 2）sex与pclass两个数据列的值都是类别型的，需要转化为数值特征，用0/1代替

# 首先需要补充age里的数据，使用平均数或者中位数都是对模型偏离造成最小影响的策略
X['age'].fillna(X['age'].mean(), inplace=True)
# X.info()

# 数据分割
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.35, random_state=17)
print(type(X_train))

# 将字符型的数据符号化，如'Beijing', 'Tianjin'可被分别符号化为：'0, 1'和'1, 0'
vec = DictVectorizer(sparse=False)
# 特征转换后，可以发现，凡是类别型的特征都单独剥离出来，独成一列特征，数值型的则保持不变。
X_train = vec.fit_transform(X_train.to_dict(orient='record'))
print(type(X_train))
# print(vec.feature_names_)

# 对测试数据的特征进行相同的转换
X_test = vec.transform(X_test.to_dict(orient='record'))

# 初始化决策树分类器
dtc = DecisionTreeClassifier()
dtc.fit(X_train, Y_train)
Y_predict = dtc.predict(X_test)

print(dtc.score(X_test, Y_test))
print(classification_report(Y_predict, Y_test, target_names=['died', 'survived']))
