# 特征筛选  (注意:要先特征向量化之后才能进行特征选择)

# 导入pandas用于数据分析
import pandas as pd
import numpy as np
# 导入数据分割工具
from sklearn.model_selection import train_test_split
# 使用特征转换器，抽取特征
from sklearn.feature_extraction import DictVectorizer
# 导入决策树分类器
from sklearn.tree import DecisionTreeClassifier
# 导入特征筛选器
from sklearn import feature_selection
# 导入交叉验证模块
from sklearn.model_selection import cross_val_score
import pylab as pl


# 利用pandas的read_csv模块直接从互联网中收集泰坦尼克号乘客数据
titanic = pd.read_csv('http://biostat.mc.vanderbilt.edu/wiki/pub/Main/DataSets/titanic.txt')

# 分离数据特征与预测目标
Y = titanic['survived']  # 提取标记列
X = titanic.drop(['row.names', 'name', 'survived'], axis=1)  # 提取除了数组中三列的其他列数据

# 首先需要补充age里的数据，使用平均数或者中位数都是对模型偏离造成最小影响的策略
X['age'].fillna(X['age'].mean(), inplace=True)
# 其余维度（非数值型）的缺失值均用unknown进行填充
X.fillna('UNKNOWN', inplace=True)

# 数据分割
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.35, random_state=17)

# 将字符型的数据符号化，如'Beijing', 'Tianjin'可被分别符号化为：'0, 1'和'1, 0'
vec = DictVectorizer()
# 特征转换后，可以发现，凡是类别型的特征都单独剥离出来，独成一列特征，数值型的则保持不变。
X_train = vec.fit_transform(X_train.to_dict(orient='record'))
# 对测试数据的特征进行相同的转换
X_test = vec.transform(X_test.to_dict(orient='record'))

# 使用单一决策树进行模型训练以及预测分析
dtc = DecisionTreeClassifier(criterion='entropy')
dtc.fit(X_train, Y_train)
print('The accuracy of decision tree is:', dtc.score(X_test, Y_test))

"""
SelectKBest和SelectPercentile的用法比较相似，前者选择排名在前n个的变量，后者选择排名在前n%的变量；其中排名的
方式通过指定参数来确定：对于regression,可以使用f_regression;对于classification,可以使用chi2或者f_classif。
此外，此外选择算法内部会根据因变量y的存在与否自主选择有监督或无监督的学习方式。
"""

# 筛选前20%的特征，使用相同配置的决策树模型进行预测，并且评估性能
fs = feature_selection.SelectPercentile(feature_selection.chi2, percentile=20)
X_train_fs = fs.fit_transform(X_train, Y_train)
dtc.fit(X_train_fs, Y_train)
X_test_fs = fs.transform(X_test)
print('The accuracy of decision tree with 20% features is:', dtc.score(X_test_fs, Y_test))


# 通过交叉验证的方法，按照固定间隔的百分比筛选特征，并作图展示性能随特征筛选比例的变化
percentiles = range(1, 100, 2)
results = []

for i in percentiles:
    fs = feature_selection.SelectPercentile(feature_selection.chi2, percentile=i)  # percentile表示选取前%i 的特征
    X_train_fs = fs.fit_transform(X_train, Y_train)
    scores = cross_val_score(dtc, X_train_fs, Y_train, cv=5)  # 5折交叉验证，返回5次验证后的scores
    results = np.append(results, scores.mean())  # 得到每次取的前%i特征所产生的score的均值
print(results)

# 找到体现最佳性能的特征筛选的百分比对应的索引
opt = np.where(results == results.max())[0]
# 此处得到的opt数据仍然是数组类型，需要强制转化为int型才可作为索引
print('Optimal number of features %d' % percentiles[int(opt)])

pl.plot(percentiles, results)
pl.xlabel('percentiles of features')
pl.ylabel('accuracy')
pl.show()

# 使用最佳筛选后的特征，利用相同配置的模型在测试集上进行性能评估
fs = feature_selection.SelectPercentile(feature_selection.chi2, percentile=int(opt))
X_train_fs = fs.fit_transform(X_train, Y_train)
dtc.fit(X_train_fs, Y_train)
X_test_fs = fs.transform(X_test)
print(dtc.score(X_test_fs, Y_test))
