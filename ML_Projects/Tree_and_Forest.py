# 单一决策树、随机森立和梯度提升决策树模型

# 导入pandas用于数据分析
import pandas as pd
# 导入数据分割工具
from sklearn.model_selection import train_test_split
# 使用特征转换器，抽取特征
from sklearn.feature_extraction import DictVectorizer
# 导入决策树分类器
from sklearn.tree import DecisionTreeClassifier
# 导入随机森林分类器
from sklearn.ensemble import RandomForestClassifier
# 导入梯度提升决策树分类器
from sklearn.ensemble import GradientBoostingClassifier
# 导入用于详细的分类性能报告模块
from sklearn.metrics import classification_report

# 利用pandas的read_csv模块直接从互联网中收集泰坦尼克号乘客数据
titanic = pd.read_csv('http://biostat.mc.vanderbilt.edu/wiki/pub/Main/DataSets/titanic.txt')
# 人工选择sex, age, pclass这些特征作为判别乘客生还的特征
X = titanic[['pclass', 'age', 'sex']]
Y = titanic['survived']

# 首先需要补充age里的数据，使用平均数或者中位数都是对模型偏离造成最小影响的策略
X['age'].fillna(X['age'].mean(), inplace=True)

# 数据分割
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.35, random_state=17)

# 将字符型的数据符号化，如'Beijing', 'Tianjin'可被分别符号化为：'0, 1'和'1, 0'
vec = DictVectorizer(sparse=False)
# 特征转换后，可以发现，凡是类别型的特征都单独剥离出来，独成一列特征，数值型的则保持不变。
X_train = vec.fit_transform(X_train.to_dict(orient='record'))
# 对测试数据的特征进行相同的转换
X_test = vec.transform(X_test.to_dict(orient='record'))

# 使用单一决策树进行模型训练以及预测分析
dtc = DecisionTreeClassifier()
dtc.fit(X_train, Y_train)
dtc_Y_predict = dtc.predict(X_test)

# 使用随机森林分类器进行集成模型的训练以及预测分析
rfc = RandomForestClassifier()
rfc.fit(X_train, Y_train)
rfc_Y_predict = rfc.predict(X_test)

# 使用梯度提升决策树进行集成模型的训练以及预测分析
gbc = GradientBoostingClassifier()
gbc.fit(X_train, Y_train)
gbc_Y_predict = gbc.predict(X_test)

print('The accuracy of decision tree is:', dtc.score(X_test, Y_test))
print(classification_report(dtc_Y_predict, Y_test, target_names=['died', 'survived']))

print('The accuracy of random forest classifier is:', rfc.score(X_test, Y_test))
print(classification_report(rfc_Y_predict, Y_test, target_names=['died', 'survived']))

print('The accuracy of gradient tree boosting is:', gbc.score(X_test, Y_test))
print(classification_report(gbc_Y_predict, Y_test, target_names=['died', 'survived']))
