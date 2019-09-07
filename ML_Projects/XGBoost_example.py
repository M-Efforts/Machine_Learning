# XGBoost分类模型对Titanic数据实例

# 导入pandas用于数据分析
import pandas as pd
# 导入训练、测试集划分工具
from sklearn.model_selection import train_test_split
# 导入字典向量化工具
from sklearn.feature_extraction import DictVectorizer
# 导入随机森林分类器
from sklearn.ensemble import RandomForestClassifier
# 导入XGBoost模型
from xgboost import XGBClassifier

# 使用URL地址来下载Titanic事故数据
titanic = pd.read_csv('http://biostat.mc.vanderbilt.edu/wiki/pub/Main/DataSets/titanic.txt')

# 选取pclass、age以及sex作为训练特征
X = titanic[['pclass', 'age', 'sex']]
Y = titanic['survived']

# 对缺失的age信息，采用平均值补全
X['age'].fillna(X['age'].mean(), inplace=True)

# 对原数据进行分割，随机采样25%作为测试集
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25, random_state=17)

vec = DictVectorizer(sparse=False)

# 对原数据进行特征向量化处理
X_train = vec.fit_transform(X_train.to_dict(orient='record'))
X_test = vec.transform(X_test.to_dict(orient='record'))

# 采用默认配置的随机森林分类器对测试集进行测试
rfc = RandomForestClassifier()
rfc.fit(X_train, Y_train)
print('The accuracy of Random Forest Classifier on testing set:', rfc.score(X_test, Y_test))

# 采用默认配置的XGBoost模型对相同的测试集进行预测
xgbc = XGBClassifier()
xgbc.fit(X_train, Y_train)
print('The accuracy of eXtreme Gradient Boosting Classifier on testing set:', xgbc.score(X_test, Y_test))
