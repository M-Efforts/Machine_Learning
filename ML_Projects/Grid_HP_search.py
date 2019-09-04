# 超参数的网格搜索方法
from time import time

from sklearn.datasets import fetch_20newsgroups
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.feature_extraction.text import TfidfVectorizer
# 导入Pipeline代码简化工具
from sklearn.pipeline import Pipeline
# 导入网格搜索模块
from sklearn.model_selection import GridSearchCV

news = fetch_20newsgroups(subset='all')
# 数据分割
X_train, X_test, Y_train, Y_test = train_test_split(news.data[:3000], news.target[:3000], test_size=0.35, random_state=17)

# 使用Pipeline简化系统搭建流程，将文本抽取与分类器模型串联起来
clf = Pipeline([('vect', TfidfVectorizer(stop_words='english', analyzer='word')), ('svc', SVC())])

# 实验中需要试验的2个超参数个数分别为4，3，一共有12种组合
# np.logspace(-2, 1, 4)表示产生一个从10^-2到10^1的4位数等比数列
parameters = {'svc__gamma': np.logspace(-2, 1, 4), 'svc__C': np.logspace(-1, 1, 3)}

# 将12组参数组合以及初始化的Pipeline包括3折交叉验证的要求全部告知GridSearchCV
# 此处设定refit=True可以使得搜索到最佳参数后可以返回以最佳参数训练的模型参数
gs = GridSearchCV(clf, parameters, verbose=2, refit=True, cv=3)

# 执行单线程网格搜索
gs.fit(X_train, Y_train)
gs.best_params_, gs.best_score_

print(gs.score(X_test, Y_test))
