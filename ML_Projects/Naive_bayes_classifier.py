# 朴素贝叶斯模型

# 从Sklearn.datasets中导入新闻数据抓取器fetch_20newsgroups
from sklearn.datasets import fetch_20newsgroups
# 导入数据分割工具
from sklearn.model_selection import train_test_split
# 导入用于文本特征向量转化模块
from sklearn.feature_extraction.text import CountVectorizer
# 导入朴素贝叶斯模型
from sklearn.naive_bayes import MultinomialNB
# 导入用于详细的分类性能报告模块
from sklearn.metrics import classification_report

# 联网下载数据
news = fetch_20newsgroups(subset='all')
# 测试数据
# print(len(news.data))
# print(news.data[0])

X_train, X_test, Y_train, Y_test = train_test_split(news.data, news.target, test_size=0.35, random_state=17)
# 特征抽取
vec = CountVectorizer()
X_train = vec.fit_transform(X_train)
X_test = vec.transform(X_test)

# 使用默认配置初始化朴素贝叶斯模型
mnb = MultinomialNB()
# 利用训练数据对模型参数进行估计
mnb.fit(X_train, Y_train)
# 对测试样本进行类别预测，结果存储在Y_predict中
Y_predict = mnb.predict(X_test)

print('The accuracy of Naive Bayes Classifier is ', mnb.score(X_test, Y_test))
print(classification_report(Y_test, Y_predict, target_names=news.target_names))
