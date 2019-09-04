# 使用停词的字符串特征提取

# 从sklearn.datasets里导入20类新闻文本数据抓取器
from sklearn.datasets import fetch_20newsgroups
# 导入train_test_split模块用于分割数据
from sklearn.model_selection import train_test_split
# 从sklearn.feature_extraction.text中导入CountVectorizer
from sklearn.feature_extraction.text import CountVectorizer
# 从sklearn.feature_extraction.text中导入TfidfVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
# 从sklearn.naive_bayes中导入朴素贝叶斯分类器
from sklearn.naive_bayes import MultinomialNB
# 导入classification_report评价指标
from sklearn.metrics import classification_report


# 从互联网上即时下载新闻样本，subset='all'参数代表下载全部近2万条文本存储在变量news中
news = fetch_20newsgroups(subset='all')
X_train, X_test, Y_train, Y_test = train_test_split(news.data, news.target, test_size=0.25, random_state=17)

# 使用停词过滤配置对CountVectorizer初始化
count_filter_vec = CountVectorizer(analyzer='word', stop_words='english')
# 只使用词频统计的方式将原始训练和测试文本转化为特征向量
X_count_filter_train = count_filter_vec.fit_transform(X_train)
X_count_filter_test = count_filter_vec.transform(X_test)
# 使用默认配置对朴素贝叶斯分类器进行初始化
mnb_count = MultinomialNB()
# 使用朴素贝叶斯分类器，对CountVectorizer(不去除停用词)后的训练样本进行参数学习
mnb_count.fit(X_count_filter_train, Y_train)
Y_count_predict = mnb_count.predict(X_count_filter_test)


# 采用默认的配置对TfidfVectorizer进行初始化(不去除停用词)
tfidf_filter_vec = TfidfVectorizer(analyzer='word', stop_words='english')
# 使用tfidf的方式，将原始训练和测试文本转化为特征向量
X_tfidf_filter_train = tfidf_filter_vec.fit_transform(X_train)
X_tfidf_filter_test = tfidf_filter_vec.transform(X_test)
# 使用默认配置对朴素贝叶斯分类器进行初始化
mnb_tfidf = MultinomialNB()
# 使用朴素贝叶斯分类器，对TfidfVectorizer(不去除停用词)后的训练样本进行参数学习
mnb_tfidf.fit(X_tfidf_filter_train, Y_train)
Y_tfidf_predict = mnb_tfidf.predict(X_tfidf_filter_test)


# 输出模型准确性结果
print('The accuracy of classifying 20 newsgroups using Navie Bayes(CountVectorizer without filtering stopwords):',
      mnb_count.score(X_count_filter_test, Y_test))
# 输出更加详细的其他评价分类性能指标
print(classification_report(Y_test, Y_count_predict, target_names=news.target_names))


# 输出模型准确性结果
print('The accuracy of classifying 20 newsgroups using Navie Bayes(TfidfVectorizer without filtering stopwords):',
      mnb_tfidf.score(X_tfidf_filter_test, Y_test))
# 输出更加详细的其他评价分类性能指标
print(classification_report(Y_test, Y_tfidf_predict, target_names=news.target_names))
