# 使用Word2Vec技术进行词向量训练


from sklearn.datasets import fetch_20newsgroups
# 从bs4中导入BeautifulSoup
from bs4 import BeautifulSoup
# 导入nltk和re工具包
import nltk

# 需要使用外网执行以下语句
# nltk.down('punkt')

import re
# 从gensim.models中导入word2vec
from gensim.models import word2vec


news = fetch_20newsgroups(subset='all')
X, Y = news.data, news.target


# 定义一个函数名为news_to_sentences的函数将每条新闻中的句子逐一剥离出来，并返回一个句子的列表
def news_to_sentences(new):
    new_text = BeautifulSoup(new).get_text()  # 返回经过BeautifulSoup处理格式化后的字符串文本
    tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')  # 加载英文的句子划分模型
    raw_sentences = tokenizer.tokenize(new_text)  # 对句子进行分割
    sentence = []
    for sent in raw_sentences:
        sentence.append(re.sub('[^a-zA-Z]', ' ', sent.lower().strip()).split())
    return sentence


sentences = []
# 将长篇新闻文本中的句子剥离出来，用于训练
for x in X:
    sentences += news_to_sentences(x)

# 配置词向量的维度
num_features = 300
# 保证被考虑的词汇的频度
min_word_count = 20
# 设定并行化训练使用CPU计算核心的数量，多核可用
num_workers = 2
# 定义训练词向量的上下文窗口大小
context = 5
downSampling = 1e-3


# 训练词向量模型
model = word2vec.Word2Vec(sentences, workers=num_workers, size=num_features, min_count=min_word_count,
                          window=context, sample=downSampling)
# 这个设定代表当前训练好的词向量为最终版，也可以加速模型的训练
model.init_sims(replace=True)

# 利用训练好的模型，寻找训练文本中与morning最相关的10个词汇
print(model.most_similar('morning'))

# 寻找文本中与email中最相关的10个词汇
print(model.most_similar('email'))
