# 使用词袋法(Bag-of-Words)对示例文本进行特征向量化
"""
词袋法即将所有文本中的单词组成一个组合（袋子），
然后用每一文本中单词在组合中的出现次数量化该单词
"""

# 导入CountVectorizer特征向量化工具
from sklearn.feature_extraction.text import CountVectorizer

sent1 = 'The cat is walking in the bedroom'
sent2 = 'A dog was running across the kitchen'

count_vec = CountVectorizer()
sentences = [sent1, sent2]

# 输出特征向量化后的表示
print(count_vec.fit_transform(sentences).toarray())

# 输出向量各个维度的特征含义
print(count_vec.get_feature_names())
