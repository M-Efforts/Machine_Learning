# 字典数据特征提取

from sklearn.feature_extraction import DictVectorizer

# 定义一组字典列表，用来表示多个数据样本(每个字典代表一个数据样本)
measurements = [{'city': 'Dubai', 'temperature': 33}, {'city': 'London', 'temperature': 12},
                {'city': 'San Fransisco', 'temperature': 18}]
# 初始化DictVectorizer特征抽取器
vec = DictVectorizer()
# 输出转化之后的特征矩阵
print(vec.fit_transform(measurements).toarray())
# 输出各维度的特征含义
print(vec.get_feature_names())
