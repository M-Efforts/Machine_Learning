# 读取文件，并将其中每一行的数据分割，然后分别对两列数据进行计数

data_txt = r'边表数据'

read_folder = open(data_txt, "r")
trainlines = read_folder.read().splitlines()  # 返回每一行的数据

# 使用数组保存截取的数据
md_M = []  # Movie-Director中Movie节点个数
md_D = []  # Movie-Director中Director节点个数
ma_M = []  # Movie-Actor中Movie节点个数
ma_A = []  # Movie-Actor中Actor节点个数
mt_M = []  # Movie-Tag中Movie节点个数
mt_T = []  # Movie-Tag中Tag节点个数

for i in range(0, 6829):
    md_M.append(trainlines[i].split('\t', 2)[0])
    md_D.append(trainlines[i].split('\t', 2)[1])

for j in range(6829, 167896):
    ma_M.append(trainlines[j].split('\t', 2)[0])
    ma_A.append(trainlines[j].split('\t', 2)[1])

for k in range(167896, 204837):
    mt_M.append(trainlines[k].split('\t', 2)[0])
    mt_T.append(trainlines[k].split('\t', 2)[1])

# 将保存节点数据的数组转换为集合（集合中的元素不重复）
md_M = set(md_M)
md_D = set(md_D)
ma_M = set(ma_M)
ma_A = set(ma_A)
mt_M = set(mt_M)
mt_T = set(mt_T)

print(len(md_M))
print(len(md_D))
print(len(ma_M))
print(len(ma_A))
print(len(mt_M))
print(len(mt_T))
