import xlrd

# 路径前加 r，读取的文件路径
file_path = r'期刊相似性文件'
save_path = r'期刊向量'

# 获取Excel数据
data = xlrd.open_workbook(file_path)

# 获取Excel中的sheet数据
table = data.sheet_by_name('Sheet1')

# 获取总行数
nrows = table.nrows
# 获取总列数
ncols = table.ncols


# 获取一行的数值，例如第1行
# rowvalue = table.row_values(0)

# 获取一列的数值，例如第2列
# col_values = table.col_values(2)

save_folder = open(save_path, "a+")
# 获取一个单元格的数值，例如第1行第2列
for i in range(18):
    for j in range(18):
        cell_value = table.cell(i*18+j, 2).value
        save_folder.write(str(cell_value) + " ")
    save_folder.write("\n")
save_folder.close()
