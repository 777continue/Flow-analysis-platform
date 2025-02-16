import csv

# 定义 CSV 文件的表头
header = ['姓名', '年龄', '数学成绩', '语文成绩', '英语成绩']

# 定义一些示例数据
data = [
    ['张三', 18, 90, 85, 88],
    ['李四', 17, 78, 82, 75],
    ['王五', 19, 92, 95, 91],
    ['赵六', 18, 85, 87, 86]
]

# 定义要保存的 CSV 文件路径
csv_file_path = 'student_scores.csv'

# 打开文件并写入数据
with open(csv_file_path, mode='w', newline='', encoding='utf-8') as file:
    writer = csv.writer(file)
    # 写入表头
    writer.writerow(header)
    # 写入数据行
    writer.writerows(data)

print(f'CSV 文件已成功生成，路径为: {csv_file_path}')