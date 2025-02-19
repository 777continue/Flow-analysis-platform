from model.example.preprocess import preprocess_data
# 假设数据集文件名为 'unsw-nb15.csv'，目标列名为 'label'
file_path = '../csv/student_scores.csv'
target_column = 'ab'

# 数据预处理
preprocessed_data = preprocess_data(file_path, target_column)
