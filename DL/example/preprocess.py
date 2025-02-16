import pandas as pd
import torch
import torch.nn as nn
import numpy as np

# 读取数据集
def read_dataset(file_path):
    data = pd.read_csv(file_path)
    print("读取数据集完成：")
    data.info()
    print(f"数据集行数: {data.shape[0]}, 列数: {data.shape[1]}")
    return data

# 数据清洗：去除空值和零值
def clean_data(data):
    rows_before = data.shape[0]
    data = data.dropna()  # 去除包含空值的行
    data = data[(data != 0).all(axis=1)]  # 去除全为零值的行
    rows_after = data.shape[0]
    print("数据清洗完成：")
    data.info()
    print(f"清洗前数据集行数: {rows_before}, 清洗后数据集行数: {rows_after}")
    return data

# 数据归一化
def normalize_data(data):
    numerical_columns = data.select_dtypes(include=[np.number]).columns
    data_before = data[numerical_columns].describe()
    # 将数值列转换为 PyTorch 张量
    tensor_data = torch.tensor(data[numerical_columns].values, dtype=torch.float32)
    # 计算最小值和最大值
    min_vals = torch.min(tensor_data, dim=0)[0]
    max_vals = torch.max(tensor_data, dim=0)[0]
    # 归一化
    normalized_tensor = (tensor_data - min_vals) / (max_vals - min_vals)
    # 将归一化后的张量转换回 DataFrame
    data[numerical_columns] = normalized_tensor.numpy()
    data_after = data[numerical_columns].describe()
    print("数据归一化完成：")
    print("归一化前数值列统计信息：")
    print(data_before)
    print("归一化后数值列统计信息：")
    print(data_after)
    return data

# 手动实现标签编码
def manual_label_encoding(series):
    unique_values = series.unique()
    mapping = {value: idx for idx, value in enumerate(unique_values)}
    return series.map(mapping)

# 特征选择：使用 PyTorch 构建简单的线性模型来评估特征重要性
def feature_selection(data, target_column):
    columns_before = data.columns
    X = data.drop(target_column, axis=1)
    y = data[target_column]

    # 处理非数值类型的特征列
    object_columns = X.select_dtypes(include=['object']).columns
    if len(object_columns) > 0:
        for col in object_columns:
            X[col] = manual_label_encoding(X[col])

    # 假设目标列是分类标签，将其转换为数值类型
    if y.dtype == 'object':
        y = manual_label_encoding(y)

    # 将数据转换为 PyTorch 张量
    X_tensor = torch.tensor(X.values, dtype=torch.float32)
    y_tensor = torch.tensor(y.values, dtype=torch.long)

    # 构建简单的全连接神经网络模型
    model = nn.Sequential(
        nn.Linear(X_tensor.shape[1], 64),
        nn.ReLU(),
        nn.Linear(64, len(y.unique()))
    )

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # 训练模型
    num_epochs = 100
    for epoch in range(num_epochs):
        outputs = model(X_tensor)
        loss = criterion(outputs, y_tensor)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # 获取特征重要性
    # 这里简单使用第一层权重的绝对值之和来表示特征重要性
    feature_importances = torch.abs(model[0].weight).sum(dim=0)
    feature_importances = pd.Series(feature_importances.detach().numpy(), index=X.columns)

    selected_features = feature_importances.nlargest(2).index  # 选择前 2 个重要特征
    selected_data = data[selected_features]
    columns_after = selected_data.columns
    print("特征选择完成：")
    print(f"选择前特征数量: {len(columns_before)}, 选择后特征数量: {len(columns_after)}")
    print("选择的特征：", columns_after)
    return selected_data

# 完整的预处理流程
def preprocess_data(file_path, target_column):
    data = read_dataset(file_path)
    data = clean_data(data)
    data = normalize_data(data)
    selected_data = feature_selection(data, target_column)
    return selected_data

# 示例调用，需替换为实际文件路径和目标列名
# file_path = 'your_file.csv'
# target_column = 'your_target_column'
# preprocess_data(file_path, target_column)