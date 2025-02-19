import torch

# 加载 .pth 文件
state_dict = torch.load('param.pth')
for key, value in state_dict.items():
    print(f"Key: {key}, Shape: {value.shape}")
