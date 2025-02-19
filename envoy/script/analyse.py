import redis
import torch
import torch.nn as nn
import numpy as np

# 连接 Redis
r = redis.Redis(host='127.0.0.1', port=6379, db=0)

class NetworkAnomalyCNN(nn.Module):
    def __init__(self):
        super(NetworkAnomalyCNN, self).__init__()
        self.conv1 = nn.Conv1d(1, 64, kernel_size=2)
        self.bn1 = nn.BatchNorm1d(64)
        padding1 = (2 - 1) // 2
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=1, padding=padding1)
        self.conv2 = nn.Conv1d(64, 64, kernel_size=2)
        self.bn2 = nn.BatchNorm1d(64)
        padding2 = (2 - 1) // 2
        self.pool2 = nn.MaxPool1d(kernel_size=2, stride=1, padding=padding2)
        self.conv3 = nn.Conv1d(64, 64, kernel_size=2)
        self.bn3 = nn.BatchNorm1d(64)
        padding3 = (2 - 1) // 2
        self.pool3 = nn.MaxPool1d(kernel_size=2, stride=1, padding=padding3)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(512, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 10)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

   
    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.pool1(x)
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.pool2(x)
        x = self.relu(self.bn3(self.conv3(x)))
        x = self.pool3(x)
        x = self.flatten(x)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.softmax(self.fc3(x))
        return x

# 订阅数据消息队列
pubsub = r.pubsub()
pubsub.subscribe('data_queue')

for message in pubsub.listen():
    if message['type'] == 'message':
        # 分离数据
        data = message['data'].decode('utf-8').split('|')
        source_ip = int(''.join([str(int(i)) for i in data[0].split('.')]))
        destination_ip = int(''.join([str(int(i)) for i in data[1].split('.')]))
        timestamp = int(data[2])
        method = data[3]
        path = data[4]
        # torch.tensor() 方法转换为二维张量
        features = torch.tensor([[source_ip, destination_ip, timestamp]], dtype=torch.float32)
        # 将特征张量 features 输入到已经定义好的 CNN 模型 model 中进行预测，得到模型的输出 output。
        model = NetworkAnomalyCNN()
        output = model(features)
        r.publish("result_queue", "Normal")
        print(output)
