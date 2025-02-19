import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, LabelEncoder

column_names = [
    'srcip', 'sport', 'dstip', 'dsport', 'proto', 'state', 'dur',
    'sbytes', 'dbytes', 'sttl', 'dttl', 'sloss', 'dloss', 'service',
    'Sload', 'Dload', 'Spkts', 'Dpkts', 'swin', 'dwin', 'stcpb',
    'dtcpb', 'smeansz', 'dmeansz', 'trans_depth', 'res_bdy_len',
    'Sjit', 'Djit', 'Stime', 'Ltime', 'Sintpkt', 'Dintpkt', 'tcprtt',
    'synack', 'ackdat', 'is_sm_ips_ports', 'ct_state_ttl',
    'ct_flw_http_mthd', 'is_ftp_login', 'ct_ftp_cmd', 'ct_srv_src',
    'ct_srv_dst', 'ct_dst_ltm', 'ct_src_ ltm', 'ct_src_dport_ltm',
    'ct_dst_sport_ltm', 'ct_dst_src_ltm', 'attack_cat', 'Label'
]

# 假设已通过随机森林选择了15个特征，这里直接使用论文中给出的特征名
selected_features = ['state', 'ct_state_ttl', 'attack_cat', 'sbytes', 'smeansz', 'Sload', 'dmeansz', 'Dpkts', 'Dload', 'dttl', 'dur', 'dbytes', 'sport', 'ct_srv_dst', 'Dintpkt']


# 自定义数据集类
class NetworkAnomalyDataset(Dataset):
    def __init__(self, data_path, selected_features):
        data = pd.read_csv(data_path, names=column_names, low_memory=False)  # 添加 low_memory=False
        data = data[selected_features]

        label_encoders = {}
        for column in data.columns:
            if data[column].dtype == 'object' or pd.api.types.infer_dtype(data[column]) == 'mixed':
                # 将列中的所有值转换为字符串类型
                data[column] = data[column].astype(str)
                le = LabelEncoder()
                data[column] = le.fit_transform(data[column])
                label_encoders[column] = le

        X = data.drop('attack_cat', axis=1)
        y = data['attack_cat']
        scaler = MinMaxScaler()
        X = scaler.fit_transform(X)

        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y.values, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]



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

        '''test_input = torch.randn(1, 1, len(selected_features) - 1)
        x = self.conv1(test_input)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.conv3(x)
        x = self.pool3(x)
        x = self.flatten(x)
        fc1_input_dim = x.shape[1]
        print(f"第一个全连接层的输入维度为: {fc1_input_dim}")'''
        self.fc1 = nn.Linear(512, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 10)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

   
    def forward(self, x):
        x = x.unsqueeze(1)
        #print(f"输入形状: {x.shape}")
        x = self.relu(self.bn1(self.conv1(x)))
        #print(f"经过 conv1 后的形状: {x.shape}")
        x = self.pool1(x)
        #print(f"经过 pool1 后的形状: {x.shape}")
        x = self.relu(self.bn2(self.conv2(x)))
        #print(f"经过 conv2 后的形状: {x.shape}")
        x = self.pool2(x)
        #print(f"经过 pool2 后的形状: {x.shape}")
        x = self.relu(self.bn3(self.conv3(x)))
        #print(f"经过 conv3 后的形状: {x.shape}")
        x = self.pool3(x)
        #print(f"经过 pool3 后的形状: {x.shape}")
        x = self.flatten(x)
        #print(f"经过 flatten 后的形状: {x.shape}")
        x = self.relu(self.fc1(x))
        #print(f"经过 fc1 后的形状: {x.shape}")
        x = self.relu(self.fc2(x))
        #print(f"经过 fc2 后的形状: {x.shape}")
        x = self.softmax(self.fc3(x))
        #print(f"经过 fc3 和 softmax 后的形状: {x.shape}")
        return x



# 训练模型
def train_model(model, train_loader, criterion, optimizer, epochs):
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        for i, (inputs, labels) in enumerate(train_loader):
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f'Epoch {epoch + 1}, Loss: {running_loss / len(train_loader)}')


# 评估模型
def evaluate_model(model, test_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = correct / total
    print(f'Test Accuracy: {accuracy * 100:.2f}%')


# 主函数
if __name__ == "__main__":
    data_path = '/home/continue/share/UNSW-NB15_1.csv'
    batch_size = 64
    epochs = 10
    learning_rate = 0.001

    dataset = NetworkAnomalyDataset(data_path, selected_features)
    train_data, test_data = train_test_split(dataset, test_size=0.3, random_state=42)
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

    model = NetworkAnomalyCNN()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    train_model(model, train_loader, criterion, optimizer, epochs)
    evaluate_model(model, test_loader)
    torch.save(model.state_dict(), 'param.pth')

# path: DL/CNN/main.py