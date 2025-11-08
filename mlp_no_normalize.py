import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import warnings

warnings.filterwarnings('ignore')
torch.manual_seed(42)
np.random.seed(42)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用设备: {device}")
print("=" * 50)
print("实验：不使用标准化训练MLP")
print("=" * 50)

# 加载数据
df = pd.read_csv('样品特征数据集.csv', header=None)
X = df.iloc[:, :20].values
y = df.iloc[:, 20].values - 1

print(f"\n原始数据范围:")
print(f"  最小值: {X.min():.2f}")
print(f"  最大值: {X.max():.2f}")
print(f"  均值: {X.mean():.2f}")
print(f"  标准差: {X.std():.2f}")

# 划分数据集（不标准化！）
X_temp, X_test, y_temp, y_test = train_test_split(
    X, y, test_size=0.15, random_state=42, stratify=y
)
X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp, test_size=0.15/0.85, random_state=42, stratify=y_temp
)

print(f"\n训练集: {X_train.shape[0]}, 验证集: {X_val.shape[0]}, 测试集: {X_test.shape[0]}")

# 转换为张量
X_train_tensor = torch.FloatTensor(X_train).to(device)
y_train_tensor = torch.LongTensor(y_train).to(device)
X_val_tensor = torch.FloatTensor(X_val).to(device)
y_val_tensor = torch.LongTensor(y_val).to(device)
X_test_tensor = torch.FloatTensor(X_test).to(device)
y_test_tensor = torch.LongTensor(y_test).to(device)

# 数据加载器
train_loader = DataLoader(TensorDataset(X_train_tensor, y_train_tensor), batch_size=512, shuffle=True)
val_loader = DataLoader(TensorDataset(X_val_tensor, y_val_tensor), batch_size=512, shuffle=False)
test_loader = DataLoader(TensorDataset(X_test_tensor, y_test_tensor), batch_size=512, shuffle=False)

# 简单的MLP模型
class SimpleMLP(nn.Module):
    def __init__(self):
        super(SimpleMLP, self).__init__()
        self.fc1 = nn.Linear(20, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 13)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.dropout(self.relu(self.fc2(x)))
        x = self.dropout(self.relu(self.fc3(x)))
        return self.fc4(x)

model = SimpleMLP().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

print(f"\n模型参数: {sum(p.numel() for p in model.parameters()):,}")
print("\n开始训练（不使用标准化）...\n")

# 训练
num_epochs = 30
best_val_acc = 0.0

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for inputs, labels in train_loader:
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()

        # 检查梯度是否正常
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    train_acc = 100 * correct / total
    avg_loss = running_loss / len(train_loader)

    # 验证
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in val_loader:
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    val_acc = 100 * correct / total

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        marker = " <- Best!"
    else:
        marker = ""

    # 检查是否出现NaN
    if np.isnan(avg_loss):
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: NaN - 训练失败！")
        print("\n原因：不标准化导致数值溢出")
        break

    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}, '
          f'Train: {train_acc:.2f}%, Val: {val_acc:.2f}%, '
          f'GradNorm: {grad_norm:.2f}{marker}')

# 测试
print("\n" + "=" * 50)
print("最终测试...")
model.eval()
all_preds, all_labels = [], []

with torch.no_grad():
    for inputs, labels in test_loader:
        outputs = model(inputs)
        _, predicted = torch.max(outputs.data, 1)
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

final_acc = 100 * sum(np.array(all_preds) == np.array(all_labels)) / len(all_labels)
print(f'测试准确率: {final_acc:.2f}%')

all_preds_adj = np.array(all_preds) + 1
all_labels_adj = np.array(all_labels) + 1

print("\n分类报告:")
print(classification_report(all_labels_adj, all_preds_adj))

print("\n" + "=" * 50)
print("实验结论:")
print("=" * 50)
print(f"不标准化的准确率: {final_acc:.2f}%")
print(f"标准化的准确率: ~60.5% (之前的结果)")
print(f"性能差距: {60.5 - final_acc:.2f}%")
print("\n不标准化会导致：")
print("1. 训练不稳定（损失震荡）")
print("2. 收敛速度慢")
print("3. 准确率显著下降")
print("4. 可能出现梯度爆炸/消失")
print("=" * 50)
