import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import os
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')

# 设置随机种子
torch.manual_seed(42)
np.random.seed(42)

os.makedirs('models', exist_ok=True)
os.makedirs('results', exist_ok=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用设备: {device}")

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
print(f"训练时间戳: {timestamp}\n")

# 1. 加载数据
print("正在加载数据...")
df = pd.read_csv('样品特征数据集.csv', header=None)
X = df.iloc[:, :20].values
y = df.iloc[:, 20].values
print(f"数据集: {X.shape}, 标签: {np.unique(y)}")

# 2. 标准化
scaler = StandardScaler()
#X_normalized = scaler.fit_transform(X)
X_normalized = X
y_adjusted = y - 1

# 3. 划分数据集
X_temp, X_test, y_temp, y_test = train_test_split(
    X_normalized, y_adjusted, test_size=0.15, random_state=42, stratify=y_adjusted
)
X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp, test_size=0.15/0.85, random_state=42, stratify=y_temp
)

print(f"训练集: {X_train.shape[0]}, 验证集: {X_val.shape[0]}, 测试集: {X_test.shape[0]}")

# 4. SMOTE过采样（只对小类别）
print("\n正在使用SMOTE过采样小类别...")
# 计算每个类别的样本数
unique, counts = np.unique(y_train, return_counts=True)
class_counts = dict(zip(unique, counts))

# 设置采样策略：小类别过采样到中等水平
sampling_strategy = {}
median_count = int(np.median(counts))
for cls, count in class_counts.items():
    if count < median_count * 0.3:  # 只对样本数<中位数30%的类别过采样
        sampling_strategy[cls] = int(median_count * 0.5)

if sampling_strategy:
    smote = SMOTE(sampling_strategy=sampling_strategy, random_state=42, k_neighbors=3)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
    print(f"SMOTE后训练集: {X_train_resampled.shape[0]} (增加了 {X_train_resampled.shape[0] - X_train.shape[0]} 个样本)")
else:
    X_train_resampled, y_train_resampled = X_train, y_train

# 5. 计算类别权重（平方根缩放）
class_weights_raw = compute_class_weight('balanced', classes=np.unique(y_train_resampled), y=y_train_resampled)
class_weights = np.sqrt(class_weights_raw)
class_weights_tensor = torch.FloatTensor(class_weights).to(device)

# 6. 转换为张量
X_train_tensor = torch.FloatTensor(X_train_resampled).to(device)
y_train_tensor = torch.LongTensor(y_train_resampled).to(device)
X_val_tensor = torch.FloatTensor(X_val).to(device)
y_val_tensor = torch.LongTensor(y_val).to(device)
X_test_tensor = torch.FloatTensor(X_test).to(device)
y_test_tensor = torch.LongTensor(y_test).to(device)

# 7. 数据加载器
batch_size = 512
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# 8. 超深MLP模型（7层隐藏层）
class UltimateMLP(nn.Module):
    def __init__(self, input_size, num_classes, dropout=0.5):
        super(UltimateMLP, self).__init__()

        # 第一阶段：大幅扩展特征
        self.fc1 = nn.Linear(input_size, 2048)
        self.bn1 = nn.BatchNorm1d(2048)
        self.dropout1 = nn.Dropout(dropout)

        # 第二阶段
        self.fc2 = nn.Linear(2048, 1024)
        self.bn2 = nn.BatchNorm1d(1024)
        self.dropout2 = nn.Dropout(dropout)

        # 第三阶段
        self.fc3 = nn.Linear(1024, 512)
        self.bn3 = nn.BatchNorm1d(512)
        self.dropout3 = nn.Dropout(dropout)

        # 第四阶段
        self.fc4 = nn.Linear(512, 256)
        self.bn4 = nn.BatchNorm1d(256)
        self.dropout4 = nn.Dropout(dropout * 0.8)

        # 第五阶段
        self.fc5 = nn.Linear(256, 128)
        self.bn5 = nn.BatchNorm1d(128)
        self.dropout5 = nn.Dropout(dropout * 0.6)

        # 第六阶段
        self.fc6 = nn.Linear(128, 64)
        self.bn6 = nn.BatchNorm1d(64)
        self.dropout6 = nn.Dropout(dropout * 0.4)

        # 第七阶段
        self.fc7 = nn.Linear(64, 32)
        self.bn7 = nn.BatchNorm1d(32)
        self.dropout7 = nn.Dropout(dropout * 0.2)

        # 输出层
        self.fc_out = nn.Linear(32, num_classes)

        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.dropout1(self.relu(self.bn1(self.fc1(x))))
        x = self.dropout2(self.relu(self.bn2(self.fc2(x))))
        x = self.dropout3(self.relu(self.bn3(self.fc3(x))))
        x = self.dropout4(self.relu(self.bn4(self.fc4(x))))
        x = self.dropout5(self.relu(self.bn5(self.fc5(x))))
        x = self.dropout6(self.relu(self.bn6(self.fc6(x))))
        x = self.dropout7(self.relu(self.bn7(self.fc7(x))))
        x = self.fc_out(x)
        return x

model = UltimateMLP(input_size=20, num_classes=13, dropout=0.5).to(device)
print(f"\n超深MLP模型 [2048-1024-512-256-128-64-32]")
print(f"模型参数: {sum(p.numel() for p in model.parameters()):,}")

# 9. Label Smoothing + 类别权重
class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self, weight=None, smoothing=0.1):
        super().__init__()
        self.smoothing = smoothing
        self.weight = weight

    def forward(self, pred, target):
        n_class = pred.size(1)
        one_hot = torch.zeros_like(pred).scatter(1, target.unsqueeze(1), 1)
        one_hot = one_hot * (1 - self.smoothing) + self.smoothing / n_class
        log_prob = F.log_softmax(pred, dim=1)

        if self.weight is not None:
            loss = -(one_hot * log_prob).sum(dim=1)
            loss = (loss * self.weight[target]).mean()
        else:
            loss = -(one_hot * log_prob).sum(dim=1).mean()
        return loss

import torch.nn.functional as F
criterion = LabelSmoothingCrossEntropy(weight=class_weights_tensor, smoothing=0.1)

# 10. 优化器和调度器
optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)

print("损失函数: Label Smoothing + 类别权重")
print("优化器: AdamW + CosineAnnealingWarmRestarts\n")

# 11. 训练
num_epochs = 150
train_losses, train_accs, val_accs = [], [], []
best_val_acc = 0.0
best_epoch = 0
patience = 35
patience_counter = 0
model_path = f'models/best_mlp_ultimate_{timestamp}.pth'

print("开始训练...\n")
for epoch in range(num_epochs):
    # 训练
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for inputs, labels in train_loader:
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    train_acc = 100 * correct / total
    avg_loss = running_loss / len(train_loader)
    train_losses.append(avg_loss)
    train_accs.append(train_acc)

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
    val_accs.append(val_acc)

    scheduler.step()
    current_lr = optimizer.param_groups[0]['lr']

    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}, '
          f'Train: {train_acc:.2f}%, Val: {val_acc:.2f}%, LR: {current_lr:.6f}', end='')

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        best_epoch = epoch + 1
        torch.save(model.state_dict(), model_path)
        patience_counter = 0
        print(' <- Best!')
    else:
        patience_counter += 1
        print()
        if patience_counter >= patience:
            print(f'\n早停！最佳验证准确率: {best_val_acc:.2f}%')
            break

# 12. 测试
print("\n" + "="*50)
print("最终测试...")
model.load_state_dict(torch.load(model_path, weights_only=True))
model.eval()

all_preds, all_labels = [], []
with torch.no_grad():
    for inputs, labels in test_loader:
        outputs = model(inputs)
        _, predicted = torch.max(outputs.data, 1)
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

final_acc = 100 * sum(np.array(all_preds) == np.array(all_labels)) / len(all_labels)
print(f'最终测试准确率: {final_acc:.2f}%\n')

all_preds_adj = np.array(all_preds) + 1
all_labels_adj = np.array(all_labels) + 1

print("分类报告:")
print(classification_report(all_labels_adj, all_preds_adj))

# 13. 保存结果
plt.figure(figsize=(15, 4))
plt.subplot(1, 3, 1)
plt.plot(train_losses)
plt.title('Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.grid(True)

plt.subplot(1, 3, 2)
plt.plot(train_accs, label='Train')
plt.plot(val_accs, label='Validation')
plt.title('Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.legend()
plt.grid(True)

plt.subplot(1, 3, 3)
plt.plot(val_accs, color='orange')
plt.axhline(y=best_val_acc, color='r', linestyle='--', label=f'Best: {best_val_acc:.2f}%')
plt.title('Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.legend()
plt.grid(True)

plt.tight_layout()
curve_path = f'results/training_curves_ultimate_{timestamp}.png'
plt.savefig(curve_path, dpi=300)
print(f"\n训练曲线: {curve_path}")

# 14. 记录到CSV
record = {
    '时间戳': timestamp,
    '训练时间': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    '模型类型': 'MLP-Ultimate',
    '隐藏层结构': '1024-512-256-128-64',
    '总参数量': sum(p.numel() for p in model.parameters()),
    '训练集大小': len(X_train_resampled),
    '验证集大小': len(X_val),
    '测试集大小': len(X_test),
    '最佳轮数': best_epoch,
    '总训练轮数': len(train_losses),
    '最佳验证准确率': f"{best_val_acc:.2f}%",
    '最终测试准确率': f"{final_acc:.2f}%",
    '学习率': '0.001 (Cosine)',
    'Dropout': 0.5,
    '批次大小': 512,
    '早停耐心值': patience,
    '优化器': 'AdamW',
    '标准化方法': 'StandardScaler',
    '类别权重': 'Sqrt-Scaled',
    '是否使用SMOTE': 'Yes',
    '是否使用LabelSmoothing': 'Yes',
    '模型路径': model_path,
    '训练曲线路径': curve_path
}

csv_path = 'MLP记录.csv'
record_df = pd.DataFrame([record])

if os.path.exists(csv_path):
    try:
        existing_df = pd.read_csv(csv_path, encoding='utf-8-sig')
    except:
        try:
            existing_df = pd.read_csv(csv_path, encoding='gbk')
        except:
            existing_df = pd.read_csv(csv_path, encoding='latin1')
    updated_df = pd.concat([existing_df, record_df], ignore_index=True)
    updated_df.to_csv(csv_path, index=False, encoding='utf-8-sig')
else:
    record_df.to_csv(csv_path, index=False, encoding='utf-8-sig')

print(f"训练记录: {csv_path}")
print("="*50)
print(f"最佳验证: {best_val_acc:.2f}% (第{best_epoch}轮)")
print(f"最终测试: {final_acc:.2f}%")
print("="*50)
