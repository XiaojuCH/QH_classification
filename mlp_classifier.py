import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils.class_weight import compute_class_weight
import matplotlib.pyplot as plt
import os
from datetime import datetime
import warnings

# 忽略警告
warnings.filterwarnings('ignore')

# 设置随机种子
torch.manual_seed(42)
np.random.seed(42)

# 创建模型保存目录
os.makedirs('models', exist_ok=True)
os.makedirs('results', exist_ok=True)

# 检查是否有GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用设备: {device}")

# 生成时间戳用于保存模型
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
print(f"训练时间戳: {timestamp}")

# 1. 加载数据
print("\n正在加载数据...")
df = pd.read_csv('样品特征数据集.csv', header=None)
print(f"数据集形状: {df.shape}")

# 2. 分离特征和标签
X = df.iloc[:, :20].values  # 前20列作为特征
y = df.iloc[:, 20].values   # 最后一列作为标签

print(f"特征形状: {X.shape}")
print(f"标签形状: {y.shape}")
print(f"标签类别: {np.unique(y)}")

# 打印类别分布
print("\n类别分布:")
unique, counts = np.unique(y, return_counts=True)
for label, count in zip(unique, counts):
    print(f"  类别{label}: {count:6d} ({count/len(y)*100:.2f}%)")

# 3. 标准化特征到[-1, 1]区间
print("正在标准化特征到[-1, 1]区间...")
scaler = MinMaxScaler(feature_range=(-1, 1))
X_normalized = scaler.fit_transform(X)

# 验证标准化结果
print(f"标准化后特征的最小值: {X_normalized.min():.4f}")
print(f"标准化后特征的最大值: {X_normalized.max():.4f}")

# 4. 将标签转换为0-based索引（因为PyTorch的CrossEntropyLoss需要0-based索引）
y_adjusted = y - 1  # 将1-13转换为0-12
print(f"调整后的标签范围: {y_adjusted.min()} - {y_adjusted.max()}")

# 5. 划分训练集、验证集和测试集
# 先划分出测试集 (15%)
X_temp, X_test, y_temp, y_test = train_test_split(
    X_normalized, y_adjusted, test_size=0.15, random_state=42, stratify=y_adjusted
)

# 再从剩余数据中划分训练集和验证集 (训练集70%, 验证集15%)
X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp, test_size=0.15/0.85, random_state=42, stratify=y_temp
)

print(f"训练集大小: {X_train.shape[0]} ({X_train.shape[0]/len(X_normalized)*100:.1f}%)")
print(f"验证集大小: {X_val.shape[0]} ({X_val.shape[0]/len(X_normalized)*100:.1f}%)")
print(f"测试集大小: {X_test.shape[0]} ({X_test.shape[0]/len(X_normalized)*100:.1f}%)")

# 5.5 计算类别权重（处理类别不平衡）
print("\n正在计算类别权重...")
class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
class_weights_tensor = torch.FloatTensor(class_weights).to(device)
print("类别权重:")
for i, weight in enumerate(class_weights):
    print(f"  类别{i+1}: {weight:.4f}")

# 6. 转换为PyTorch张量
X_train_tensor = torch.FloatTensor(X_train).to(device)
y_train_tensor = torch.LongTensor(y_train).to(device)
X_val_tensor = torch.FloatTensor(X_val).to(device)
y_val_tensor = torch.LongTensor(y_val).to(device)
X_test_tensor = torch.FloatTensor(X_test).to(device)
y_test_tensor = torch.LongTensor(y_test).to(device)

# 7. 创建数据加载器
batch_size = 256
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# 8. 定义MLP模型
class MLP(nn.Module):
    def __init__(self, input_size, hidden_sizes, num_classes):
        super(MLP, self).__init__()
        layers = []

        # 第一层
        layers.append(nn.Linear(input_size, hidden_sizes[0]))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(0.3))

        # 中间隐藏层
        for i in range(len(hidden_sizes) - 1):
            layers.append(nn.Linear(hidden_sizes[i], hidden_sizes[i+1]))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.3))

        # 输出层
        layers.append(nn.Linear(hidden_sizes[-1], num_classes))

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

# 9. 创建模型实例
input_size = 20
hidden_sizes = [128, 64, 32]  # 三层隐藏层
num_classes = 13

model = MLP(input_size, hidden_sizes, num_classes).to(device)
print("\n模型结构:")
print(model)
print(f"\n模型参数总数: {sum(p.numel() for p in model.parameters())}")

# 10. 定义损失函数和优化器（使用类别权重）
criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)  # 交叉熵损失函数（带类别权重）
optimizer = optim.Adam(model.parameters(), lr=0.001)
print("\n损失函数: CrossEntropyLoss (带类别权重)")
print("优化器: Adam (lr=0.001)")

# 11. 训练模型（带验证集和早停）
num_epochs = 50
train_losses = []
train_accuracies = []
val_accuracies = []
best_val_accuracy = 0.0
best_epoch = 0  # 记录最佳轮数
patience = 10  # 早停耐心值
patience_counter = 0
model_path = f'models/best_mlp_{timestamp}.pth'  # 预定义模型路径

print("\n开始训练...")
for epoch in range(num_epochs):
    # ===== 训练阶段 =====
    model.train()
    running_loss = 0.0
    correct_train = 0
    total_train = 0

    for inputs, labels in train_loader:
        # 前向传播
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        # 反向传播和优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # 统计
        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total_train += labels.size(0)
        correct_train += (predicted == labels).sum().item()

    # 计算训练准确率
    train_accuracy = 100 * correct_train / total_train
    avg_loss = running_loss / len(train_loader)
    train_losses.append(avg_loss)
    train_accuracies.append(train_accuracy)

    # ===== 验证阶段 =====
    model.eval()
    correct_val = 0
    total_val = 0

    with torch.no_grad():
        for inputs, labels in val_loader:
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total_val += labels.size(0)
            correct_val += (predicted == labels).sum().item()

    val_accuracy = 100 * correct_val / total_val
    val_accuracies.append(val_accuracy)

    # 打印进度
    print(f'Epoch [{epoch+1}/{num_epochs}], '
          f'Loss: {avg_loss:.4f}, '
          f'Train Acc: {train_accuracy:.2f}%, '
          f'Val Acc: {val_accuracy:.2f}%', end='')

    # ===== 早停和模型保存 =====
    if val_accuracy > best_val_accuracy:
        best_val_accuracy = val_accuracy
        best_epoch = epoch + 1
        # 保存到models文件夹
        model_path = f'models/best_mlp_{timestamp}.pth'
        torch.save(model.state_dict(), model_path)
        patience_counter = 0
        print(' <- Best model saved!')
    else:
        patience_counter += 1
        print()
        if patience_counter >= patience:
            print(f'\n早停触发！验证集准确率已经{patience}轮没有提升。')
            print(f'最佳验证集准确率: {best_val_accuracy:.2f}%')
            break

# 12. 加载最佳模型并在测试集上评估
print("\n" + "="*50)
print("加载最佳模型并在测试集上进行最终评估...")
model.load_state_dict(torch.load(model_path, weights_only=True))
model.eval()
correct = 0
total = 0
all_predictions = []
all_labels = []

with torch.no_grad():
    for inputs, labels in test_loader:
        outputs = model(inputs)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        all_predictions.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

final_accuracy = 100 * correct / total
print(f'\n最终测试集准确率: {final_accuracy:.2f}%')

# 13. 每个类别的准确率
from sklearn.metrics import classification_report, confusion_matrix

# 将标签转回1-13用于报告
all_predictions_adjusted = np.array(all_predictions) + 1
all_labels_adjusted = np.array(all_labels) + 1

print("\n分类报告:")
print(classification_report(all_labels_adjusted, all_predictions_adjusted))

print("\n混淆矩阵:")
cm = confusion_matrix(all_labels_adjusted, all_predictions_adjusted)
print(cm)

# 14. 绘制训练曲线
plt.figure(figsize=(15, 4))

plt.subplot(1, 3, 1)
plt.plot(train_losses)
plt.title('Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.grid(True)

plt.subplot(1, 3, 2)
plt.plot(train_accuracies, label='Train Accuracy')
plt.plot(val_accuracies, label='Validation Accuracy')
plt.title('Accuracy over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.legend()
plt.grid(True)

plt.subplot(1, 3, 3)
plt.plot(val_accuracies, label='Validation Accuracy', color='orange')
plt.axhline(y=best_val_accuracy, color='r', linestyle='--', label=f'Best Val Acc: {best_val_accuracy:.2f}%')
plt.title('Validation Accuracy with Best')
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.legend()
plt.grid(True)

plt.tight_layout()
curve_path = f'results/training_curves_{timestamp}.png'
plt.savefig(curve_path, dpi=300)
print(f"\n训练曲线已保存到 {curve_path}")

# 15. 记录训练结果到CSV
print("\n正在记录训练结果...")
record = {
    '时间戳': timestamp,
    '训练时间': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    '模型类型': 'MLP',
    '隐藏层结构': '128-64-32',
    '总参数量': sum(p.numel() for p in model.parameters()),
    '训练集大小': len(X_train),
    '验证集大小': len(X_val),
    '测试集大小': len(X_test),
    '最佳轮数': best_epoch,
    '总训练轮数': len(train_losses),
    '最佳验证准确率': f"{best_val_accuracy:.2f}%",
    '最终测试准确率': f"{final_accuracy:.2f}%",
    '学习率': 0.001,
    'Dropout': 0.3,
    '批次大小': 256,
    '早停耐心值': patience,
    '是否使用类别权重': 'Yes',
    '模型路径': model_path,
    '训练曲线路径': curve_path
}

# 保存到CSV
csv_path = 'MLP记录.csv'
record_df = pd.DataFrame([record])

if os.path.exists(csv_path):
    # 如果文件存在，追加（尝试多种编码）
    try:
        existing_df = pd.read_csv(csv_path, encoding='utf-8-sig')
    except UnicodeDecodeError:
        try:
            existing_df = pd.read_csv(csv_path, encoding='gbk')
        except:
            existing_df = pd.read_csv(csv_path, encoding='latin1')
    updated_df = pd.concat([existing_df, record_df], ignore_index=True)
    updated_df.to_csv(csv_path, index=False, encoding='utf-8-sig')
    print(f"训练记录已追加到 {csv_path}")
else:
    # 如果文件不存在，创建新文件
    record_df.to_csv(csv_path, index=False, encoding='utf-8-sig')
    print(f"训练记录已保存到 {csv_path}")

# 16. 打印总结
print("\n" + "="*50)
print("训练完成总结:")
print("="*50)
print(f"模型保存路径: {model_path}")
print(f"训练曲线路径: {curve_path}")
print(f"最佳验证准确率: {best_val_accuracy:.2f}% (第{best_epoch}轮)")
print(f"最终测试准确率: {final_accuracy:.2f}%")
print(f"训练记录已保存到: {csv_path}")
print("="*50)
