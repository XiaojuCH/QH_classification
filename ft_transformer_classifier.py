import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score
from sklearn.utils.class_weight import compute_class_weight
import matplotlib.pyplot as plt
from datetime import datetime
import os
import warnings

warnings.filterwarnings('ignore')
torch.manual_seed(42)
np.random.seed(42)

os.makedirs('models', exist_ok=True)
os.makedirs('results', exist_ok=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用设备: {device}")

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
print(f"训练时间戳: {timestamp}\n")

# ==================== 1. 加载数据 ====================
print("正在加载数据...")
df = pd.read_csv('样品特征数据集.csv', header=None)
X = df.iloc[:, :20].values
y = df.iloc[:, 20].values - 1  # 转换为0-12
print(f"数据集: {X.shape}, 类别数: {len(np.unique(y))}")

# ==================== 2. 数据预处理 ====================
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.15, random_state=42, stratify=y
)
X_train, X_val, y_train, y_val = train_test_split(
    X_train, y_train, test_size=0.15/0.85, random_state=42, stratify=y_train
)

print(f"训练集: {X_train.shape[0]}, 验证集: {X_val.shape[0]}, 测试集: {X_test.shape[0]}\n")

# 计算类别权重
class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
class_weights_tensor = torch.FloatTensor(class_weights).to(device)

# 转换为张量
X_train_tensor = torch.FloatTensor(X_train).to(device)
y_train_tensor = torch.LongTensor(y_train).to(device)
X_val_tensor = torch.FloatTensor(X_val).to(device)
y_val_tensor = torch.LongTensor(y_val).to(device)
X_test_tensor = torch.FloatTensor(X_test).to(device)
y_test_tensor = torch.LongTensor(y_test).to(device)

# 数据加载器
batch_size = 512
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# ==================== 3. FT-Transformer模型 ====================
class FeatureTokenizer(nn.Module):
    """将每个特征转换为token embedding"""
    def __init__(self, num_features, d_token):
        super().__init__()
        self.num_features = num_features
        self.d_token = d_token
        # 为每个特征创建一个线性层
        self.feature_embeddings = nn.ModuleList([
            nn.Linear(1, d_token) for _ in range(num_features)
        ])
        # 可学习的bias
        self.bias = nn.Parameter(torch.zeros(num_features, d_token))

    def forward(self, x):
        # x: [batch_size, num_features]
        batch_size = x.shape[0]
        tokens = []
        for i in range(self.num_features):
            # 取出第i个特征: [batch_size, 1]
            feature = x[:, i:i+1]
            # 通过对应的embedding层: [batch_size, d_token]
            token = self.feature_embeddings[i](feature) + self.bias[i]
            tokens.append(token)
        # 堆叠所有tokens: [batch_size, num_features, d_token]
        tokens = torch.stack(tokens, dim=1)
        return tokens


class FTTransformer(nn.Module):
    """Feature Tokenizer + Transformer for Tabular Data"""
    def __init__(self, num_features, num_classes, d_token=192, n_layers=3, n_heads=8,
                 d_ffn_factor=4, dropout=0.1):
        super().__init__()

        # Feature Tokenizer
        self.feature_tokenizer = FeatureTokenizer(num_features, d_token)

        # CLS token (用于分类)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_token))

        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_token,
            nhead=n_heads,
            dim_feedforward=d_token * d_ffn_factor,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # 分类头
        self.head = nn.Sequential(
            nn.LayerNorm(d_token),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_token, num_classes)
        )

    def forward(self, x):
        # Feature tokenization: [batch_size, num_features, d_token]
        tokens = self.feature_tokenizer(x)

        # 添加CLS token: [batch_size, num_features+1, d_token]
        batch_size = tokens.shape[0]
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        tokens = torch.cat([cls_tokens, tokens], dim=1)

        # Transformer encoding
        encoded = self.transformer(tokens)

        # 取CLS token的输出用于分类
        cls_output = encoded[:, 0, :]

        # 分类
        logits = self.head(cls_output)
        return logits


# ==================== 4. 创建模型 ====================
model = FTTransformer(
    num_features=20,
    num_classes=13,
    d_token=192,      # token维度
    n_layers=3,       # Transformer层数
    n_heads=8,        # 注意力头数
    d_ffn_factor=4,   # FFN维度倍数
    dropout=0.2
).to(device)

total_params = sum(p.numel() for p in model.parameters())
print(f"FT-Transformer模型")
print(f"模型参数: {total_params:,}")
print(f"Token维度: 192, Transformer层数: 3, 注意力头数: 8\n")

# ==================== 5. 训练设置 ====================
criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)

print("损失函数: CrossEntropyLoss (类别权重)")
print("优化器: AdamW + CosineAnnealingWarmRestarts\n")

# ==================== 6. 训练 ====================
num_epochs = 100
train_losses, train_accs, val_accs = [], [], []
best_val_acc = 0.0
best_epoch = 0
patience = 20
patience_counter = 0
model_path = f'models/ft_transformer_{timestamp}.pth'

print("="*60)
print("开始训练...")
print("="*60)

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

# ==================== 7. 测试 ====================
print("\n" + "="*60)
print("最终测试...")
print("="*60)

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

# ==================== 8. 可视化 ====================
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
curve_path = f'results/ft_transformer_curves_{timestamp}.png'
plt.savefig(curve_path, dpi=300)
print(f"训练曲线: {curve_path}")

# ==================== 9. 保存记录 ====================
record = {
    '时间戳': timestamp,
    '训练时间': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    '模型类型': 'FT-Transformer',
    '模型参数量': total_params,
    'Token维度': 192,
    'Transformer层数': 3,
    '注意力头数': 8,
    '训练集大小': len(X_train),
    '验证集大小': len(X_val),
    '测试集大小': len(X_test),
    '最佳轮数': best_epoch,
    '总训练轮数': len(train_losses),
    '最佳验证准确率': f"{best_val_acc:.2f}%",
    '最终测试准确率': f"{final_acc:.2f}%",
    '学习率': '1e-4 (Cosine)',
    'Dropout': 0.2,
    '批次大小': batch_size,
    '早停耐心值': patience,
    '优化器': 'AdamW',
    '模型路径': model_path,
    '训练曲线路径': curve_path
}

csv_path = 'FT-Transformer记录.csv'
record_df = pd.DataFrame([record])

if os.path.exists(csv_path):
    existing_df = pd.read_csv(csv_path, encoding='utf-8-sig')
    updated_df = pd.concat([existing_df, record_df], ignore_index=True)
    updated_df.to_csv(csv_path, index=False, encoding='utf-8-sig')
else:
    record_df.to_csv(csv_path, index=False, encoding='utf-8-sig')

print(f"训练记录: {csv_path}")
print("="*60)
print(f"最佳验证: {best_val_acc:.2f}% (第{best_epoch}轮)")
print(f"最终测试: {final_acc:.2f}%")
print("="*60)
