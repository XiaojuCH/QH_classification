import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import os

# 设置随机种子
np.random.seed(42)

os.makedirs('models', exist_ok=True)
os.makedirs('results', exist_ok=True)

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
print(f"训练时间戳: {timestamp}\n")

# 1. 加载数据
print("正在加载数据...")
df = pd.read_csv('样品特征数据集.csv', header=None)
X = df.iloc[:, :20].values
y = df.iloc[:, 20].values - 1  # 转换为0-12
print(f"数据集: {X.shape}, 类别数: {len(np.unique(y))}")

# 2. 划分数据集
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.15, random_state=42, stratify=y
)
X_train, X_val, y_train, y_val = train_test_split(
    X_train, y_train, test_size=0.15/0.85, random_state=42, stratify=y_train
)

print(f"训练集: {X_train.shape[0]}, 验证集: {X_val.shape[0]}, 测试集: {X_test.shape[0]}\n")

# 3. 创建LightGBM数据集（不使用样本权重，改用is_unbalance参数）
train_data = lgb.Dataset(X_train, label=y_train)
val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

# 4. 设置参数（平衡性能和泛化）
params = {
    'objective': 'multiclass',
    'num_class': 13,
    'metric': 'multi_logloss',
    'boosting_type': 'gbdt',
    'num_leaves': 50,  # 适中的复杂度
    'learning_rate': 0.05,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'max_depth': 10,
    'min_child_samples': 30,
    'lambda_l1': 0.05,  # 轻度正则化
    'lambda_l2': 0.05,
    'is_unbalance': True,  # 自动处理类别不平衡
    'verbose': -1
}

print("LightGBM参数:")
for k, v in params.items():
    print(f"  {k}: {v}")
print()

# 5. 训练模型
print("开始训练...\n")
evals_result = {}
model = lgb.train(
    params,
    train_data,
    num_boost_round=1000,
    valid_sets=[train_data, val_data],
    valid_names=['train', 'valid'],
    callbacks=[
        lgb.early_stopping(stopping_rounds=50),
        lgb.log_evaluation(period=20),
        lgb.record_evaluation(evals_result)
    ]
)

# 6. 保存模型
model_path = f'models/lightgbm_{timestamp}.txt'
model.save_model(model_path)
print(f"\n模型已保存: {model_path}")

# 7. 预测和评估
print("\n" + "="*50)
print("评估结果:")
print("="*50)

y_train_pred = model.predict(X_train).argmax(axis=1)
y_val_pred = model.predict(X_val).argmax(axis=1)
y_test_pred = model.predict(X_test).argmax(axis=1)

train_acc = accuracy_score(y_train, y_train_pred) * 100
val_acc = accuracy_score(y_val, y_val_pred) * 100
test_acc = accuracy_score(y_test, y_test_pred) * 100

print(f"训练集准确率: {train_acc:.2f}%")
print(f"验证集准确率: {val_acc:.2f}%")
print(f"测试集准确率: {test_acc:.2f}%\n")

# 8. 分类报告
print("测试集分类报告:")
print(classification_report(y_test + 1, y_test_pred + 1))

# 9. 特征重要性
feature_importance = model.feature_importance(importance_type='gain')
feature_names = [f'Feature_{i}' for i in range(20)]

plt.figure(figsize=(10, 6))
indices = np.argsort(feature_importance)[::-1][:20]
plt.barh(range(len(indices)), feature_importance[indices])
plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
plt.xlabel('Feature Importance (Gain)')
plt.title('Top 20 Feature Importance')
plt.tight_layout()
importance_path = f'results/feature_importance_{timestamp}.png'
plt.savefig(importance_path, dpi=300)
print(f"特征重要性图: {importance_path}")

# 10. 训练曲线
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(evals_result['train']['multi_logloss'], label='Train')
plt.plot(evals_result['valid']['multi_logloss'], label='Valid')
plt.xlabel('Iterations')
plt.ylabel('Multi-logloss')
plt.title('Training Curve')
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
plt.bar(['Train', 'Valid', 'Test'], [train_acc, val_acc, test_acc])
plt.ylabel('Accuracy (%)')
plt.title('Model Performance')
plt.ylim([0, 100])
for i, v in enumerate([train_acc, val_acc, test_acc]):
    plt.text(i, v + 1, f'{v:.2f}%', ha='center')

plt.tight_layout()
curve_path = f'results/training_curves_{timestamp}.png'
plt.savefig(curve_path, dpi=300)
print(f"训练曲线: {curve_path}")

# 11. 记录到CSV
record = {
    '时间戳': timestamp,
    '训练时间': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    '模型类型': 'LightGBM',
    '训练集大小': len(X_train),
    '验证集大小': len(X_val),
    '测试集大小': len(X_test),
    '训练准确率': f"{train_acc:.2f}%",
    '验证准确率': f"{val_acc:.2f}%",
    '测试准确率': f"{test_acc:.2f}%",
    '最佳迭代轮数': model.best_iteration,
    '学习率': params['learning_rate'],
    'num_leaves': params['num_leaves'],
    '模型路径': model_path,
    '训练曲线路径': curve_path
}

csv_path = 'LightGBM记录.csv'
record_df = pd.DataFrame([record])

if os.path.exists(csv_path):
    existing_df = pd.read_csv(csv_path, encoding='utf-8-sig')
    updated_df = pd.concat([existing_df, record_df], ignore_index=True)
    updated_df.to_csv(csv_path, index=False, encoding='utf-8-sig')
else:
    record_df.to_csv(csv_path, index=False, encoding='utf-8-sig')

print(f"训练记录: {csv_path}")
print("="*50)
print(f"最终测试准确率: {test_acc:.2f}%")
print("="*50)
