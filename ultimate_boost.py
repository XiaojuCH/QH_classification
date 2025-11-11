import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler, QuantileTransformer, RobustScaler
from sklearn.metrics import classification_report, accuracy_score
from datetime import datetime
import os
import warnings

warnings.filterwarnings('ignore')
np.random.seed(42)

os.makedirs('models', exist_ok=True)
os.makedirs('results', exist_ok=True)

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
print("="*60)
print("终极冲刺：特征工程 + 5折交叉验证集成")
print("="*60)
print(f"时间戳: {timestamp}\n")

# ==================== 1. 加载数据 ====================
print("1. 加载数据...")
df = pd.read_csv('样品特征数据集.csv', header=None)
X = df.iloc[:, :20].values
y = df.iloc[:, 20].values - 1
print(f"   原始数据: {X.shape}, 类别数: {len(np.unique(y))}")

# ==================== 2. 增强特征工程 ====================
print("\n2. 增强特征工程...")

# 2.1 多种标准化
scaler_std = StandardScaler()
scaler_robust = RobustScaler()
scaler_quantile = QuantileTransformer(output_distribution='normal', random_state=42)

X_std = scaler_std.fit_transform(X)
X_robust = scaler_robust.fit_transform(X)
X_quantile = scaler_quantile.fit_transform(X)

# 2.2 根据特征重要性，选择最重要的特征
# 从之前的结果看：Feature 19, 9, 4, 5, 14, 6, 16最重要
important_indices = [19, 9, 4, 5, 14, 6, 16, 18, 15, 0]  # Top 10
X_important = X_std[:, important_indices]

# 2.3 特征交互（只做最重要的）
X_interactions = []
for i in range(min(5, len(important_indices))):
    for j in range(i+1, min(5, len(important_indices))):
        X_interactions.append(X_std[:, important_indices[i]] * X_std[:, important_indices[j]])
X_interactions = np.column_stack(X_interactions)

# 2.4 比率特征
X_ratios = []
for i in range(min(5, len(important_indices))):
    for j in range(i+1, min(5, len(important_indices))):
        ratio = X[:, important_indices[i]] / (X[:, important_indices[j]] + 1e-8)
        X_ratios.append(ratio)
X_ratios = np.column_stack(X_ratios)

# 2.5 统计特征
X_stats = np.column_stack([
    np.mean(X_std, axis=1),
    np.std(X_std, axis=1),
    np.max(X_std, axis=1),
    np.min(X_std, axis=1),
    np.max(X_std, axis=1) - np.min(X_std, axis=1),
    np.median(X_std, axis=1)
])

# 2.6 非线性变换
X_squared = X_important ** 2
X_cubed = X_important ** 3
X_sqrt = np.sqrt(np.abs(X_important))

# 2.7 对数变换（处理偏态）
X_log = np.log1p(np.abs(X_important))

# 2.8 合并所有特征
X_enhanced = np.hstack([
    X_std,              # 20
    X_important,        # 10
    X_robust[:, important_indices],  # 10
    X_quantile[:, important_indices],  # 10
    X_interactions,     # 10
    X_ratios,           # 10
    X_stats,            # 6
    X_squared,          # 10
    X_cubed,            # 10
    X_sqrt,             # 10
    X_log               # 10
])

print(f"   [OK] 特征工程完成: {X.shape[1]} -> {X_enhanced.shape[1]} 维")

# ==================== 3. 划分数据集 ====================
print("\n3. 划分数据集...")
X_train_full, X_test, y_train_full, y_test = train_test_split(
    X_enhanced, y, test_size=0.15, random_state=42, stratify=y
)
print(f"   训练集: {X_train_full.shape[0]}, 测试集: {X_test.shape[0]}")

# ==================== 4. 5折交叉验证训练 ====================
print("\n4. 5折交叉验证训练...")
print("="*60)

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
models = []
val_scores = []

params = {
    'objective': 'multiclass',
    'num_class': 13,
    'metric': 'multi_logloss',
    'boosting_type': 'gbdt',
    'num_leaves': 45,
    'learning_rate': 0.035,
    'feature_fraction': 0.75,
    'bagging_fraction': 0.85,
    'bagging_freq': 5,
    'max_depth': 10,
    'min_child_samples': 25,
    'lambda_l1': 0.1,
    'lambda_l2': 0.1,
    'min_gain_to_split': 0.01,
    'max_bin': 255,
    'is_unbalance': True,
    'feature_pre_filter': False,
    'verbose': -1
}

for fold, (train_idx, val_idx) in enumerate(skf.split(X_train_full, y_train_full), 1):
    print(f"\n训练 Fold {fold}/5...")

    X_train = X_train_full[train_idx]
    X_val = X_train_full[val_idx]
    y_train = y_train_full[train_idx]
    y_val = y_train_full[val_idx]

    train_data = lgb.Dataset(X_train, label=y_train)
    val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

    model = lgb.train(
        params,
        train_data,
        num_boost_round=1500,
        valid_sets=[val_data],
        callbacks=[
            lgb.early_stopping(stopping_rounds=60),
            lgb.log_evaluation(period=0)  # 不打印
        ]
    )

    models.append(model)

    # 验证集评估
    y_val_pred = model.predict(X_val).argmax(axis=1)
    val_acc = accuracy_score(y_val, y_val_pred) * 100
    val_scores.append(val_acc)

    print(f"  Fold {fold} 验证准确率: {val_acc:.2f}%")

print(f"\n平均验证准确率: {np.mean(val_scores):.2f}% ± {np.std(val_scores):.2f}%")

# ==================== 5. 集成预测 ====================
print("\n5. 集成预测...")
print("="*60)

# 对测试集进行5个模型的平均预测
test_preds = []
for i, model in enumerate(models, 1):
    pred_proba = model.predict(X_test)
    test_preds.append(pred_proba)

# 平均概率
test_pred_avg = np.mean(test_preds, axis=0)
y_test_pred = test_pred_avg.argmax(axis=1)

test_acc = accuracy_score(y_test, y_test_pred) * 100

print(f"测试集准确率 (5折集成): {test_acc:.2f}%")

print("\n测试集分类报告:")
print(classification_report(y_test + 1, y_test_pred + 1))

# ==================== 6. 对比基线 ====================
print("\n" + "="*60)
print("与基线对比")
print("="*60)

baseline_acc = 65.0
improvement = test_acc - baseline_acc

print(f"基线准确率 (原始特征):     {baseline_acc:.2f}%")
print(f"优化后准确率 (5折集成):    {test_acc:.2f}%")
print(f"提升幅度:                  {improvement:+.2f}%")

if improvement > 1:
    print("\n[SUCCESS] 显著提升！")
elif improvement > 0:
    print("\n[OK] 有提升")
else:
    print("\n[WARNING] 未见提升")

# ==================== 7. 保存最佳模型 ====================
best_fold = np.argmax(val_scores)
best_model = models[best_fold]
model_path = f'models/ultimate_lgb_{timestamp}.txt'
best_model.save_model(model_path)
print(f"\n最佳模型 (Fold {best_fold+1}) 已保存: {model_path}")

# ==================== 8. 保存记录 ====================
record = {
    '时间戳': timestamp,
    '训练时间': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    '模型类型': 'LightGBM-Ultimate (5-Fold Ensemble)',
    '数据优化': '增强特征工程(106维)+5折交叉验证',
    '原始特征数': 20,
    '最终特征数': X_enhanced.shape[1],
    '训练集大小': len(X_train_full),
    '测试集大小': len(X_test),
    '平均验证准确率': f"{np.mean(val_scores):.2f}%",
    '测试准确率': f"{test_acc:.2f}%",
    '基线准确率': f"{baseline_acc:.2f}%",
    '提升幅度': f"{improvement:+.2f}%",
    '模型路径': model_path
}

csv_path = '终极优化记录.csv'
record_df = pd.DataFrame([record])

if os.path.exists(csv_path):
    existing_df = pd.read_csv(csv_path, encoding='utf-8-sig')
    updated_df = pd.concat([existing_df, record_df], ignore_index=True)
    updated_df.to_csv(csv_path, index=False, encoding='utf-8-sig')
else:
    record_df.to_csv(csv_path, index=False, encoding='utf-8-sig')

print(f"训练记录: {csv_path}")
print("="*60)
print(f"最终测试准确率: {test_acc:.2f}%")
print(f"相比基线提升: {improvement:+.2f}%")
print("="*60)
