import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split
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
print("最终优化方案：精选特征工程")
print("="*60)
print(f"时间戳: {timestamp}\n")

# ==================== 1. 加载数据 ====================
print("1. 加载数据...")
df = pd.read_csv('样品特征数据集.csv', header=None)
X = df.iloc[:, :20].values
y = df.iloc[:, 20].values - 1
print(f"   原始数据: {X.shape}, 类别数: {len(np.unique(y))}")

# ==================== 2. 精选特征工程 ====================
print("\n2. 精选特征工程...")

# 2.1 三种标准化方法
scaler_std = StandardScaler()
scaler_robust = RobustScaler()  # 对异常值更鲁棒
scaler_quantile = QuantileTransformer(output_distribution='normal', random_state=42)

X_std = scaler_std.fit_transform(X)
X_robust = scaler_robust.fit_transform(X)
X_quantile = scaler_quantile.fit_transform(X)

# 2.2 只保留最重要的5个原始特征的多种变换
important_indices = [0, 1, 2, 3, 5]  # Feature 0,1,2,3,5最重要
X_important_std = X_std[:, important_indices]
X_important_robust = X_robust[:, important_indices]
X_important_quantile = X_quantile[:, important_indices]

# 2.3 特征交互（只做最重要的组合）
X_interactions = []
for i in range(len(important_indices)):
    for j in range(i+1, len(important_indices)):
        X_interactions.append(X_std[:, important_indices[i]] * X_std[:, important_indices[j]])
X_interactions = np.column_stack(X_interactions)

# 2.4 比率特征（只做最重要的组合）
X_ratios = []
for i in range(len(important_indices)):
    for j in range(i+1, len(important_indices)):
        ratio = X[:, important_indices[i]] / (X[:, important_indices[j]] + 1e-8)
        X_ratios.append(ratio)
X_ratios = np.column_stack(X_ratios)

# 2.5 统计特征（全局）
X_stats = np.column_stack([
    np.mean(X_std, axis=1),
    np.std(X_std, axis=1),
    np.max(X_std, axis=1),
    np.min(X_std, axis=1),
    np.max(X_std, axis=1) - np.min(X_std, axis=1),  # 极差
])

# 2.6 平方和立方特征（捕捉非线性）
X_squared = X_std[:, important_indices] ** 2
X_cubed = X_std[:, important_indices] ** 3

# 2.7 合并所有特征
X_enhanced = np.hstack([
    X_std,                    # 20个标准化特征
    X_important_robust,       # 5个鲁棒标准化特征
    X_important_quantile,     # 5个Quantile特征
    X_interactions,           # 10个交互特征
    X_ratios,                 # 10个比率特征
    X_stats,                  # 5个统计特征
    X_squared,                # 5个平方特征
    X_cubed                   # 5个立方特征
])

print(f"   [OK] 特征工程完成: {X.shape[1]} -> {X_enhanced.shape[1]} 维")
print(f"       - 标准化: 20")
print(f"       - 鲁棒标准化: 5")
print(f"       - Quantile: 5")
print(f"       - 交互: {X_interactions.shape[1]}")
print(f"       - 比率: {X_ratios.shape[1]}")
print(f"       - 统计: {X_stats.shape[1]}")
print(f"       - 平方: {X_squared.shape[1]}")
print(f"       - 立方: {X_cubed.shape[1]}")

# ==================== 3. 划分数据集 ====================
print("\n3. 划分数据集...")
X_train, X_test, y_train, y_test = train_test_split(
    X_enhanced, y, test_size=0.15, random_state=42, stratify=y
)
X_train, X_val, y_train, y_val = train_test_split(
    X_train, y_train, test_size=0.15/0.85, random_state=42, stratify=y_train
)
print(f"   训练集: {X_train.shape[0]}, 验证集: {X_val.shape[0]}, 测试集: {X_test.shape[0]}")

# ==================== 4. LightGBM训练（针对高维特征优化）====================
print("\n4. 训练LightGBM...")
print("="*60)

train_data = lgb.Dataset(X_train, label=y_train)
val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

# 针对65维特征优化的参数
params = {
    'objective': 'multiclass',
    'num_class': 13,
    'metric': 'multi_logloss',
    'boosting_type': 'gbdt',
    'num_leaves': 50,
    'learning_rate': 0.04,
    'feature_fraction': 0.75,  # 特征多，降低采样
    'bagging_fraction': 0.85,
    'bagging_freq': 5,
    'max_depth': 10,
    'min_child_samples': 25,
    'lambda_l1': 0.08,  # 增加正则化
    'lambda_l2': 0.08,
    'min_gain_to_split': 0.008,
    'max_bin': 255,
    'is_unbalance': True,
    'feature_pre_filter': False,
    'verbose': -1
}

print("训练参数:")
for k, v in params.items():
    if k not in ['verbose', 'feature_pre_filter']:
        print(f"  {k}: {v}")

evals_result = {}
model = lgb.train(
    params,
    train_data,
    num_boost_round=1500,  # 增加轮数
    valid_sets=[train_data, val_data],
    valid_names=['train', 'valid'],
    callbacks=[
        lgb.early_stopping(stopping_rounds=60),  # 增加耐心
        lgb.log_evaluation(period=50),
        lgb.record_evaluation(evals_result)
    ]
)

print(f"\n最佳迭代轮数: {model.best_iteration}")

# ==================== 5. 评估 ====================
print("\n" + "="*60)
print("评估结果")
print("="*60)

y_train_pred = model.predict(X_train).argmax(axis=1)
y_val_pred = model.predict(X_val).argmax(axis=1)
y_test_pred = model.predict(X_test).argmax(axis=1)

train_acc = accuracy_score(y_train, y_train_pred) * 100
val_acc = accuracy_score(y_val, y_val_pred) * 100
test_acc = accuracy_score(y_test, y_test_pred) * 100

print(f"训练集准确率: {train_acc:.2f}%")
print(f"验证集准确率: {val_acc:.2f}%")
print(f"测试集准确率: {test_acc:.2f}%")
print(f"过拟合程度: {train_acc - test_acc:.2f}%")

print("\n测试集分类报告:")
print(classification_report(y_test + 1, y_test_pred + 1))

# ==================== 6. 特征重要性 ====================
print("\n" + "="*60)
print("Top 15 特征重要性:")
print("="*60)

feature_importance = model.feature_importance(importance_type='gain')
feature_names = (
    [f'标准_{i}' for i in range(20)] +
    [f'鲁棒_{i}' for i in important_indices] +
    [f'Quantile_{i}' for i in important_indices] +
    [f'交互_{i}' for i in range(X_interactions.shape[1])] +
    [f'比率_{i}' for i in range(X_ratios.shape[1])] +
    ['均值', '标准差', '最大值', '最小值', '极差'] +
    [f'平方_{i}' for i in important_indices] +
    [f'立方_{i}' for i in important_indices]
)

indices = np.argsort(feature_importance)[::-1][:15]
for i, idx in enumerate(indices):
    print(f"  {i+1}. {feature_names[idx]}: {feature_importance[idx]:.1f}")

# ==================== 7. 对比基线 ====================
print("\n" + "="*60)
print("与基线对比")
print("="*60)

baseline_acc = 65.0
improvement = test_acc - baseline_acc

print(f"基线准确率 (原始特征):  {baseline_acc:.2f}%")
print(f"优化后准确率 (本方法):  {test_acc:.2f}%")
print(f"提升幅度:              {improvement:+.2f}%")

if improvement > 1:
    print("\n[SUCCESS] 有提升！特征工程有效")
elif improvement > -0.5:
    print("\n[OK] 基本持平，特征工程部分有效")
else:
    print("\n[WARNING] 未见提升")

# ==================== 8. 保存模型 ====================
model_path = f'models/final_lgb_{timestamp}.txt'
model.save_model(model_path)
print(f"\n模型已保存: {model_path}")

# ==================== 9. 保存记录 ====================
record = {
    '时间戳': timestamp,
    '训练时间': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    '模型类型': 'LightGBM-Final',
    '数据优化': '精选特征工程(65维)+无数据清洗',
    '原始特征数': 20,
    '最终特征数': X_enhanced.shape[1],
    '训练集大小': len(X_train),
    '验证集大小': len(X_val),
    '测试集大小': len(X_test),
    '训练准确率': f"{train_acc:.2f}%",
    '验证准确率': f"{val_acc:.2f}%",
    '测试准确率': f"{test_acc:.2f}%",
    '基线准确率': f"{baseline_acc:.2f}%",
    '提升幅度': f"{improvement:+.2f}%",
    '最佳迭代轮数': model.best_iteration,
    '模型路径': model_path
}

csv_path = '最终优化记录.csv'
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
