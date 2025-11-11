import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, QuantileTransformer
from sklearn.metrics import classification_report, accuracy_score
from sklearn.cluster import KMeans
from datetime import datetime
import os
import warnings

warnings.filterwarnings('ignore')
np.random.seed(42)

os.makedirs('models', exist_ok=True)
os.makedirs('results', exist_ok=True)

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
print("="*60)
print("高级优化方案：特征扩展 + 数据清洗")
print("="*60)
print(f"时间戳: {timestamp}\n")

# ==================== 1. 加载数据 ====================
print("1. 加载数据...")
df = pd.read_csv('样品特征数据集.csv', header=None)
X = df.iloc[:, :20].values
y = df.iloc[:, 20].values - 1
print(f"   原始数据: {X.shape}, 类别数: {len(np.unique(y))}")

# ==================== 2. 特征扩展（不降维，而是增加） ====================
print("\n2. 特征扩展...")

# 2.1 标准化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 2.2 Quantile Transform (使数据更均匀分布)
quantile_transformer = QuantileTransformer(output_distribution='normal', random_state=42)
X_quantile = quantile_transformer.fit_transform(X)

# 2.3 特征交互（只选择最重要的几对）
# 根据之前的特征重要性，选择前5个特征做交互
important_features = [0, 1, 2, 5, 19]  # Feature_0, 1, 2, 5, 19最重要
X_interactions = []
for i in range(len(important_features)):
    for j in range(i+1, len(important_features)):
        feat_i = important_features[i]
        feat_j = important_features[j]
        X_interactions.append(X_scaled[:, feat_i] * X_scaled[:, feat_j])
X_interactions = np.column_stack(X_interactions)

# 2.4 统计特征
X_stats = np.column_stack([
    np.mean(X_scaled, axis=1),
    np.std(X_scaled, axis=1),
    np.max(X_scaled, axis=1),
    np.min(X_scaled, axis=1),
    np.median(X_scaled, axis=1),
    np.percentile(X_scaled, 25, axis=1),
    np.percentile(X_scaled, 75, axis=1)
])

# 2.5 比率特征
X_ratios = []
for i in range(5):
    for j in range(i+1, 5):
        feat_i = important_features[i]
        feat_j = important_features[j]
        # 避免除零
        ratio = X[:, feat_i] / (X[:, feat_j] + 1e-8)
        X_ratios.append(ratio)
X_ratios = np.column_stack(X_ratios)

# 2.6 合并所有特征
X_enhanced = np.hstack([
    X_scaled,           # 20个原始特征
    X_quantile,         # 20个Quantile变换特征
    X_interactions,     # 10个交互特征
    X_stats,            # 7个统计特征
    X_ratios            # 10个比率特征
])

print(f"   [OK] 特征扩展完成: {X.shape[1]} -> {X_enhanced.shape[1]} 维")
print(f"       - 原始特征: 20")
print(f"       - Quantile变换: 20")
print(f"       - 交互特征: {X_interactions.shape[1]}")
print(f"       - 统计特征: {X_stats.shape[1]}")
print(f"       - 比率特征: {X_ratios.shape[1]}")

# ==================== 3. 数据清洗：去除边界噪声样本 ====================
print("\n3. 数据清洗（去除噪声样本）...")

# 使用KMeans找出每个类别的中心，去除离中心太远的样本
X_cleaned = []
y_cleaned = []

for cls in np.unique(y):
    class_mask = (y == cls)
    class_data = X_enhanced[class_mask]

    if len(class_data) < 100:  # 小类别不清洗
        X_cleaned.append(class_data)
        y_cleaned.append(np.full(len(class_data), cls))
        continue

    # 计算到类中心的距离
    center = np.mean(class_data, axis=0)
    distances = np.linalg.norm(class_data - center, axis=1)

    # 去除最远的5%样本（可能是噪声或边界样本）
    threshold = np.percentile(distances, 95)
    clean_mask = distances <= threshold

    X_cleaned.append(class_data[clean_mask])
    y_cleaned.append(np.full(np.sum(clean_mask), cls))

X_cleaned = np.vstack(X_cleaned)
y_cleaned = np.concatenate(y_cleaned)

removed_samples = len(X_enhanced) - len(X_cleaned)
print(f"   [OK] 清洗完成: 移除 {removed_samples} 个噪声样本 ({removed_samples/len(X_enhanced)*100:.2f}%)")

# ==================== 4. 划分数据集 ====================
print("\n4. 划分数据集...")
X_train, X_test, y_train, y_test = train_test_split(
    X_cleaned, y_cleaned, test_size=0.15, random_state=42, stratify=y_cleaned
)
X_train, X_val, y_train, y_val = train_test_split(
    X_train, y_train, test_size=0.15/0.85, random_state=42, stratify=y_train
)
print(f"   训练集: {X_train.shape[0]}, 验证集: {X_val.shape[0]}, 测试集: {X_test.shape[0]}")

# ==================== 5. LightGBM训练 ====================
print("\n5. 训练LightGBM...")
print("="*60)

train_data = lgb.Dataset(X_train, label=y_train)
val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

# 针对高维特征优化的参数
params = {
    'objective': 'multiclass',
    'num_class': 13,
    'metric': 'multi_logloss',
    'boosting_type': 'gbdt',
    'num_leaves': 60,
    'learning_rate': 0.03,
    'feature_fraction': 0.7,  # 特征多了，降低采样率
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'max_depth': 12,
    'min_child_samples': 20,
    'lambda_l1': 0.05,
    'lambda_l2': 0.05,
    'min_gain_to_split': 0.005,
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
    num_boost_round=1000,
    valid_sets=[train_data, val_data],
    valid_names=['train', 'valid'],
    callbacks=[
        lgb.early_stopping(stopping_rounds=50),
        lgb.log_evaluation(period=50),
        lgb.record_evaluation(evals_result)
    ]
)

print(f"\n最佳迭代轮数: {model.best_iteration}")

# ==================== 6. 评估 ====================
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

# ==================== 7. 特征重要性分析 ====================
print("\n" + "="*60)
print("Top 20 特征重要性:")
print("="*60)

feature_importance = model.feature_importance(importance_type='gain')
feature_names = (
    [f'原始_{i}' for i in range(20)] +
    [f'Quantile_{i}' for i in range(20)] +
    [f'交互_{i}' for i in range(X_interactions.shape[1])] +
    ['均值', '标准差', '最大值', '最小值', '中位数', 'Q25', 'Q75'] +
    [f'比率_{i}' for i in range(X_ratios.shape[1])]
)

indices = np.argsort(feature_importance)[::-1][:20]
for i, idx in enumerate(indices):
    print(f"  {i+1}. {feature_names[idx]}: {feature_importance[idx]:.1f}")

# ==================== 8. 对比基线 ====================
print("\n" + "="*60)
print("与基线对比")
print("="*60)

baseline_acc = 65.0
improvement = test_acc - baseline_acc

print(f"基线准确率 (原始特征):  {baseline_acc:.2f}%")
print(f"优化后准确率 (本方法):  {test_acc:.2f}%")
print(f"提升幅度:              {improvement:+.2f}%")

if improvement > 2:
    print("\n[SUCCESS] 显著提升！特征扩展+数据清洗有效")
elif improvement > 0:
    print("\n[OK] 有小幅提升")
else:
    print("\n[WARNING] 未见提升")

# ==================== 9. 保存模型 ====================
model_path = f'models/advanced_lgb_{timestamp}.txt'
model.save_model(model_path)
print(f"\n模型已保存: {model_path}")

# ==================== 10. 保存记录 ====================
record = {
    '时间戳': timestamp,
    '训练时间': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    '模型类型': 'LightGBM-Advanced',
    '数据优化': '特征扩展(67维)+数据清洗(去除5%噪声)',
    '原始特征数': 20,
    '最终特征数': X_enhanced.shape[1],
    '清洗前样本数': len(X_enhanced),
    '清洗后样本数': len(X_cleaned),
    '移除样本数': removed_samples,
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

csv_path = '高级优化记录.csv'
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
