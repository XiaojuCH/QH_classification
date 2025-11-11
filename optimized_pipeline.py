import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, PowerTransformer
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import classification_report, accuracy_score
from imblearn.over_sampling import ADASYN
from datetime import datetime
import os
import warnings

warnings.filterwarnings('ignore')
np.random.seed(42)

os.makedirs('models', exist_ok=True)
os.makedirs('results', exist_ok=True)

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
print("="*60)
print("优化数据处理流程 + LightGBM")
print("="*60)
print(f"时间戳: {timestamp}\n")

# ==================== 1. 加载数据 ====================
print("1. 加载数据...")
df = pd.read_csv('样品特征数据集.csv', header=None)
X = df.iloc[:, :20].values
y = df.iloc[:, 20].values - 1
print(f"   原始数据: {X.shape}, 类别数: {len(np.unique(y))}")

# ==================== 2. 数据预处理 ====================
print("\n2. 数据预处理...")

# 2.1 标准化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
print("   [OK] 标准化完成")

# 2.2 Power Transform (处理偏态分布)
power_transformer = PowerTransformer(method='yeo-johnson')
X_power = power_transformer.fit_transform(X_scaled)
print("   [OK] Power Transform完成 (处理偏态)")

# 2.3 特征选择 (去除低区分度特征)
selector = SelectKBest(f_classif, k=15)  # 从20个特征中选15个最好的
X_selected = selector.fit_transform(X_power, y)
selected_features = selector.get_support(indices=True)
print(f"   [OK] 特征选择完成: 保留 {len(selected_features)}/20 个特征")
print(f"       保留的特征索引: {selected_features.tolist()}")

# 2.4 PCA降维 (进一步去除冗余)
pca = PCA(n_components=0.95, random_state=42)  # 保留95%方差
X_pca = pca.fit_transform(X_selected)
print(f"   [OK] PCA降维完成: {X_selected.shape[1]} -> {X_pca.shape[1]} 维")
print(f"       累计方差解释: {pca.explained_variance_ratio_.sum()*100:.2f}%")

# 2.5 合并原始特征和PCA特征
X_enhanced = np.hstack([X_selected, X_pca])
print(f"   [OK] 特征增强完成: 最终特征数 = {X_enhanced.shape[1]}")

# ==================== 3. 划分数据集 ====================
print("\n3. 划分数据集...")
X_train, X_test, y_train, y_test = train_test_split(
    X_enhanced, y, test_size=0.15, random_state=42, stratify=y
)
X_train, X_val, y_train, y_val = train_test_split(
    X_train, y_train, test_size=0.15/0.85, random_state=42, stratify=y_train
)
print(f"   训练集: {X_train.shape[0]}, 验证集: {X_val.shape[0]}, 测试集: {X_test.shape[0]}")

# ==================== 4. ADASYN过采样 ====================
print("\n4. ADASYN过采样 (比SMOTE更智能)...")
class_counts = np.bincount(y_train)
print("   过采样前类别分布:")
for i, count in enumerate(class_counts):
    print(f"     类别 {i+1}: {count:5d} 样本")

# 只对样本数<3000的类别过采样
sampling_strategy = {}
for cls, count in enumerate(class_counts):
    if count < 3000:
        sampling_strategy[cls] = 3000

if sampling_strategy:
    try:
        adasyn = ADASYN(sampling_strategy=sampling_strategy, random_state=42, n_neighbors=5)
        X_train_resampled, y_train_resampled = adasyn.fit_resample(X_train, y_train)
        print(f"   [OK] ADASYN完成: {X_train.shape[0]} -> {X_train_resampled.shape[0]} 样本")
    except Exception as e:
        print(f"   [WARNING] ADASYN失败，使用原始数据: {e}")
        X_train_resampled, y_train_resampled = X_train, y_train
else:
    X_train_resampled, y_train_resampled = X_train, y_train
    print("   [SKIP] 无需过采样")

# ==================== 5. LightGBM训练 ====================
print("\n5. 训练LightGBM...")
print("="*60)

train_data = lgb.Dataset(X_train_resampled, label=y_train_resampled)
val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

# 优化后的参数
params = {
    'objective': 'multiclass',
    'num_class': 13,
    'metric': 'multi_logloss',
    'boosting_type': 'gbdt',
    'num_leaves': 40,
    'learning_rate': 0.05,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'max_depth': 8,
    'min_child_samples': 25,
    'lambda_l1': 0.1,
    'lambda_l2': 0.1,
    'min_gain_to_split': 0.01,
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

y_train_pred = model.predict(X_train_resampled).argmax(axis=1)
y_val_pred = model.predict(X_val).argmax(axis=1)
y_test_pred = model.predict(X_test).argmax(axis=1)

train_acc = accuracy_score(y_train_resampled, y_train_pred) * 100
val_acc = accuracy_score(y_val, y_val_pred) * 100
test_acc = accuracy_score(y_test, y_test_pred) * 100

print(f"训练集准确率: {train_acc:.2f}%")
print(f"验证集准确率: {val_acc:.2f}%")
print(f"测试集准确率: {test_acc:.2f}%")
print(f"过拟合程度: {train_acc - test_acc:.2f}%")

print("\n测试集分类报告:")
print(classification_report(y_test + 1, y_test_pred + 1))

# ==================== 7. 对比基线 ====================
print("\n" + "="*60)
print("与基线对比")
print("="*60)

baseline_acc = 65.0  # 之前LightGBM的结果
improvement = test_acc - baseline_acc

print(f"基线准确率 (原始特征):  {baseline_acc:.2f}%")
print(f"优化后准确率 (本方法):  {test_acc:.2f}%")
print(f"提升幅度:              {improvement:+.2f}%")

if improvement > 2:
    print("\n[SUCCESS] 显著提升！数据优化有效")
elif improvement > 0:
    print("\n[OK] 有小幅提升")
else:
    print("\n[WARNING] 未见提升，可能需要更多特征或重新定义问题")

# ==================== 8. 保存模型 ====================
model_path = f'models/optimized_lgb_{timestamp}.txt'
model.save_model(model_path)
print(f"\n模型已保存: {model_path}")

# ==================== 9. 保存记录 ====================
record = {
    '时间戳': timestamp,
    '训练时间': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    '模型类型': 'LightGBM-Optimized',
    '数据优化': 'StandardScaler+PowerTransform+FeatureSelection+PCA+ADASYN',
    '原始特征数': 20,
    '选择特征数': len(selected_features),
    'PCA维度': X_pca.shape[1],
    '最终特征数': X_enhanced.shape[1],
    '训练集大小': len(X_train_resampled),
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

csv_path = '优化模型记录.csv'
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
