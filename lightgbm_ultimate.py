import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.metrics import classification_report, accuracy_score
from imblearn.over_sampling import SMOTE
import optuna
from datetime import datetime
import os
import warnings

warnings.filterwarnings('ignore')
np.random.seed(42)

os.makedirs('models', exist_ok=True)
os.makedirs('results', exist_ok=True)

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
print(f"训练时间戳: {timestamp}\n")

# ==================== 1. 加载数据 ====================
print("正在加载数据...")
df = pd.read_csv('样品特征数据集.csv', header=None)
X = df.iloc[:, :20].values
y = df.iloc[:, 20].values - 1
print(f"原始数据集: {X.shape}, 类别数: {len(np.unique(y))}")

# ==================== 2. 特征工程 ====================
print("\n正在进行特征工程...")

# 2.1 标准化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 2.2 多项式特征（只选择重要的交互特征，避免维度爆炸）
# 选择前10个特征进行二次交互
poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
X_poly = poly.fit_transform(X_scaled[:, :10])  # 只对前10个特征做交互

# 2.3 统计特征
X_stats = np.column_stack([
    X_scaled.mean(axis=1),      # 均值
    X_scaled.std(axis=1),       # 标准差
    X_scaled.max(axis=1),       # 最大值
    X_scaled.min(axis=1),       # 最小值
    X_scaled.max(axis=1) - X_scaled.min(axis=1)  # 极差
])

# 2.4 合并所有特征
X_enhanced = np.hstack([X_scaled, X_poly, X_stats])
print(f"特征工程后: {X_enhanced.shape} (增加了 {X_enhanced.shape[1] - X.shape[1]} 个特征)")

# ==================== 3. 划分数据集 ====================
X_train, X_test, y_train, y_test = train_test_split(
    X_enhanced, y, test_size=0.15, random_state=42, stratify=y
)
X_train, X_val, y_train, y_val = train_test_split(
    X_train, y_train, test_size=0.15/0.85, random_state=42, stratify=y_train
)

print(f"训练集: {X_train.shape[0]}, 验证集: {X_val.shape[0]}, 测试集: {X_test.shape[0]}")

# ==================== 4. SMOTE过采样（只对小类别） ====================
print("\n正在使用SMOTE过采样...")
class_counts = np.bincount(y_train)
print("过采样前类别分布:")
for i, count in enumerate(class_counts):
    print(f"  类别 {i+1}: {count} 样本")

# 只对样本数<5000的类别过采样到5000
sampling_strategy = {}
for cls, count in enumerate(class_counts):
    if count < 5000:
        sampling_strategy[cls] = 5000

if sampling_strategy:
    smote = SMOTE(sampling_strategy=sampling_strategy, random_state=42, k_neighbors=5)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
    print(f"\n过采样后训练集: {X_train_resampled.shape[0]} (增加了 {X_train_resampled.shape[0] - X_train.shape[0]} 个样本)")
else:
    X_train_resampled, y_train_resampled = X_train, y_train

# ==================== 5. Optuna超参数优化 ====================
print("\n" + "="*60)
print("开始超参数优化 (Optuna)...")
print("="*60)

train_data = lgb.Dataset(X_train_resampled, label=y_train_resampled)
val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

def objective(trial):
    params = {
        'objective': 'multiclass',
        'num_class': 13,
        'metric': 'multi_logloss',
        'boosting_type': 'gbdt',
        'num_leaves': trial.suggest_int('num_leaves', 30, 100),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1),
        'feature_fraction': trial.suggest_float('feature_fraction', 0.6, 0.9),
        'bagging_fraction': trial.suggest_float('bagging_fraction', 0.6, 0.9),
        'bagging_freq': trial.suggest_int('bagging_freq', 1, 7),
        'max_depth': trial.suggest_int('max_depth', 6, 15),
        'min_child_samples': trial.suggest_int('min_child_samples', 10, 50),
        'lambda_l1': trial.suggest_float('lambda_l1', 0.0, 0.2),
        'lambda_l2': trial.suggest_float('lambda_l2', 0.0, 0.2),
        'min_gain_to_split': trial.suggest_float('min_gain_to_split', 0.0, 0.05),
        'is_unbalance': True,
        'feature_pre_filter': False,  # 允许动态改变min_child_samples
        'verbose': -1
    }

    model = lgb.train(
        params,
        train_data,
        num_boost_round=500,
        valid_sets=[val_data],
        callbacks=[lgb.early_stopping(stopping_rounds=30)]
    )

    y_pred = model.predict(X_val).argmax(axis=1)
    accuracy = accuracy_score(y_val, y_pred)
    return accuracy

# 运行优化（30次试验）
study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=42))
study.optimize(objective, n_trials=30, show_progress_bar=True)

print(f"\n最佳验证准确率: {study.best_value * 100:.2f}%")
print("最佳参数:")
for key, value in study.best_params.items():
    print(f"  {key}: {value}")

# ==================== 6. 使用最佳参数训练最终模型 ====================
print("\n" + "="*60)
print("使用最佳参数训练最终模型...")
print("="*60)

best_params = study.best_params.copy()
best_params.update({
    'objective': 'multiclass',
    'num_class': 13,
    'metric': 'multi_logloss',
    'boosting_type': 'gbdt',
    'is_unbalance': True,
    'verbose': -1
})

final_model = lgb.train(
    best_params,
    train_data,
    num_boost_round=1000,
    valid_sets=[val_data],
    callbacks=[
        lgb.early_stopping(stopping_rounds=50),
        lgb.log_evaluation(period=50)
    ]
)

# ==================== 7. 评估 ====================
print("\n" + "="*60)
print("最终评估")
print("="*60)

y_train_pred = final_model.predict(X_train_resampled).argmax(axis=1)
y_val_pred = final_model.predict(X_val).argmax(axis=1)
y_test_pred = final_model.predict(X_test).argmax(axis=1)

train_acc = accuracy_score(y_train_resampled, y_train_pred) * 100
val_acc = accuracy_score(y_val, y_val_pred) * 100
test_acc = accuracy_score(y_test, y_test_pred) * 100

print(f"训练集准确率: {train_acc:.2f}%")
print(f"验证集准确率: {val_acc:.2f}%")
print(f"测试集准确率: {test_acc:.2f}%")

print("\n测试集分类报告:")
print(classification_report(y_test + 1, y_test_pred + 1))

# ==================== 8. 保存模型 ====================
model_path = f'models/lightgbm_ultimate_{timestamp}.txt'
final_model.save_model(model_path)
print(f"\n模型已保存: {model_path}")

# ==================== 9. 保存记录 ====================
record = {
    '时间戳': timestamp,
    '训练时间': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    '模型类型': 'LightGBM-Ultimate (特征工程+SMOTE+Optuna)',
    '原始特征数': 20,
    '增强后特征数': X_enhanced.shape[1],
    '训练集大小': len(X_train_resampled),
    '验证集大小': len(X_val),
    '测试集大小': len(X_test),
    '训练准确率': f"{train_acc:.2f}%",
    '验证准确率': f"{val_acc:.2f}%",
    '测试准确率': f"{test_acc:.2f}%",
    'Optuna试验次数': 30,
    '最佳迭代轮数': final_model.best_iteration,
    '模型路径': model_path
}

csv_path = 'LightGBM终极版记录.csv'
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
print("="*60)

# 保存最佳参数
params_df = pd.DataFrame([best_params])
params_path = f'results/best_params_{timestamp}.csv'
params_df.to_csv(params_path, index=False, encoding='utf-8-sig')
print(f"最佳参数已保存: {params_path}")
