import pandas as pd
import numpy as np
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import matplotlib.pyplot as plt
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

# ==================== 模型1: LightGBM ====================
print("="*60)
print("训练 LightGBM...")
print("="*60)

train_data_lgb = lgb.Dataset(X_train, label=y_train)
val_data_lgb = lgb.Dataset(X_val, label=y_val, reference=train_data_lgb)

params_lgb = {
    'objective': 'multiclass',
    'num_class': 13,
    'metric': 'multi_logloss',
    'boosting_type': 'gbdt',
    'num_leaves': 50,
    'learning_rate': 0.05,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'max_depth': 10,
    'min_child_samples': 30,
    'lambda_l1': 0.05,
    'lambda_l2': 0.05,
    'is_unbalance': True,
    'feature_pre_filter': False,  # 允许动态改变min_child_samples
    'verbose': -1
}

model_lgb = lgb.train(
    params_lgb,
    train_data_lgb,
    num_boost_round=1000,
    valid_sets=[val_data_lgb],
    valid_names=['valid'],
    callbacks=[
        lgb.early_stopping(stopping_rounds=50),
        lgb.log_evaluation(period=100)
    ]
)

# LightGBM预测（概率）
y_pred_lgb_proba = model_lgb.predict(X_test)
y_pred_lgb = y_pred_lgb_proba.argmax(axis=1)
acc_lgb = accuracy_score(y_test, y_pred_lgb) * 100
print(f"LightGBM 测试准确率: {acc_lgb:.2f}%\n")

# ==================== 模型2: XGBoost ====================
print("="*60)
print("训练 XGBoost...")
print("="*60)

# 计算类别权重
class_counts = np.bincount(y_train)
total_samples = len(y_train)
class_weights = total_samples / (len(class_counts) * class_counts)

# 为每个样本分配权重
sample_weights_xgb = np.array([class_weights[y] for y in y_train])

dtrain = xgb.DMatrix(X_train, label=y_train, weight=sample_weights_xgb)
dval = xgb.DMatrix(X_val, label=y_val)
dtest = xgb.DMatrix(X_test, label=y_test)

params_xgb = {
    'objective': 'multi:softprob',
    'num_class': 13,
    'eval_metric': 'mlogloss',
    'max_depth': 10,
    'learning_rate': 0.05,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'min_child_weight': 30,
    'reg_alpha': 0.05,
    'reg_lambda': 0.05,
    'tree_method': 'hist',
    'seed': 42
}

evals = [(dval, 'valid')]
model_xgb = xgb.train(
    params_xgb,
    dtrain,
    num_boost_round=1000,
    evals=evals,
    early_stopping_rounds=50,
    verbose_eval=100
)

# XGBoost预测（概率）
y_pred_xgb_proba = model_xgb.predict(dtest)
y_pred_xgb = y_pred_xgb_proba.argmax(axis=1)
acc_xgb = accuracy_score(y_test, y_pred_xgb) * 100
print(f"XGBoost 测试准确率: {acc_xgb:.2f}%\n")

# ==================== 模型3: CatBoost ====================
print("="*60)
print("训练 CatBoost...")
print("="*60)

model_cat = CatBoostClassifier(
    iterations=1000,
    learning_rate=0.05,
    depth=10,
    l2_leaf_reg=0.05,
    bootstrap_type='Bernoulli',  # 使用Bernoulli才能支持subsample
    subsample=0.8,
    colsample_bylevel=0.8,
    min_data_in_leaf=30,
    auto_class_weights='Balanced',
    random_seed=42,
    verbose=100,
    early_stopping_rounds=50
)

model_cat.fit(
    X_train, y_train,
    eval_set=(X_val, y_val),
    use_best_model=True
)

# CatBoost预测（概率）
y_pred_cat_proba = model_cat.predict_proba(X_test)
y_pred_cat = y_pred_cat_proba.argmax(axis=1)
acc_cat = accuracy_score(y_test, y_pred_cat) * 100
print(f"CatBoost 测试准确率: {acc_cat:.2f}%\n")

# ==================== 集成预测 ====================
print("="*60)
print("集成模型预测...")
print("="*60)

# 方法1: 简单平均
y_pred_avg_proba = (y_pred_lgb_proba + y_pred_xgb_proba + y_pred_cat_proba) / 3
y_pred_avg = y_pred_avg_proba.argmax(axis=1)
acc_avg = accuracy_score(y_test, y_pred_avg) * 100

# 方法2: 加权平均（根据验证集准确率）
val_pred_lgb = model_lgb.predict(X_val).argmax(axis=1)
val_pred_xgb = model_xgb.predict(xgb.DMatrix(X_val)).argmax(axis=1)
val_pred_cat = model_cat.predict(X_val)

weight_lgb = accuracy_score(y_val, val_pred_lgb)
weight_xgb = accuracy_score(y_val, val_pred_xgb)
weight_cat = accuracy_score(y_val, val_pred_cat)

total_weight = weight_lgb + weight_xgb + weight_cat
weight_lgb /= total_weight
weight_xgb /= total_weight
weight_cat /= total_weight

print(f"模型权重: LightGBM={weight_lgb:.3f}, XGBoost={weight_xgb:.3f}, CatBoost={weight_cat:.3f}")

y_pred_weighted_proba = (
    weight_lgb * y_pred_lgb_proba +
    weight_xgb * y_pred_xgb_proba +
    weight_cat * y_pred_cat_proba
)
y_pred_weighted = y_pred_weighted_proba.argmax(axis=1)
acc_weighted = accuracy_score(y_test, y_pred_weighted) * 100

# 方法3: 投票
y_pred_vote = []
for i in range(len(y_test)):
    votes = [y_pred_lgb[i], y_pred_xgb[i], y_pred_cat[i]]
    y_pred_vote.append(max(set(votes), key=votes.count))
y_pred_vote = np.array(y_pred_vote)
acc_vote = accuracy_score(y_test, y_pred_vote) * 100

# ==================== 结果汇总 ====================
print("\n" + "="*60)
print("最终结果汇总")
print("="*60)
print(f"LightGBM 单模型:     {acc_lgb:.2f}%")
print(f"XGBoost 单模型:      {acc_xgb:.2f}%")
print(f"CatBoost 单模型:     {acc_cat:.2f}%")
print(f"简单平均集成:        {acc_avg:.2f}%")
print(f"加权平均集成:        {acc_weighted:.2f}%")
print(f"投票集成:            {acc_vote:.2f}%")
print("="*60)

# 选择最佳集成方法
best_method = max([
    ('简单平均', acc_avg, y_pred_avg),
    ('加权平均', acc_weighted, y_pred_weighted),
    ('投票', acc_vote, y_pred_vote)
], key=lambda x: x[1])

print(f"\n最佳集成方法: {best_method[0]} - {best_method[1]:.2f}%\n")

# ==================== 分类报告 ====================
print("最佳集成方法的分类报告:")
print(classification_report(y_test + 1, best_method[2] + 1))

# ==================== 可视化 ====================
plt.figure(figsize=(12, 5))

# 子图1: 各模型准确率对比
plt.subplot(1, 2, 1)
models = ['LightGBM', 'XGBoost', 'CatBoost', '简单平均', '加权平均', '投票']
accuracies = [acc_lgb, acc_xgb, acc_cat, acc_avg, acc_weighted, acc_vote]
colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12', '#9b59b6', '#1abc9c']
bars = plt.bar(models, accuracies, color=colors)
plt.ylabel('准确率 (%)')
plt.title('模型性能对比')
plt.ylim([60, max(accuracies) + 2])
plt.xticks(rotation=45, ha='right')
for bar, acc in zip(bars, accuracies):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
             f'{acc:.2f}%', ha='center', va='bottom', fontsize=9)

# 子图2: 提升幅度
plt.subplot(1, 2, 2)
baseline = acc_lgb  # 以LightGBM为基线
improvements = [0, acc_xgb - baseline, acc_cat - baseline,
                acc_avg - baseline, acc_weighted - baseline, acc_vote - baseline]
colors_imp = ['gray' if x <= 0 else 'green' for x in improvements]
bars = plt.bar(models, improvements, color=colors_imp)
plt.ylabel('相对LightGBM的提升 (%)')
plt.title('集成提升效果')
plt.axhline(y=0, color='black', linestyle='--', linewidth=0.8)
plt.xticks(rotation=45, ha='right')
for bar, imp in zip(bars, improvements):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1 if imp > 0 else bar.get_height() - 0.3,
             f'{imp:+.2f}%', ha='center', va='bottom' if imp > 0 else 'top', fontsize=9)

plt.tight_layout()
result_path = f'results/ensemble_comparison_{timestamp}.png'
plt.savefig(result_path, dpi=300)
print(f"结果图表: {result_path}")

# ==================== 保存记录 ====================
record = {
    '时间戳': timestamp,
    '训练时间': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    '模型类型': 'Ensemble (LightGBM+XGBoost+CatBoost)',
    'LightGBM准确率': f"{acc_lgb:.2f}%",
    'XGBoost准确率': f"{acc_xgb:.2f}%",
    'CatBoost准确率': f"{acc_cat:.2f}%",
    '简单平均准确率': f"{acc_avg:.2f}%",
    '加权平均准确率': f"{acc_weighted:.2f}%",
    '投票准确率': f"{acc_vote:.2f}%",
    '最佳方法': best_method[0],
    '最佳准确率': f"{best_method[1]:.2f}%",
    '相比LightGBM提升': f"{best_method[1] - acc_lgb:+.2f}%",
    '训练集大小': len(X_train),
    '验证集大小': len(X_val),
    '测试集大小': len(X_test),
    '结果图表': result_path
}

csv_path = '集成模型记录.csv'
record_df = pd.DataFrame([record])

if os.path.exists(csv_path):
    existing_df = pd.read_csv(csv_path, encoding='utf-8-sig')
    updated_df = pd.concat([existing_df, record_df], ignore_index=True)
    updated_df.to_csv(csv_path, index=False, encoding='utf-8-sig')
else:
    record_df.to_csv(csv_path, index=False, encoding='utf-8-sig')

print(f"训练记录: {csv_path}")
print("="*60)
