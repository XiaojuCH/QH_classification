import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score
from sklearn.utils.class_weight import compute_class_weight
import matplotlib.pyplot as plt
from datetime import datetime
import os
import joblib
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
y = df.iloc[:, 20].values - 1  # 转换为0-12
print(f"数据集: {X.shape}, 类别数: {len(np.unique(y))}")

# ==================== 2. 数据预处理 ====================
# 标准化（SVM需要）
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
class_weight_dict = {i: w for i, w in enumerate(class_weights)}

# ==================== 3. 随机森林 ====================
print("="*60)
print("训练 Random Forest...")
print("="*60)

rf_model = RandomForestClassifier(
    n_estimators=500,           # 树的数量
    max_depth=20,               # 最大深度
    min_samples_split=10,       # 分裂所需最小样本数
    min_samples_leaf=5,         # 叶子节点最小样本数
    max_features='sqrt',        # 每次分裂考虑的特征数
    class_weight='balanced',    # 自动处理类别不平衡
    n_jobs=-1,                  # 使用所有CPU核心
    random_state=42,
    verbose=1
)

print("\n开始训练...")
rf_model.fit(X_train, y_train)

# 预测
y_train_pred_rf = rf_model.predict(X_train)
y_val_pred_rf = rf_model.predict(X_val)
y_test_pred_rf = rf_model.predict(X_test)

train_acc_rf = accuracy_score(y_train, y_train_pred_rf) * 100
val_acc_rf = accuracy_score(y_val, y_val_pred_rf) * 100
test_acc_rf = accuracy_score(y_test, y_test_pred_rf) * 100

print(f"\nRandom Forest 结果:")
print(f"  训练集准确率: {train_acc_rf:.2f}%")
print(f"  验证集准确率: {val_acc_rf:.2f}%")
print(f"  测试集准确率: {test_acc_rf:.2f}%")

# 保存模型
rf_model_path = f'models/random_forest_{timestamp}.pkl'
joblib.dump(rf_model, rf_model_path)
print(f"  模型已保存: {rf_model_path}")

# ==================== 4. SVM ====================
print("\n" + "="*60)
print("训练 SVM (可能需要较长时间)...")
print("="*60)

# 由于SVM训练很慢，对于大数据集使用线性核
# 如果数据量小于5万，可以尝试RBF核
if len(X_train) > 50000:
    print("数据量较大，使用线性核SVM...")
    svm_model = SVC(
        kernel='linear',
        C=1.0,
        class_weight='balanced',
        random_state=42,
        verbose=False,  # 关闭刷屏
        max_iter=5000   # 增加迭代次数
    )
else:
    print("使用RBF核SVM...")
    svm_model = SVC(
        kernel='rbf',
        C=10.0,
        gamma='scale',
        class_weight='balanced',
        random_state=42,
        verbose=False,  # 关闭刷屏
        max_iter=5000   # 增加迭代次数
    )

print("\n开始训练...")
svm_model.fit(X_train, y_train)

# 预测
y_train_pred_svm = svm_model.predict(X_train)
y_val_pred_svm = svm_model.predict(X_val)
y_test_pred_svm = svm_model.predict(X_test)

train_acc_svm = accuracy_score(y_train, y_train_pred_svm) * 100
val_acc_svm = accuracy_score(y_val, y_val_pred_svm) * 100
test_acc_svm = accuracy_score(y_test, y_test_pred_svm) * 100

print(f"\nSVM 结果:")
print(f"  训练集准确率: {train_acc_svm:.2f}%")
print(f"  验证集准确率: {val_acc_svm:.2f}%")
print(f"  测试集准确率: {test_acc_svm:.2f}%")

# 保存模型
svm_model_path = f'models/svm_{timestamp}.pkl'
joblib.dump(svm_model, svm_model_path)
print(f"  模型已保存: {svm_model_path}")

# ==================== 5. 结果对比 ====================
print("\n" + "="*60)
print("最终结果对比")
print("="*60)

print(f"\nRandom Forest:")
print(f"  训练集: {train_acc_rf:.2f}%")
print(f"  验证集: {val_acc_rf:.2f}%")
print(f"  测试集: {test_acc_rf:.2f}%")

print(f"\nSVM:")
print(f"  训练集: {train_acc_svm:.2f}%")
print(f"  验证集: {val_acc_svm:.2f}%")
print(f"  测试集: {test_acc_svm:.2f}%")

# 选择最佳模型
if test_acc_rf > test_acc_svm:
    best_model_name = "Random Forest"
    best_acc = test_acc_rf
    best_pred = y_test_pred_rf
else:
    best_model_name = "SVM"
    best_acc = test_acc_svm
    best_pred = y_test_pred_svm

print(f"\n最佳模型: {best_model_name} - {best_acc:.2f}%")

# ==================== 6. 详细分类报告 ====================
print("\n" + "="*60)
print(f"{best_model_name} 分类报告:")
print("="*60)
print(classification_report(y_test + 1, best_pred + 1))

# ==================== 7. 特征重要性（Random Forest）====================
print("\n" + "="*60)
print("Random Forest 特征重要性:")
print("="*60)

feature_importance = rf_model.feature_importances_
feature_names = [f'Feature_{i}' for i in range(20)]

# 排序
indices = np.argsort(feature_importance)[::-1]
print("\nTop 10 重要特征:")
for i in range(min(10, len(indices))):
    idx = indices[i]
    print(f"  {i+1}. {feature_names[idx]}: {feature_importance[idx]:.4f}")

# ==================== 8. 可视化 ====================
fig = plt.figure(figsize=(15, 10))

# 8.1 准确率对比
ax1 = plt.subplot(2, 3, 1)
models = ['RF Train', 'RF Val', 'RF Test', 'SVM Train', 'SVM Val', 'SVM Test']
accuracies = [train_acc_rf, val_acc_rf, test_acc_rf, train_acc_svm, val_acc_svm, test_acc_svm]
colors = ['#3498db', '#5dade2', '#85c1e9', '#e74c3c', '#ec7063', '#f1948a']
bars = ax1.bar(models, accuracies, color=colors)
ax1.set_ylabel('Accuracy (%)')
ax1.set_title('Model Performance Comparison')
ax1.set_ylim([0, 100])
plt.xticks(rotation=45, ha='right')
for bar, acc in zip(bars, accuracies):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
             f'{acc:.1f}%', ha='center', va='bottom', fontsize=9)

# 8.2 测试集准确率对比
ax2 = plt.subplot(2, 3, 2)
test_models = ['Random Forest', 'SVM']
test_accs = [test_acc_rf, test_acc_svm]
colors_test = ['#3498db', '#e74c3c']
bars = ax2.bar(test_models, test_accs, color=colors_test)
ax2.set_ylabel('Accuracy (%)')
ax2.set_title('Test Set Accuracy')
ax2.set_ylim([0, 100])
for bar, acc in zip(bars, test_accs):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
             f'{acc:.2f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')

# 8.3 Random Forest 特征重要性
ax3 = plt.subplot(2, 3, 3)
ax3.barh(range(20), feature_importance[indices], color='green')
ax3.set_yticks(range(20))
ax3.set_yticklabels([feature_names[i] for i in indices], fontsize=8)
ax3.set_xlabel('Importance')
ax3.set_title('Random Forest Feature Importance')
ax3.invert_yaxis()

# 8.4 过拟合分析 - Random Forest
ax4 = plt.subplot(2, 3, 4)
rf_data = ['Train', 'Val', 'Test']
rf_accs = [train_acc_rf, val_acc_rf, test_acc_rf]
ax4.plot(rf_data, rf_accs, marker='o', linewidth=2, markersize=10, color='#3498db')
ax4.set_ylabel('Accuracy (%)')
ax4.set_title('Random Forest: Overfitting Analysis')
ax4.set_ylim([min(rf_accs) - 5, max(rf_accs) + 5])
ax4.grid(True, alpha=0.3)
for i, (x, y) in enumerate(zip(rf_data, rf_accs)):
    ax4.text(x, y + 1, f'{y:.1f}%', ha='center', fontsize=9)

# 8.5 过拟合分析 - SVM
ax5 = plt.subplot(2, 3, 5)
svm_data = ['Train', 'Val', 'Test']
svm_accs = [train_acc_svm, val_acc_svm, test_acc_svm]
ax5.plot(svm_data, svm_accs, marker='s', linewidth=2, markersize=10, color='#e74c3c')
ax5.set_ylabel('Accuracy (%)')
ax5.set_title('SVM: Overfitting Analysis')
ax5.set_ylim([min(svm_accs) - 5, max(svm_accs) + 5])
ax5.grid(True, alpha=0.3)
for i, (x, y) in enumerate(zip(svm_data, svm_accs)):
    ax5.text(x, y + 1, f'{y:.1f}%', ha='center', fontsize=9)

# 8.6 总结
ax6 = plt.subplot(2, 3, 6)
ax6.axis('off')
summary_text = f"""
模型性能总结

Random Forest:
  测试准确率: {test_acc_rf:.2f}%
  过拟合程度: {train_acc_rf - test_acc_rf:.2f}%

SVM:
  测试准确率: {test_acc_svm:.2f}%
  过拟合程度: {train_acc_svm - test_acc_svm:.2f}%

最佳模型: {best_model_name}
最佳准确率: {best_acc:.2f}%

训练时间: {timestamp}
"""
ax6.text(0.1, 0.5, summary_text, fontsize=11, verticalalignment='center',
         family='monospace', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))

plt.tight_layout()
result_path = f'results/rf_svm_comparison_{timestamp}.png'
plt.savefig(result_path, dpi=300)
print(f"\n结果图表: {result_path}")

# ==================== 9. 保存记录 ====================
record = {
    '时间戳': timestamp,
    '训练时间': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    'RF_训练准确率': f"{train_acc_rf:.2f}%",
    'RF_验证准确率': f"{val_acc_rf:.2f}%",
    'RF_测试准确率': f"{test_acc_rf:.2f}%",
    'SVM_训练准确率': f"{train_acc_svm:.2f}%",
    'SVM_验证准确率': f"{val_acc_svm:.2f}%",
    'SVM_测试准确率': f"{test_acc_svm:.2f}%",
    '最佳模型': best_model_name,
    '最佳准确率': f"{best_acc:.2f}%",
    'RF_模型路径': rf_model_path,
    'SVM_模型路径': svm_model_path,
    '结果图表': result_path
}

csv_path = 'RF_SVM记录.csv'
record_df = pd.DataFrame([record])

if os.path.exists(csv_path):
    existing_df = pd.read_csv(csv_path, encoding='utf-8-sig')
    updated_df = pd.concat([existing_df, record_df], ignore_index=True)
    updated_df.to_csv(csv_path, index=False, encoding='utf-8-sig')
else:
    record_df.to_csv(csv_path, index=False, encoding='utf-8-sig')

print(f"训练记录: {csv_path}")
print("="*60)
print(f"最佳模型: {best_model_name}")
print(f"最佳测试准确率: {best_acc:.2f}%")
print("="*60)
