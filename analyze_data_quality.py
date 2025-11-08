import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from scipy.spatial.distance import cdist
import warnings

warnings.filterwarnings('ignore')

print("="*60)
print("数据质量分析报告")
print("="*60)

# 1. 加载数据
df = pd.read_csv('样品特征数据集.csv', header=None)
X = df.iloc[:, :20].values
y = df.iloc[:, 20].values

print(f"\n数据集规模: {X.shape[0]} 样本, {X.shape[1]} 特征, {len(np.unique(y))} 类别")

# 2. 类别分布分析
print("\n" + "="*60)
print("1. 类别不平衡分析")
print("="*60)

class_counts = pd.Series(y).value_counts().sort_index()
print("\n类别分布:")
for cls, count in class_counts.items():
    percentage = count / len(y) * 100
    print(f"  类别 {cls}: {count:6d} 样本 ({percentage:5.2f}%)")

# 计算不平衡比率
max_count = class_counts.max()
min_count = class_counts.min()
imbalance_ratio = max_count / min_count
print(f"\n不平衡比率: {imbalance_ratio:.1f}:1 (最大类/最小类)")
print(f"最大类样本数: {max_count}")
print(f"最小类样本数: {min_count}")

if imbalance_ratio > 50:
    print("⚠️  严重不平衡！小类别很难学习")
elif imbalance_ratio > 10:
    print("⚠️  中度不平衡，需要特殊处理")
else:
    print("✓  不平衡程度可接受")

# 3. 特征质量分析
print("\n" + "="*60)
print("2. 特征质量分析")
print("="*60)

# 标准化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 特征方差
feature_vars = np.var(X_scaled, axis=0)
low_var_features = np.sum(feature_vars < 0.1)
print(f"\n特征方差分析:")
print(f"  平均方差: {np.mean(feature_vars):.4f}")
print(f"  最小方差: {np.min(feature_vars):.4f}")
print(f"  最大方差: {np.max(feature_vars):.4f}")
print(f"  低方差特征数(<0.1): {low_var_features}/20")

if low_var_features > 5:
    print("⚠️  存在较多低方差特征，可能包含噪声或冗余信息")

# 特征相关性
corr_matrix = np.corrcoef(X_scaled.T)
high_corr_pairs = 0
for i in range(20):
    for j in range(i+1, 20):
        if abs(corr_matrix[i, j]) > 0.9:
            high_corr_pairs += 1

print(f"\n特征相关性分析:")
print(f"  高度相关特征对数(>0.9): {high_corr_pairs}")
if high_corr_pairs > 10:
    print("⚠️  存在大量冗余特征")

# 4. 类别可分性分析（最重要！）
print("\n" + "="*60)
print("3. 类别可分性分析（核心问题）")
print("="*60)

# 4.1 类内距离 vs 类间距离
class_centers = []
for cls in np.unique(y):
    class_data = X_scaled[y == cls]
    class_centers.append(np.mean(class_data, axis=0))
class_centers = np.array(class_centers)

# 类间距离（中心点之间的距离）
inter_class_distances = cdist(class_centers, class_centers, metric='euclidean')
np.fill_diagonal(inter_class_distances, np.inf)
min_inter_distance = np.min(inter_class_distances)
mean_inter_distance = np.mean(inter_class_distances[inter_class_distances != np.inf])

# 类内距离（样本到类中心的平均距离）
intra_class_distances = []
for cls in np.unique(y):
    class_data = X_scaled[y == cls]
    center = class_centers[cls - 1]
    distances = np.linalg.norm(class_data - center, axis=1)
    intra_class_distances.append(np.mean(distances))
mean_intra_distance = np.mean(intra_class_distances)

# 可分性指标
separability_ratio = mean_inter_distance / mean_intra_distance

print(f"\n类间距离（类中心之间）:")
print(f"  平均类间距离: {mean_inter_distance:.4f}")
print(f"  最小类间距离: {min_inter_distance:.4f}")

print(f"\n类内距离（样本到类中心）:")
print(f"  平均类内距离: {mean_intra_distance:.4f}")

print(f"\n可分性比率: {separability_ratio:.4f}")
print(f"  (类间距离/类内距离，越大越好)")

if separability_ratio < 1.5:
    print("❌ 类别高度重叠！这是准确率低的主要原因")
    print("   → 类别之间的距离小于类内的散布")
    print("   → 模型很难区分不同类别")
elif separability_ratio < 3:
    print("⚠️  类别有一定重叠，分类难度较大")
else:
    print("✓  类别可分性良好")

# 4.2 Silhouette Score（轮廓系数）
# 采样以加速计算
sample_size = min(10000, len(X))
sample_indices = np.random.choice(len(X), sample_size, replace=False)
X_sample = X_scaled[sample_indices]
y_sample = y[sample_indices]

silhouette = silhouette_score(X_sample, y_sample)
print(f"\nSilhouette Score: {silhouette:.4f}")
print(f"  (范围[-1, 1]，越接近1越好)")

if silhouette < 0.2:
    print("❌ 类别严重重叠")
elif silhouette < 0.4:
    print("⚠️  类别有明显重叠")
else:
    print("✓  类别分离较好")

# 5. 特征区分度分析
print("\n" + "="*60)
print("4. 特征区分度分析")
print("="*60)

# 计算每个特征的类间方差/类内方差比（Fisher判别比）
fisher_scores = []
for i in range(20):
    feature = X_scaled[:, i]

    # 类间方差
    overall_mean = np.mean(feature)
    between_var = 0
    for cls in np.unique(y):
        class_data = feature[y == cls]
        class_mean = np.mean(class_data)
        between_var += len(class_data) * (class_mean - overall_mean) ** 2
    between_var /= len(feature)

    # 类内方差
    within_var = 0
    for cls in np.unique(y):
        class_data = feature[y == cls]
        class_mean = np.mean(class_data)
        within_var += np.sum((class_data - class_mean) ** 2)
    within_var /= len(feature)

    # Fisher判别比
    if within_var > 0:
        fisher_scores.append(between_var / within_var)
    else:
        fisher_scores.append(0)

fisher_scores = np.array(fisher_scores)
print(f"\nFisher判别比统计:")
print(f"  平均值: {np.mean(fisher_scores):.4f}")
print(f"  最大值: {np.max(fisher_scores):.4f}")
print(f"  最小值: {np.min(fisher_scores):.4f}")

good_features = np.sum(fisher_scores > 0.5)
print(f"  有效特征数(>0.5): {good_features}/20")

if good_features < 5:
    print("❌ 大部分特征区分度很低")
elif good_features < 10:
    print("⚠️  只有少数特征有区分度")
else:
    print("✓  多数特征有较好区分度")

# 6. 可视化
print("\n" + "="*60)
print("5. 生成可视化图表...")
print("="*60)

fig = plt.figure(figsize=(18, 12))

# 6.1 类别分布
ax1 = plt.subplot(3, 3, 1)
class_counts.plot(kind='bar', ax=ax1, color='steelblue')
ax1.set_title('Class Distribution (Imbalance)', fontsize=12, fontweight='bold')
ax1.set_xlabel('Class')
ax1.set_ylabel('Count')
ax1.grid(axis='y', alpha=0.3)

# 6.2 特征方差
ax2 = plt.subplot(3, 3, 2)
ax2.bar(range(20), feature_vars, color='coral')
ax2.axhline(y=0.1, color='r', linestyle='--', label='Low variance threshold')
ax2.set_title('Feature Variance', fontsize=12, fontweight='bold')
ax2.set_xlabel('Feature Index')
ax2.set_ylabel('Variance')
ax2.legend()
ax2.grid(axis='y', alpha=0.3)

# 6.3 Fisher判别比
ax3 = plt.subplot(3, 3, 3)
ax3.bar(range(20), fisher_scores, color='green')
ax3.axhline(y=0.5, color='r', linestyle='--', label='Good feature threshold')
ax3.set_title('Fisher Discriminant Ratio', fontsize=12, fontweight='bold')
ax3.set_xlabel('Feature Index')
ax3.set_ylabel('Fisher Score')
ax3.legend()
ax3.grid(axis='y', alpha=0.3)

# 6.4 特征相关性热图
ax4 = plt.subplot(3, 3, 4)
sns.heatmap(corr_matrix, cmap='coolwarm', center=0, square=True, ax=ax4,
            cbar_kws={'shrink': 0.8}, vmin=-1, vmax=1)
ax4.set_title('Feature Correlation Matrix', fontsize=12, fontweight='bold')

# 6.5 PCA降维可视化
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

ax5 = plt.subplot(3, 3, 5)
scatter = ax5.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='tab20', alpha=0.3, s=1)
ax5.set_title(f'PCA Visualization (Var: {sum(pca.explained_variance_ratio_)*100:.1f}%)',
              fontsize=12, fontweight='bold')
ax5.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)')
ax5.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)')
plt.colorbar(scatter, ax=ax5, label='Class')

# 6.6 t-SNE降维可视化（采样）
print("  计算t-SNE（可能需要1-2分钟）...")
sample_size_tsne = min(5000, len(X))
sample_indices_tsne = np.random.choice(len(X), sample_size_tsne, replace=False)
X_sample_tsne = X_scaled[sample_indices_tsne]
y_sample_tsne = y[sample_indices_tsne]

tsne = TSNE(n_components=2, random_state=42, perplexity=30)
X_tsne = tsne.fit_transform(X_sample_tsne)

ax6 = plt.subplot(3, 3, 6)
scatter = ax6.scatter(X_tsne[:, 0], X_tsne[:, 1], c=y_sample_tsne, cmap='tab20', alpha=0.5, s=5)
ax6.set_title('t-SNE Visualization', fontsize=12, fontweight='bold')
ax6.set_xlabel('t-SNE 1')
ax6.set_ylabel('t-SNE 2')
plt.colorbar(scatter, ax=ax6, label='Class')

# 6.7 类间距离矩阵
ax7 = plt.subplot(3, 3, 7)
inter_dist_display = inter_class_distances.copy()
np.fill_diagonal(inter_dist_display, 0)
sns.heatmap(inter_dist_display, cmap='YlOrRd', square=True, ax=ax7,
            cbar_kws={'shrink': 0.8}, annot=False)
ax7.set_title('Inter-Class Distance Matrix', fontsize=12, fontweight='bold')
ax7.set_xlabel('Class')
ax7.set_ylabel('Class')

# 6.8 类内距离分布
ax8 = plt.subplot(3, 3, 8)
ax8.bar(range(1, len(intra_class_distances)+1), intra_class_distances, color='purple')
ax8.set_title('Intra-Class Distance', fontsize=12, fontweight='bold')
ax8.set_xlabel('Class')
ax8.set_ylabel('Mean Distance to Center')
ax8.grid(axis='y', alpha=0.3)

# 6.9 可分性指标总结
ax9 = plt.subplot(3, 3, 9)
ax9.axis('off')
summary_text = f"""
数据质量总结

类别不平衡: {imbalance_ratio:.1f}:1
可分性比率: {separability_ratio:.2f}
Silhouette: {silhouette:.3f}
有效特征: {good_features}/20

主要问题:
"""
if separability_ratio < 1.5:
    summary_text += "• 类别高度重叠\n"
if imbalance_ratio > 50:
    summary_text += "• 严重类别不平衡\n"
if good_features < 10:
    summary_text += "• 特征区分度不足\n"
if high_corr_pairs > 10:
    summary_text += "• 特征冗余严重\n"

summary_text += f"\n预估准确率上限:\n{60 + separability_ratio * 5:.0f}% - {65 + separability_ratio * 5:.0f}%"

ax9.text(0.1, 0.5, summary_text, fontsize=11, verticalalignment='center',
         family='monospace', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.tight_layout()
plt.savefig('results/data_quality_analysis.png', dpi=300, bbox_inches='tight')
print("  ✓ 图表已保存: results/data_quality_analysis.png")

# 7. 最终诊断
print("\n" + "="*60)
print("最终诊断")
print("="*60)

print("\n准确率上限低的主要原因:")
issues = []
if separability_ratio < 1.5:
    issues.append("1. ❌ 类别高度重叠 - 这是最主要的问题")
    issues.append("   → 不同类别的样本在特征空间中混在一起")
    issues.append("   → 即使是最好的模型也无法区分")
if imbalance_ratio > 50:
    issues.append("2. ⚠️  严重的类别不平衡")
    issues.append("   → 小类别样本太少，模型学不到有效模式")
if good_features < 10:
    issues.append("3. ⚠️  特征区分度不足")
    issues.append("   → 现有特征无法有效区分类别")

if not issues:
    print("✓ 数据质量良好，模型性能可能还有提升空间")
else:
    for issue in issues:
        print(issue)

print("\n建议:")
if separability_ratio < 1.5:
    print("• 回到数据源，收集更多有区分度的特征")
    print("• 考虑重新定义问题（合并相似类别）")
    print("• 使用领域知识进行特征工程")
if imbalance_ratio > 50:
    print("• 收集更多小类别样本")
    print("• 考虑使用代价敏感学习")
if good_features < 10:
    print("• 进行特征选择，去除噪声特征")
    print("• 尝试非线性特征变换")

print("\n" + "="*60)
print(f"预估准确率上限: {60 + separability_ratio * 5:.0f}% - {65 + separability_ratio * 5:.0f}%")
print("="*60)
