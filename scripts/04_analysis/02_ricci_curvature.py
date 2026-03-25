#!/usr/bin/env python3
# -- Project root setup --
import os as _os
from pathlib import Path as _Path
_os.chdir(_Path(__file__).resolve().parents[2])
"""
局部曲率估计 - Ollivier-Ricci曲率

输入文件:
- data/pca/pca_embeddings_50d.npy (11068×50 PCA嵌入)
- data/metadata_final_with_en.csv (元数据)

输出文件:
- data/curvature/ricci_curvature.npy (11068维曲率值)
- data/curvature/curvature_statistics.txt (统计报告)
- data/curvature/curvature_visualization.png (可视化)
- data/curvature/curvature_by_category.png (按类别分布)

功能描述:
使用Ollivier-Ricci曲率估计流形的局部几何性质。
正曲率表示局部收缩（类似球面），负曲率表示局部扩张（类似双曲面）。
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neighbors import NearestNeighbors
from scipy.spatial.distance import cdist
from scipy.stats import spearmanr
import os
from tqdm import tqdm
import time

# 设置随机种子
np.random.seed(42)

# 创建输出目录
os.makedirs('data/curvature', exist_ok=True)

print("=" * 80)
print("局部曲率估计 - Ollivier-Ricci曲率")
print("=" * 80)

# 1. 加载数据
print("\n[1/6] 加载数据...")
embeddings_50d = np.load('data/pca/pca_embeddings_50d.npy')
metadata = pd.read_csv('data/metadata_final_with_en.csv')

n_samples = embeddings_50d.shape[0]
print(f"  - 嵌入维度: {embeddings_50d.shape}")
print(f"  - 元数据: {metadata.shape}")

# 2. 构建k-NN图
print("\n[2/6] 构建k-NN图...")
k = 15  # 与UMAP参数一致
start_time = time.time()

nbrs = NearestNeighbors(n_neighbors=k+1, algorithm='auto', metric='euclidean', n_jobs=-1)
nbrs.fit(embeddings_50d)
distances, indices = nbrs.kneighbors(embeddings_50d)

# 移除自身（第一个邻居）
distances = distances[:, 1:]
indices = indices[:, 1:]

elapsed = time.time() - start_time
print(f"  - k-NN图构建完成: {elapsed:.2f}秒")
print(f"  - 平均邻居距离: {distances.mean():.4f} ± {distances.std():.4f}")

# 3. 计算Ollivier-Ricci曲率
print("\n[3/6] 计算Ollivier-Ricci曲率...")
print("  - 方法: 基于概率测度的离散曲率")
print("  - 公式: κ(x,y) = 1 - W(μ_x, μ_y) / d(x,y)")
print("  - W: Wasserstein距离, μ_x: 点x的邻域分布")

start_time = time.time()
ricci_curvature = np.zeros(n_samples)

# 对每个点计算平均曲率（与其k个邻居的边）
for i in tqdm(range(n_samples), desc="  计算曲率"):
    neighbors_i = indices[i]
    dist_i = distances[i]

    curvatures_i = []

    for j_idx, j in enumerate(neighbors_i):
        # 点i和点j之间的距离
        d_ij = dist_i[j_idx]

        if d_ij < 1e-10:  # 避免除零
            continue

        # 点i的邻域分布（均匀分布）
        neighbors_j = indices[j]

        # 计算Wasserstein距离的简化版本：
        # 使用邻域之间的平均距离作为近似
        # W(μ_i, μ_j) ≈ 平均距离(neighbors_i, neighbors_j)

        # 计算i的邻居到j的邻居的距离矩阵
        coords_i_neighbors = embeddings_50d[neighbors_i]
        coords_j_neighbors = embeddings_50d[neighbors_j]

        # 使用最小匹配距离作为Wasserstein距离的近似
        dist_matrix = cdist(coords_i_neighbors, coords_j_neighbors, metric='euclidean')
        wasserstein_approx = dist_matrix.min(axis=1).mean()

        # Ollivier-Ricci曲率
        kappa = 1.0 - wasserstein_approx / d_ij
        curvatures_i.append(kappa)

    # 平均曲率
    if len(curvatures_i) > 0:
        ricci_curvature[i] = np.mean(curvatures_i)

elapsed = time.time() - start_time
print(f"  - 曲率计算完成: {elapsed:.2f}秒 ({elapsed/n_samples*1000:.2f}毫秒/点)")

# 保存曲率值
np.save('data/curvature/ricci_curvature.npy', ricci_curvature)
print(f"  - 曲率值已保存")

# 4. 统计分析
print("\n[4/6] 统计分析...")

# 全局统计
stats = {
    'mean': ricci_curvature.mean(),
    'std': ricci_curvature.std(),
    'min': ricci_curvature.min(),
    'max': ricci_curvature.max(),
    'median': np.median(ricci_curvature),
    'q25': np.percentile(ricci_curvature, 25),
    'q75': np.percentile(ricci_curvature, 75)
}

print(f"  全局统计:")
print(f"    - 均值: {stats['mean']:.4f} ± {stats['std']:.4f}")
print(f"    - 范围: [{stats['min']:.4f}, {stats['max']:.4f}]")
print(f"    - 中位数: {stats['median']:.4f}")
print(f"    - 四分位数: [{stats['q25']:.4f}, {stats['q75']:.4f}]")

# 按类别统计
metadata['ricci_curvature'] = ricci_curvature
category_stats = metadata.groupby('category')['ricci_curvature'].agg(['count', 'mean', 'std', 'min', 'max'])
print(f"\n  按类别统计:")
print(category_stats)

# 与物理指标E[n]的相关性（仅anchor序列）
anchor_mask = metadata['category'] == 'anchor'
anchor_data = metadata[anchor_mask].copy()

# 移除E_n为NaN的行
valid_mask = ~anchor_data['E_n'].isna()
anchor_valid = anchor_data[valid_mask]

if len(anchor_valid) > 0:
    corr_en, pval_en = spearmanr(anchor_valid['ricci_curvature'], anchor_valid['E_n'])
    print(f"\n  相关性分析 (anchor序列, n={len(anchor_valid)}):")
    print(f"    - Ricci曲率 vs E[n]: ρ={corr_en:.4f}, p={pval_en:.4f}")

    if pval_en < 0.05:
        print(f"    ✓ 显著相关 (p < 0.05)")
    else:
        print(f"    ✗ 无显著相关 (p ≥ 0.05)")

# 保存统计报告
with open('data/curvature/curvature_statistics.txt', 'w') as f:
    f.write("=" * 80 + "\n")
    f.write("Ollivier-Ricci曲率统计报告\n")
    f.write("=" * 80 + "\n\n")

    f.write("全局统计:\n")
    f.write(f"  样本数: {n_samples}\n")
    f.write(f"  均值: {stats['mean']:.6f} ± {stats['std']:.6f}\n")
    f.write(f"  范围: [{stats['min']:.6f}, {stats['max']:.6f}]\n")
    f.write(f"  中位数: {stats['median']:.6f}\n")
    f.write(f"  四分位数: [{stats['q25']:.6f}, {stats['q75']:.6f}]\n\n")

    f.write("按类别统计:\n")
    f.write(category_stats.to_string() + "\n\n")

    if len(anchor_valid) > 0:
        f.write(f"相关性分析 (anchor序列, n={len(anchor_valid)}):\n")
        f.write(f"  Ricci曲率 vs E[n]: ρ={corr_en:.6f}, p={pval_en:.6f}\n")
        if pval_en < 0.05:
            f.write(f"  结论: 显著相关 (p < 0.05)\n")
        else:
            f.write(f"  结论: 无显著相关 (p ≥ 0.05)\n")

print(f"  - 统计报告已保存")

# 5. 可视化
print("\n[5/6] 可视化...")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 5.1 曲率分布直方图
ax = axes[0, 0]
ax.hist(ricci_curvature, bins=50, edgecolor='black', alpha=0.7)
ax.axvline(stats['mean'], color='red', linestyle='--', linewidth=2, label=f"均值={stats['mean']:.3f}")
ax.axvline(stats['median'], color='green', linestyle='--', linewidth=2, label=f"中位数={stats['median']:.3f}")
ax.set_xlabel('Ricci曲率', fontsize=12)
ax.set_ylabel('频数', fontsize=12)
ax.set_title('Ricci曲率分布', fontsize=14, fontweight='bold')
ax.legend()
ax.grid(alpha=0.3)

# 5.2 按类别的箱线图
ax = axes[0, 1]
categories = metadata['category'].unique()
data_by_category = [metadata[metadata['category'] == cat]['ricci_curvature'].values for cat in categories]
bp = ax.boxplot(data_by_category, labels=categories, patch_artist=True)
for patch in bp['boxes']:
    patch.set_facecolor('lightblue')
ax.set_xlabel('类别', fontsize=12)
ax.set_ylabel('Ricci曲率', fontsize=12)
ax.set_title('按类别的曲率分布', fontsize=14, fontweight='bold')
ax.grid(alpha=0.3, axis='y')
plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

# 5.3 曲率 vs E[n] 散点图（anchor序列）
ax = axes[1, 0]
if len(anchor_valid) > 0:
    ax.scatter(anchor_valid['E_n'], anchor_valid['ricci_curvature'],
               alpha=0.5, s=20, edgecolors='black', linewidths=0.5)
    ax.set_xlabel('E[n] (可积性误差)', fontsize=12)
    ax.set_ylabel('Ricci曲率', fontsize=12)
    ax.set_title(f'曲率 vs E[n] (anchor, n={len(anchor_valid)})\nρ={corr_en:.3f}, p={pval_en:.3f}',
                 fontsize=14, fontweight='bold')
    ax.grid(alpha=0.3)
else:
    ax.text(0.5, 0.5, '无有效anchor数据', ha='center', va='center', fontsize=14)
    ax.set_title('曲率 vs E[n]', fontsize=14, fontweight='bold')

# 5.4 曲率的累积分布函数
ax = axes[1, 1]
sorted_curvature = np.sort(ricci_curvature)
cumulative = np.arange(1, len(sorted_curvature) + 1) / len(sorted_curvature)
ax.plot(sorted_curvature, cumulative, linewidth=2)
ax.axvline(0, color='red', linestyle='--', linewidth=2, label='曲率=0')
ax.set_xlabel('Ricci曲率', fontsize=12)
ax.set_ylabel('累积概率', fontsize=12)
ax.set_title('曲率累积分布函数', fontsize=14, fontweight='bold')
ax.legend()
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('data/curvature/curvature_visualization.png', dpi=300, bbox_inches='tight')
print(f"  - 可视化已保存: curvature_visualization.png")

# 6. 按类别详细可视化
print("\n[6/6] 按类别详细可视化...")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# 6.1 小提琴图
ax = axes[0]
parts = ax.violinplot(data_by_category, positions=range(len(categories)),
                       showmeans=True, showmedians=True)
ax.set_xticks(range(len(categories)))
ax.set_xticklabels(categories, rotation=45, ha='right')
ax.set_xlabel('类别', fontsize=12)
ax.set_ylabel('Ricci曲率', fontsize=12)
ax.set_title('按类别的曲率分布（小提琴图）', fontsize=14, fontweight='bold')
ax.grid(alpha=0.3, axis='y')

# 6.2 核密度估计
ax = axes[1]
for cat in categories:
    cat_data = metadata[metadata['category'] == cat]['ricci_curvature'].values
    ax.hist(cat_data, bins=30, alpha=0.5, label=cat, density=True)
ax.set_xlabel('Ricci曲率', fontsize=12)
ax.set_ylabel('密度', fontsize=12)
ax.set_title('按类别的曲率密度分布', fontsize=14, fontweight='bold')
ax.legend()
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('data/curvature/curvature_by_category.png', dpi=300, bbox_inches='tight')
print(f"  - 可视化已保存: curvature_by_category.png")

# 完成
print("\n" + "=" * 80)
print("局部曲率估计完成!")
print("=" * 80)
print(f"\n输出文件:")
print(f"  - data/curvature/ricci_curvature.npy")
print(f"  - data/curvature/curvature_statistics.txt")
print(f"  - data/curvature/curvature_visualization.png")
print(f"  - data/curvature/curvature_by_category.png")
print(f"\n关键发现:")
print(f"  - 平均曲率: {stats['mean']:.4f} ± {stats['std']:.4f}")
if stats['mean'] > 0:
    print(f"  - 流形整体呈正曲率（局部收缩，类似球面）")
elif stats['mean'] < 0:
    print(f"  - 流形整体呈负曲率（局部扩张，类似双曲面）")
else:
    print(f"  - 流形整体接近平坦")
if len(anchor_valid) > 0 and pval_en < 0.05:
    print(f"  - ✓ 曲率与E[n]显著相关 (ρ={corr_en:.3f}, p={pval_en:.3f})")
elif len(anchor_valid) > 0:
    print(f"  - ✗ 曲率与E[n]无显著相关 (ρ={corr_en:.3f}, p={pval_en:.3f})")
print()
