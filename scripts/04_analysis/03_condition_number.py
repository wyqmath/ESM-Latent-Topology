# -- Project root setup --
import os as _os
from pathlib import Path as _Path
_os.chdir(_Path(__file__).resolve().parents[2])
"""
流形条件数计算

输入文件:
- data/pca/pca_embeddings_50d.npy (11068×50 PCA嵌入)
- data/metadata_final_with_en.csv (元数据+物理指标)

输出文件:
- data/condition_number/condition_numbers.npy (11068维条件数)
- data/condition_number/condition_statistics.txt (统计报告)
- data/condition_number/condition_visualization.png (可视化)
- data/condition_number/condition_by_category.png (按类别分布)

功能描述:
对每个点计算局部协方差矩阵的条件数(最大/最小特征值)，衡量局部几何的各向异性程度。
高条件数表示局部几何"扁平"或"拉伸"，低条件数表示各向同性。
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
from scipy import stats
import os
import time

# 创建输出目录
os.makedirs('data/condition_number', exist_ok=True)

print("=" * 80)
print("流形条件数计算")
print("=" * 80)

# 1. 加载数据
print("\n[1/5] 加载数据...")
embeddings_50d = np.load('data/pca/pca_embeddings_50d.npy')
metadata = pd.read_csv('data/metadata_final_with_en.csv')

print(f"  嵌入维度: {embeddings_50d.shape}")
print(f"  元数据行数: {len(metadata)}")

# 2. 计算局部条件数
print("\n[2/5] 计算局部条件数...")
print("  使用k近邻构建局部协方差矩阵")

k = 15  # 近邻数量（与UMAP参数一致）
n_points = len(embeddings_50d)

# 构建k-NN图
print(f"  构建k-NN图 (k={k})...")
start_time = time.time()
nbrs = NearestNeighbors(n_neighbors=k+1, algorithm='auto', n_jobs=-1)
nbrs.fit(embeddings_50d)
distances, indices = nbrs.kneighbors(embeddings_50d)
knn_time = time.time() - start_time
print(f"  k-NN构建完成，耗时 {knn_time:.2f}秒")

# 计算每个点的局部条件数
print(f"  计算局部协方差矩阵条件数...")
condition_numbers = np.zeros(n_points)

start_time = time.time()
for i in range(n_points):
    # 获取k近邻（排除自身）
    neighbor_indices = indices[i, 1:]  # 排除第一个（自身）
    neighbors = embeddings_50d[neighbor_indices]

    # 中心化
    centered = neighbors - neighbors.mean(axis=0)

    # 计算协方差矩阵
    cov_matrix = np.cov(centered.T)

    # 计算特征值
    eigenvalues = np.linalg.eigvalsh(cov_matrix)
    eigenvalues = np.abs(eigenvalues)  # 取绝对值避免数值误差
    eigenvalues = eigenvalues[eigenvalues > 1e-10]  # 过滤接近零的特征值

    if len(eigenvalues) > 0:
        condition_numbers[i] = eigenvalues.max() / eigenvalues.min()
    else:
        condition_numbers[i] = 1.0  # 默认值

    if (i + 1) % 2000 == 0:
        elapsed = time.time() - start_time
        rate = (i + 1) / elapsed
        eta = (n_points - i - 1) / rate
        print(f"    进度: {i+1}/{n_points} ({100*(i+1)/n_points:.1f}%), "
              f"速率: {rate:.1f} 点/秒, ETA: {eta:.1f}秒")

compute_time = time.time() - start_time
print(f"  条件数计算完成，耗时 {compute_time:.2f}秒 ({n_points/compute_time:.2f} 点/秒)")

# 保存条件数
np.save('data/condition_number/condition_numbers.npy', condition_numbers)
print(f"  已保存: condition_numbers.npy")

# 3. 统计分析
print("\n[3/5] 统计分析...")
print(f"  条件数范围: [{condition_numbers.min():.2f}, {condition_numbers.max():.2f}]")
print(f"  平均条件数: {condition_numbers.mean():.2f} ± {condition_numbers.std():.2f}")
print(f"  中位数: {np.median(condition_numbers):.2f}")
print(f"  25%分位数: {np.percentile(condition_numbers, 25):.2f}")
print(f"  75%分位数: {np.percentile(condition_numbers, 75):.2f}")

# 按类别统计
print("\n  按类别统计:")
for category in metadata['category'].unique():
    mask = metadata['category'] == category
    cond_cat = condition_numbers[mask]
    print(f"    {category:12s}: {cond_cat.mean():.2f} ± {cond_cat.std():.2f} "
          f"(范围: [{cond_cat.min():.2f}, {cond_cat.max():.2f}])")

# 4. 与物理指标相关性分析
print("\n[4/5] 与物理指标相关性分析...")
anchor_mask = metadata['category'] == 'anchor'
anchor_indices = np.where(anchor_mask)[0]

if anchor_mask.sum() > 0:
    anchor_metadata = metadata[anchor_mask].copy()
    anchor_condition = condition_numbers[anchor_indices]

    # 与E[n]相关性
    valid_mask = ~anchor_metadata['E_n'].isna()
    if valid_mask.sum() > 0:
        e_n_values = anchor_metadata.loc[valid_mask, 'E_n'].values
        cond_values = anchor_condition[valid_mask.values]

        corr, pval = stats.spearmanr(cond_values, e_n_values)
        print(f"  条件数 vs E[n] (anchor序列, n={valid_mask.sum()}):")
        print(f"    Spearman ρ = {corr:.4f}, p = {pval:.4f}")
        if pval < 0.05:
            print(f"    ✓ 显著相关 (p < 0.05)")
        else:
            print(f"    ✗ 无显著相关 (p ≥ 0.05)")

# 5. 可视化
print("\n[5/5] 可视化...")

# 5.1 条件数分布直方图
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 整体分布
ax = axes[0, 0]
ax.hist(condition_numbers, bins=50, edgecolor='black', alpha=0.7)
ax.axvline(condition_numbers.mean(), color='red', linestyle='--',
           label=f'Mean: {condition_numbers.mean():.2f}')
ax.axvline(np.median(condition_numbers), color='blue', linestyle='--',
           label=f'Median: {np.median(condition_numbers):.2f}')
ax.set_xlabel('Condition Number', fontsize=12)
ax.set_ylabel('Frequency', fontsize=12)
ax.set_title('Overall Distribution of Condition Numbers', fontsize=13, fontweight='bold')
ax.legend()
ax.grid(alpha=0.3)

# 按类别分布（箱线图）
ax = axes[0, 1]
categories = metadata['category'].unique()
data_by_cat = [condition_numbers[metadata['category'] == cat] for cat in categories]
bp = ax.boxplot(data_by_cat, labels=categories, patch_artist=True)
for patch in bp['boxes']:
    patch.set_facecolor('lightblue')
ax.set_ylabel('Condition Number', fontsize=12)
ax.set_title('Condition Number by Category', fontsize=13, fontweight='bold')
ax.grid(axis='y', alpha=0.3)
plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

# 条件数 vs E[n] (anchor序列)
ax = axes[1, 0]
if anchor_mask.sum() > 0 and valid_mask.sum() > 0:
    scatter = ax.scatter(cond_values, e_n_values, alpha=0.5, s=20)
    ax.set_xlabel('Condition Number', fontsize=12)
    ax.set_ylabel('E[n]', fontsize=12)
    ax.set_title(f'Condition Number vs E[n] (anchor, ρ={corr:.3f}, p={pval:.3f})',
                 fontsize=13, fontweight='bold')
    ax.grid(alpha=0.3)

    # 添加趋势线
    z = np.polyfit(cond_values, e_n_values, 1)
    p = np.poly1d(z)
    x_trend = np.linspace(cond_values.min(), cond_values.max(), 100)
    ax.plot(x_trend, p(x_trend), 'r--', alpha=0.8, label='Linear fit')
    ax.legend()
else:
    ax.text(0.5, 0.5, 'No anchor data with E[n]',
            ha='center', va='center', transform=ax.transAxes)
    ax.set_title('Condition Number vs E[n]', fontsize=13, fontweight='bold')

# Log-scale分布
ax = axes[1, 1]
ax.hist(np.log10(condition_numbers), bins=50, edgecolor='black', alpha=0.7, color='orange')
ax.set_xlabel('log10(Condition Number)', fontsize=12)
ax.set_ylabel('Frequency', fontsize=12)
ax.set_title('Log-scale Distribution', fontsize=13, fontweight='bold')
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('data/condition_number/condition_visualization.png', dpi=300, bbox_inches='tight')
print("  已保存: condition_visualization.png")
plt.close()

# 5.2 按类别详细分布
fig, ax = plt.subplots(figsize=(12, 6))
for category in categories:
    mask = metadata['category'] == category
    cond_cat = condition_numbers[mask]
    ax.hist(cond_cat, bins=30, alpha=0.5, label=f'{category} (n={mask.sum()})', edgecolor='black')

ax.set_xlabel('Condition Number', fontsize=12)
ax.set_ylabel('Frequency', fontsize=12)
ax.set_title('Condition Number Distribution by Category', fontsize=13, fontweight='bold')
ax.legend()
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('data/condition_number/condition_by_category.png', dpi=300, bbox_inches='tight')
print("  已保存: condition_by_category.png")
plt.close()

# 6. 保存统计报告
print("\n[6/6] 保存统计报告...")
with open('data/condition_number/condition_statistics.txt', 'w') as f:
    f.write("=" * 80 + "\n")
    f.write("流形条件数统计报告\n")
    f.write("=" * 80 + "\n\n")

    f.write("计算参数:\n")
    f.write(f"  k近邻数: {k}\n")
    f.write(f"  嵌入维度: {embeddings_50d.shape[1]}\n")
    f.write(f"  数据点数: {n_points}\n")
    f.write(f"  计算时间: {compute_time:.2f}秒 ({n_points/compute_time:.2f} 点/秒)\n\n")

    f.write("整体统计:\n")
    f.write(f"  条件数范围: [{condition_numbers.min():.2f}, {condition_numbers.max():.2f}]\n")
    f.write(f"  平均值: {condition_numbers.mean():.2f} ± {condition_numbers.std():.2f}\n")
    f.write(f"  中位数: {np.median(condition_numbers):.2f}\n")
    f.write(f"  25%分位数: {np.percentile(condition_numbers, 25):.2f}\n")
    f.write(f"  75%分位数: {np.percentile(condition_numbers, 75):.2f}\n\n")

    f.write("按类别统计:\n")
    for category in categories:
        mask = metadata['category'] == category
        cond_cat = condition_numbers[mask]
        f.write(f"  {category}:\n")
        f.write(f"    样本数: {mask.sum()}\n")
        f.write(f"    平均值: {cond_cat.mean():.2f} ± {cond_cat.std():.2f}\n")
        f.write(f"    范围: [{cond_cat.min():.2f}, {cond_cat.max():.2f}]\n")
        f.write(f"    中位数: {np.median(cond_cat):.2f}\n\n")

    f.write("与物理指标相关性 (anchor序列):\n")
    if anchor_mask.sum() > 0 and valid_mask.sum() > 0:
        f.write(f"  条件数 vs E[n]:\n")
        f.write(f"    样本数: {valid_mask.sum()}\n")
        f.write(f"    Spearman ρ: {corr:.4f}\n")
        f.write(f"    p-value: {pval:.4f}\n")
        if pval < 0.05:
            f.write(f"    结论: 显著相关 (p < 0.05)\n")
        else:
            f.write(f"    结论: 无显著相关 (p ≥ 0.05)\n")
    else:
        f.write("  无可用数据\n")

print("  已保存: condition_statistics.txt")

print("\n" + "=" * 80)
print("流形条件数计算完成!")
print("=" * 80)
print(f"\n输出文件:")
print(f"  - condition_numbers.npy ({condition_numbers.nbytes / 1024:.2f} KB)")
print(f"  - condition_statistics.txt")
print(f"  - condition_visualization.png")
print(f"  - condition_by_category.png")
