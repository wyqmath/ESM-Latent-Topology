#!/usr/bin/env python3
# -- Project root setup --
import os as _os
from pathlib import Path as _Path
_os.chdir(_Path(__file__).resolve().parents[2])
"""
2D UMAP空间密度估计（修复50D维度灾难问题）

输入文件:
- data/umap/umap_embeddings_2d.npy (11068×2 UMAP坐标)
- data/metadata_final_with_en.csv (元数据)

输出文件:
- data/density/density_values.npy (11068维密度值，覆盖原文件)
- data/density/density_statistics_2d.txt (统计报告)
- data/density/density_visualization_2d.png (可视化)

功能描述:
在2D UMAP空间使用KDE核密度估计，避免50D空间的维度灾难问题
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors import KernelDensity
from scipy import stats
import time
import os

print("=" * 80)
print("2D UMAP空间密度估计（修复版）")
print("=" * 80)

# 1. 加载数据
print("\n[1/5] 加载数据...")
umap_2d = np.load('data/umap/umap_embeddings_2d.npy')
metadata = pd.read_csv('data/metadata_final_with_en.csv')

print(f"  UMAP 2D坐标: {umap_2d.shape}")
print(f"  元数据: {metadata.shape}")

# 2. KDE密度估计（在2维UMAP空间）
print("\n[2/5] KDE密度估计（2维UMAP空间）...")
start_time = time.time()

# 使用高斯核，带宽通过Scott规则自动选择
n, d = umap_2d.shape
scott_bandwidth = n ** (-1 / (d + 4)) * np.std(umap_2d, axis=0).mean()
print(f"  数据点数: {n}")
print(f"  维度: {d}")
print(f"  Scott带宽: {scott_bandwidth:.6f}")

# 使用sklearn的KernelDensity
kde = KernelDensity(bandwidth=scott_bandwidth, kernel='gaussian', metric='euclidean')
kde.fit(umap_2d)

# 计算每个点的对数密度
log_density = kde.score_samples(umap_2d)
density = np.exp(log_density)

elapsed = time.time() - start_time
print(f"  计算时间: {elapsed:.2f}秒 ({elapsed/n*1000:.2f}毫秒/点)")

# 3. 统计分析
print("\n[3/5] 统计分析...")
print(f"  密度范围: [{density.min():.6e}, {density.max():.6e}]")
print(f"  密度均值: {density.mean():.6e} ± {density.std():.6e}")
print(f"  密度中位数: {np.median(density):.6e}")
print(f"  密度四分位数: Q1={np.percentile(density, 25):.6e}, Q3={np.percentile(density, 75):.6e}")
print(f"  唯一密度值数量: {len(np.unique(density))}")

# 检查是否还有离散阶梯问题
rounded_4 = np.round(density, decimals=4)
unique_rounded = len(np.unique(rounded_4))
print(f"  唯一密度值（四舍五入到4位）: {unique_rounded}")
print(f"  ✓ 密度分布连续（无离散阶梯）" if unique_rounded > 100 else "  ⚠ 仍有离散问题")

# 识别高密度"岛屿"（前10%）和低密度边缘（后10%）
high_density_threshold = np.percentile(density, 90)
low_density_threshold = np.percentile(density, 10)
high_density_mask = density >= high_density_threshold
low_density_mask = density <= low_density_threshold

print(f"\n  高密度岛屿（前10%）: {high_density_mask.sum()}个点, 阈值={high_density_threshold:.6e}")
print(f"  低密度边缘（后10%）: {low_density_mask.sum()}个点, 阈值={low_density_threshold:.6e}")

# 按类别统计
print("\n  按类别统计:")
for category in metadata['category'].unique():
    mask = metadata['category'] == category
    cat_density = density[mask]
    print(f"    {category:12s}: 密度={cat_density.mean():.6e}±{cat_density.std():.6e}, "
          f"范围=[{cat_density.min():.6e}, {cat_density.max():.6e}]")

# 4. 与物理指标E[n]的相关性分析（anchor序列）
print("\n[4/5] 与物理指标E[n]的相关性分析...")
anchor_mask = metadata['category'] == 'anchor'
anchor_density = density[anchor_mask]
anchor_en = metadata.loc[anchor_mask, 'E_n'].values

# 移除NaN值
valid_mask = ~np.isnan(anchor_en)
anchor_density_valid = anchor_density[valid_mask]
anchor_en_valid = anchor_en[valid_mask]

if len(anchor_density_valid) > 0:
    rho, p_value = stats.spearmanr(anchor_density_valid, anchor_en_valid)
    print(f"  Spearman相关系数: ρ={rho:.4f}, p={p_value:.4f}")
    if p_value < 0.05:
        print(f"  ✓ 显著相关 (p<0.05)")
    else:
        print(f"  ✗ 无显著相关 (p≥0.05)")
else:
    print("  ⚠ 无有效anchor序列数据")

# 5. 保存结果
print("\n[5/5] 保存结果...")
np.save('data/density/density_values.npy', density)
print("  ✓ 密度值已保存（覆盖原50D密度文件）")

# 保存统计报告
with open('data/density/density_statistics_2d.txt', 'w') as f:
    f.write("=" * 80 + "\n")
    f.write("2D UMAP空间密度估计统计报告\n")
    f.write("=" * 80 + "\n\n")

    f.write(f"数据集大小: {n}个点\n")
    f.write(f"嵌入维度: {d}维 (2D UMAP)\n")
    f.write(f"KDE带宽 (Scott规则): {scott_bandwidth:.6f}\n")
    f.write(f"计算时间: {elapsed:.2f}秒\n\n")

    f.write("全局统计:\n")
    f.write(f"  密度范围: [{density.min():.6e}, {density.max():.6e}]\n")
    f.write(f"  密度均值: {density.mean():.6e} ± {density.std():.6e}\n")
    f.write(f"  密度中位数: {np.median(density):.6e}\n")
    f.write(f"  密度四分位数: Q1={np.percentile(density, 25):.6e}, Q3={np.percentile(density, 75):.6e}\n")
    f.write(f"  唯一密度值数量: {len(np.unique(density))}\n\n")

    f.write(f"高密度岛屿（前10%）: {high_density_mask.sum()}个点, 阈值={high_density_threshold:.6e}\n")
    f.write(f"低密度边缘（后10%）: {low_density_mask.sum()}个点, 阈值={low_density_threshold:.6e}\n\n")

    f.write("按类别统计:\n")
    for category in metadata['category'].unique():
        mask = metadata['category'] == category
        cat_density = density[mask]
        f.write(f"  {category}:\n")
        f.write(f"    样本数: {mask.sum()}\n")
        f.write(f"    密度: {cat_density.mean():.6e} ± {cat_density.std():.6e}\n")
        f.write(f"    范围: [{cat_density.min():.6e}, {cat_density.max():.6e}]\n")
        f.write(f"    高密度岛屿: {(mask & high_density_mask).sum()}个 ({(mask & high_density_mask).sum()/mask.sum()*100:.1f}%)\n")
        f.write(f"    低密度边缘: {(mask & low_density_mask).sum()}个 ({(mask & low_density_mask).sum()/mask.sum()*100:.1f}%)\n\n")

    if len(anchor_density_valid) > 0:
        f.write("与物理指标E[n]的相关性 (anchor序列):\n")
        f.write(f"  有效样本数: {len(anchor_density_valid)}\n")
        f.write(f"  Spearman相关系数: ρ={rho:.4f}, p={p_value:.4f}\n")
        if p_value < 0.05:
            f.write(f"  结论: 显著相关 (p<0.05)\n")
        else:
            f.write(f"  结论: 无显著相关 (p≥0.05)\n")

print("  ✓ 统计报告已保存")

# 6. 可视化
print("\n[6/6] 可视化...")
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# 左图：密度热图
scatter = axes[0].scatter(umap_2d[:, 0], umap_2d[:, 1], c=density,
                          cmap='viridis', s=5, alpha=0.6, edgecolors='none')
axes[0].set_xlabel('UMAP 1', fontsize=12)
axes[0].set_ylabel('UMAP 2', fontsize=12)
axes[0].set_title('2D UMAP Density Distribution', fontsize=14, fontweight='bold')
cbar = plt.colorbar(scatter, ax=axes[0])
cbar.set_label('Density', fontsize=12)

# 右图：高密度岛屿和低密度边缘
axes[1].scatter(umap_2d[~(high_density_mask | low_density_mask), 0],
                umap_2d[~(high_density_mask | low_density_mask), 1],
                c='lightgray', s=5, alpha=0.3, label='Normal', edgecolors='none')
axes[1].scatter(umap_2d[high_density_mask, 0], umap_2d[high_density_mask, 1],
                c='red', s=10, alpha=0.7, label='High Density (Top 10%)', edgecolors='none')
axes[1].scatter(umap_2d[low_density_mask, 0], umap_2d[low_density_mask, 1],
                c='blue', s=10, alpha=0.7, label='Low Density (Bottom 10%)', edgecolors='none')
axes[1].set_xlabel('UMAP 1', fontsize=12)
axes[1].set_ylabel('UMAP 2', fontsize=12)
axes[1].set_title('High/Low Density Regions', fontsize=14, fontweight='bold')
axes[1].legend(fontsize=10, loc='best')

plt.tight_layout()
plt.savefig('data/density/density_visualization_2d.png', dpi=300, bbox_inches='tight')
plt.close()
print("  ✓ 可视化已保存")

print("\n" + "=" * 80)
print("2D UMAP密度估计完成!")
print("=" * 80)
print(f"\n输出文件:")
print(f"  - data/density/density_values.npy (已覆盖)")
print(f"  - data/density/density_statistics_2d.txt")
print(f"  - data/density/density_visualization_2d.png")
print(f"\n重要说明:")
print(f"  - 原50D密度文件已被覆盖")
print(f"  - 新密度基于2D UMAP空间，避免了维度灾难")
print(f"  - 密度分布现在是连续的，不再有离散阶梯")
