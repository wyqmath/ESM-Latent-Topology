#!/usr/bin/env python3
# -- Project root setup --
import os as _os
from pathlib import Path as _Path
_os.chdir(_Path(__file__).resolve().parents[2])
"""
识别流形上的相边界

输入文件:
- data/umap/umap_embeddings_2d.npy (UMAP降维结果)
- data/density/density_values.npy (密度值)
- data/curvature/ricci_curvature.npy (曲率值)
- data/metadata_final_with_en.csv (元数据)

输出文件:
- data/phase_boundaries/density_gradient_field.npy (密度梯度场)
- data/phase_boundaries/phase_boundary_candidates.csv (相边界候选点)
- data/phase_boundaries/phase_boundary_visualization.png (可视化)
- data/phase_boundaries/phase_boundary_report.txt (分析报告)

功能描述:
通过密度梯度、曲率变化、拓扑特征突变识别流形上的相边界区域，
分析这些区域是否对应蛋白质折叠的相变
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy.interpolate import griddata
from scipy.ndimage import gaussian_filter, sobel
from scipy.spatial import KDTree
from sklearn.cluster import DBSCAN

# 设置随机种子
np.random.seed(42)

# 创建输出目录
output_dir = Path("data/phase_boundaries")
output_dir.mkdir(parents=True, exist_ok=True)

print("=" * 80)
print("识别流形上的相边界")
print("=" * 80)

# 1. 加载数据
print("\n[1/6] 加载数据...")
umap_coords = np.load("data/umap/umap_embeddings_2d.npy")
density = np.load("data/density/density_values.npy")
curvature = np.load("data/curvature/ricci_curvature.npy")
metadata = pd.read_csv("data/metadata_final_with_en.csv")

print(f"  UMAP坐标: {umap_coords.shape}")
print(f"  密度值: {density.shape}")
print(f"  曲率值: {curvature.shape}")
print(f"  元数据: {metadata.shape}")

# 计算局部密度变化率 (替代topological_features)
print("  计算局部密度变化率...")
from sklearn.neighbors import NearestNeighbors
nbrs = NearestNeighbors(n_neighbors=15, metric='euclidean').fit(umap_coords)
distances, indices = nbrs.kneighbors(umap_coords)
local_density_std = np.array([density[idx].std() for idx in indices])
print(f"  局部密度变化率: {local_density_std.shape}")

# 2. 计算密度梯度场
print("\n[2/6] 计算密度梯度场...")
# 创建网格
x_min, x_max = umap_coords[:, 0].min(), umap_coords[:, 0].max()
y_min, y_max = umap_coords[:, 1].min(), umap_coords[:, 1].max()
grid_resolution = 200
grid_x, grid_y = np.mgrid[x_min:x_max:complex(grid_resolution),
                           y_min:y_max:complex(grid_resolution)]

# 插值密度到网格
log_density = np.log10(density + 1e-60)
grid_density = griddata(umap_coords, log_density, (grid_x, grid_y), method='cubic')
grid_density = gaussian_filter(grid_density, sigma=2)  # 平滑

# 计算梯度
grad_y, grad_x = np.gradient(grid_density)
grad_magnitude = np.sqrt(grad_x**2 + grad_y**2)

# 保存梯度场
np.save(output_dir / "density_gradient_field.npy",
        {'grid_x': grid_x, 'grid_y': grid_y, 'grad_magnitude': grad_magnitude,
         'grad_x': grad_x, 'grad_y': grad_y, 'grid_density': grid_density})

print(f"  梯度幅值范围: {np.nanmin(grad_magnitude):.4f} - {np.nanmax(grad_magnitude):.4f}")
print(f"  梯度幅值均值: {np.nanmean(grad_magnitude):.4f} ± {np.nanstd(grad_magnitude):.4f}")

# 3. 识别相边界候选点
print("\n[3/6] 识别相边界候选点...")
# 方法1: 密度梯度最大的点
grad_threshold = np.nanpercentile(grad_magnitude, 95)
print(f"  密度梯度阈值 (top 5%): {grad_threshold:.4f}")

# 为每个数据点计算最近网格点的梯度
tree = KDTree(np.c_[grid_x.ravel(), grid_y.ravel()])
distances, indices = tree.query(umap_coords)
point_gradients = grad_magnitude.ravel()[indices]

high_gradient_mask = point_gradients > grad_threshold
print(f"  高梯度点数: {np.sum(high_gradient_mask)} ({np.sum(high_gradient_mask)/len(umap_coords)*100:.2f}%)")

# 方法2: 局部密度变化率 (已在加载数据时计算)
density_change_threshold = np.percentile(local_density_std, 95)
high_density_change_mask = local_density_std > density_change_threshold
print(f"  高密度变化点数: {np.sum(high_density_change_mask)} ({np.sum(high_density_change_mask)/len(umap_coords)*100:.2f}%)")

# 方法3: 曲率突变点
curvature_diff = np.abs(curvature - np.median(curvature))
curvature_threshold = np.percentile(curvature_diff, 95)
high_curvature_change_mask = curvature_diff > curvature_threshold
print(f"  高曲率变化点数: {np.sum(high_curvature_change_mask)} ({np.sum(high_curvature_change_mask)/len(umap_coords)*100:.2f}%)")

# 综合判断：至少满足两个条件
phase_boundary_score = (high_gradient_mask.astype(int) +
                        high_density_change_mask.astype(int) +
                        high_curvature_change_mask.astype(int))
phase_boundary_mask = phase_boundary_score >= 2
print(f"  相边界候选点数 (≥2条件): {np.sum(phase_boundary_mask)} ({np.sum(phase_boundary_mask)/len(umap_coords)*100:.2f}%)")

# 4. 聚类相边界区域
print("\n[4/6] 聚类相边界区域...")
if np.sum(phase_boundary_mask) > 0:
    boundary_coords = umap_coords[phase_boundary_mask]
    clustering = DBSCAN(eps=2.0, min_samples=5).fit(boundary_coords)
    n_clusters = len(set(clustering.labels_)) - (1 if -1 in clustering.labels_ else 0)
    n_noise = list(clustering.labels_).count(-1)
    print(f"  识别到 {n_clusters} 个相边界区域")
    print(f"  噪声点数: {n_noise}")

    # 为每个簇计算中心和统计信息
    cluster_info = []
    for cluster_id in range(n_clusters):
        cluster_mask_local = clustering.labels_ == cluster_id
        cluster_indices = np.where(phase_boundary_mask)[0][cluster_mask_local]

        cluster_center = umap_coords[cluster_indices].mean(axis=0)
        cluster_size = len(cluster_indices)
        cluster_density_mean = density[cluster_indices].mean()
        cluster_curvature_mean = curvature[cluster_indices].mean()

        # 统计类别分布
        cluster_categories = metadata.loc[cluster_indices, 'category'].value_counts()
        dominant_category = cluster_categories.index[0] if len(cluster_categories) > 0 else 'unknown'

        cluster_info.append({
            'cluster_id': cluster_id,
            'center_x': cluster_center[0],
            'center_y': cluster_center[1],
            'size': cluster_size,
            'density_mean': cluster_density_mean,
            'curvature_mean': cluster_curvature_mean,
            'dominant_category': dominant_category
        })

    cluster_df = pd.DataFrame(cluster_info)
    print(f"\n  相边界区域统计:")
    print(cluster_df.to_string(index=False))
else:
    n_clusters = 0
    clustering = None
    cluster_df = pd.DataFrame()

# 5. 保存相边界候选点
print("\n[5/6] 保存相边界候选点...")
boundary_data = pd.DataFrame({
    'seq_id': metadata['seq_id'],
    'category': metadata['category'],
    'umap1': umap_coords[:, 0],
    'umap2': umap_coords[:, 1],
    'gradient_magnitude': point_gradients,
    'local_density_std': local_density_std,
    'curvature_diff': curvature_diff,
    'phase_boundary_score': phase_boundary_score,
    'is_boundary': phase_boundary_mask
})

if n_clusters > 0:
    cluster_labels = np.full(len(umap_coords), -1)
    cluster_labels[phase_boundary_mask] = clustering.labels_
    boundary_data['cluster_id'] = cluster_labels

output_path = output_dir / "phase_boundary_candidates.csv"
boundary_data.to_csv(output_path, index=False)
print(f"  保存候选点: {output_path}")

# 6. 可视化
print("\n[6/6] 创建可视化...")
fig = plt.figure(figsize=(20, 12))
gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)

# 6.1 密度梯度场
ax1 = fig.add_subplot(gs[0, 0])
im1 = ax1.contourf(grid_x, grid_y, grad_magnitude, levels=20, cmap='YlOrRd', alpha=0.8)
plt.colorbar(im1, ax=ax1, label='Gradient Magnitude')
ax1.scatter(umap_coords[:, 0], umap_coords[:, 1], c='gray', s=1, alpha=0.3, rasterized=True)
if np.sum(phase_boundary_mask) > 0:
    ax1.scatter(umap_coords[phase_boundary_mask, 0], umap_coords[phase_boundary_mask, 1],
                c='red', s=20, alpha=0.8, marker='*', label='Phase Boundary')
ax1.set_xlabel('UMAP 1', fontsize=12)
ax1.set_ylabel('UMAP 2', fontsize=12)
ax1.set_title('Density Gradient Field', fontsize=14, fontweight='bold')
ax1.legend(fontsize=10)
ax1.grid(True, alpha=0.3)

# 6.2 相边界候选点（按分数着色）
ax2 = fig.add_subplot(gs[0, 1])
scatter = ax2.scatter(umap_coords[:, 0], umap_coords[:, 1],
                     c=phase_boundary_score, s=10, alpha=0.6,
                     cmap='RdYlGn_r', vmin=0, vmax=3, rasterized=True)
plt.colorbar(scatter, ax=ax2, label='Boundary Score (0-3)', ticks=[0, 1, 2, 3])
ax2.set_xlabel('UMAP 1', fontsize=12)
ax2.set_ylabel('UMAP 2', fontsize=12)
ax2.set_title('Phase Boundary Score', fontsize=14, fontweight='bold')
ax2.grid(True, alpha=0.3)

# 6.3 相边界聚类
ax3 = fig.add_subplot(gs[0, 2])
ax3.scatter(umap_coords[:, 0], umap_coords[:, 1], c='lightgray', s=1, alpha=0.3, rasterized=True)
if n_clusters > 0:
    colors = plt.cm.tab10(np.linspace(0, 1, n_clusters))
    for cluster_id in range(n_clusters):
        cluster_mask_local = clustering.labels_ == cluster_id
        cluster_indices = np.where(phase_boundary_mask)[0][cluster_mask_local]
        ax3.scatter(umap_coords[cluster_indices, 0], umap_coords[cluster_indices, 1],
                   c=[colors[cluster_id]], s=30, alpha=0.8, label=f'Cluster {cluster_id}')
    # 标注噪声点
    noise_mask_local = clustering.labels_ == -1
    if np.sum(noise_mask_local) > 0:
        noise_indices = np.where(phase_boundary_mask)[0][noise_mask_local]
        ax3.scatter(umap_coords[noise_indices, 0], umap_coords[noise_indices, 1],
                   c='black', s=10, alpha=0.5, marker='x', label='Noise')
ax3.set_xlabel('UMAP 1', fontsize=12)
ax3.set_ylabel('UMAP 2', fontsize=12)
ax3.set_title(f'Phase Boundary Clusters (n={n_clusters})', fontsize=14, fontweight='bold')
if n_clusters > 0:
    ax3.legend(fontsize=8, loc='best', ncol=2)
ax3.grid(True, alpha=0.3)

# 6.4 按类别分布
ax4 = fig.add_subplot(gs[1, 0])
if np.sum(phase_boundary_mask) > 0:
    boundary_categories = metadata.loc[phase_boundary_mask, 'category'].value_counts()
    ax4.bar(range(len(boundary_categories)), boundary_categories.values, color='steelblue')
    ax4.set_xticks(range(len(boundary_categories)))
    ax4.set_xticklabels(boundary_categories.index, rotation=45, ha='right')
    ax4.set_ylabel('Count', fontsize=12)
    ax4.set_title('Category Distribution in Phase Boundaries', fontsize=14, fontweight='bold')
    ax4.grid(True, alpha=0.3, axis='y')

    # 添加百分比标签
    total = boundary_categories.sum()
    for i, v in enumerate(boundary_categories.values):
        ax4.text(i, v + total*0.01, f'{v/total*100:.1f}%', ha='center', fontsize=10)

# 6.5 梯度vs密度散点图
ax5 = fig.add_subplot(gs[1, 1])
scatter = ax5.scatter(log_density, point_gradients, c=phase_boundary_score,
                     s=5, alpha=0.5, cmap='RdYlGn_r', vmin=0, vmax=3, rasterized=True)
ax5.axhline(grad_threshold, color='red', linestyle='--', linewidth=2, label=f'Threshold (95%)')
ax5.set_xlabel('log10(Density)', fontsize=12)
ax5.set_ylabel('Gradient Magnitude', fontsize=12)
ax5.set_title('Gradient vs Density', fontsize=14, fontweight='bold')
ax5.legend(fontsize=10)
ax5.grid(True, alpha=0.3)

# 6.6 曲率vs梯度散点图
ax6 = fig.add_subplot(gs[1, 2])
scatter = ax6.scatter(curvature, point_gradients, c=phase_boundary_score,
                     s=5, alpha=0.5, cmap='RdYlGn_r', vmin=0, vmax=3, rasterized=True)
ax6.axhline(grad_threshold, color='red', linestyle='--', linewidth=2, label=f'Gradient Threshold')
ax6.set_xlabel('Ricci Curvature', fontsize=12)
ax6.set_ylabel('Gradient Magnitude', fontsize=12)
ax6.set_title('Gradient vs Curvature', fontsize=14, fontweight='bold')
ax6.legend(fontsize=10)
ax6.grid(True, alpha=0.3)

plt.suptitle('Phase Boundary Identification on UMAP Manifold',
             fontsize=18, fontweight='bold', y=0.995)

output_path = output_dir / "phase_boundary_visualization.png"
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"  保存可视化: {output_path}")
plt.close()

# 7. 生成分析报告
print("\n[7/7] 生成分析报告...")
report_lines = []
report_lines.append("=" * 80)
report_lines.append("流形相边界识别报告")
report_lines.append("=" * 80)
report_lines.append("")

report_lines.append("1. 相边界识别方法")
report_lines.append("   - 方法1: 密度梯度最大区域 (top 5%)")
report_lines.append("   - 方法2: 局部密度变化率最大区域 (top 5%)")
report_lines.append("   - 方法3: 曲率突变区域 (top 5%)")
report_lines.append("   - 综合判断: 至少满足2个条件的点被标记为相边界候选点")
report_lines.append("")

report_lines.append("2. 相边界候选点统计")
report_lines.append(f"   总数据点: {len(umap_coords)}")
report_lines.append(f"   高梯度点: {np.sum(high_gradient_mask)} ({np.sum(high_gradient_mask)/len(umap_coords)*100:.2f}%)")
report_lines.append(f"   高密度变化点: {np.sum(high_density_change_mask)} ({np.sum(high_density_change_mask)/len(umap_coords)*100:.2f}%)")
report_lines.append(f"   高曲率变化点: {np.sum(high_curvature_change_mask)} ({np.sum(high_curvature_change_mask)/len(umap_coords)*100:.2f}%)")
report_lines.append(f"   相边界候选点: {np.sum(phase_boundary_mask)} ({np.sum(phase_boundary_mask)/len(umap_coords)*100:.2f}%)")
report_lines.append("")

report_lines.append("3. 相边界区域聚类")
report_lines.append(f"   识别到的相边界区域数: {n_clusters}")
if n_clusters > 0:
    report_lines.append(f"   噪声点数: {n_noise}")
    report_lines.append("")
    report_lines.append("   各区域详细信息:")
    for _, row in cluster_df.iterrows():
        report_lines.append(f"   - Cluster {int(row['cluster_id'])}: {int(row['size'])}点, "
                          f"中心({row['center_x']:.2f}, {row['center_y']:.2f}), "
                          f"主要类别={row['dominant_category']}")
report_lines.append("")

report_lines.append("4. 类别分布分析")
if np.sum(phase_boundary_mask) > 0:
    boundary_categories = metadata.loc[phase_boundary_mask, 'category'].value_counts()
    report_lines.append("   相边界区域的类别分布:")
    for cat, count in boundary_categories.items():
        pct = count / np.sum(phase_boundary_mask) * 100
        report_lines.append(f"   - {cat}: {count} ({pct:.1f}%)")
report_lines.append("")

report_lines.append("5. 物理意义解释")
report_lines.append("   相边界候选区域可能对应:")
report_lines.append("   - 蛋白质折叠的相变边界（如有序-无序转变）")
report_lines.append("   - 不同折叠类型之间的过渡区域")
report_lines.append("   - 结构不稳定性高的区域（如fold-switching蛋白）")
report_lines.append("")

report_lines.append("=" * 80)

report_text = "\n".join(report_lines)
print(report_text)

output_path = output_dir / "phase_boundary_report.txt"
with open(output_path, 'w') as f:
    f.write(report_text)
print(f"\n保存报告: {output_path}")

print("\n" + "=" * 80)
print("相边界识别完成！")
print("=" * 80)
print(f"输出文件:")
print(f"  1. {output_dir / 'density_gradient_field.npy'}")
print(f"  2. {output_dir / 'phase_boundary_candidates.csv'}")
print(f"  3. {output_dir / 'phase_boundary_visualization.png'}")
print(f"  4. {output_dir / 'phase_boundary_report.txt'}")
