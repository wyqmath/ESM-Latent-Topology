#!/usr/bin/env python3
# -- Project root setup --
import os as _os
from pathlib import Path as _Path
_os.chdir(_Path(__file__).resolve().parents[2])
"""
Wilson环在ESM-2隐空间的计算

输入文件:
- data/pca/pca_embeddings_50d.npy (11068×50 PCA嵌入)
- data/umap/umap_embeddings_2d.npy (11068×2 UMAP坐标)
- data/metadata_final_with_en.csv (元数据)
- data/phase_boundaries/phase_boundary_candidates.csv (相边界点)

输出文件:
- data/wilson_loops/wilson_loop_values.csv (Wilson环值)
- data/wilson_loops/wilson_loop_paths.npy (路径坐标)
- data/wilson_loops/wilson_loop_visualization.png (可视化)
- data/wilson_loops/wilson_loop_report.txt (分析报告)

功能描述:
在ESM-2隐空间中计算Wilson环，检测非平凡拓扑相位。
Wilson环定义为沿闭合路径的平行输运相位累积。
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.spatial import cKDTree
from scipy.linalg import svd
import warnings
warnings.filterwarnings('ignore')

# 创建输出目录
output_dir = Path('data/wilson_loops')
output_dir.mkdir(parents=True, exist_ok=True)

print("=" * 80)
print("Wilson环计算 - ESM-2隐空间拓扑相位检测")
print("=" * 80)

# 1. 加载数据
print("\n[1/6] 加载数据...")
pca_50d = np.load('data/pca/pca_embeddings_50d.npy')
umap_2d = np.load('data/umap/umap_embeddings_2d.npy')
metadata = pd.read_csv('data/metadata_final_with_en.csv')

print(f"  - PCA 50D嵌入: {pca_50d.shape}")
print(f"  - UMAP 2D坐标: {umap_2d.shape}")
print(f"  - 元数据: {len(metadata)}条序列")

# 2. 定义闭合路径
print("\n[2/6] 定义闭合路径...")

def create_circular_path(center, radius, n_points=100):
    """在UMAP 2D空间中创建圆形路径"""
    angles = np.linspace(0, 2*np.pi, n_points, endpoint=False)
    path = np.column_stack([
        center[0] + radius * np.cos(angles),
        center[1] + radius * np.sin(angles)
    ])
    return path

def find_nearest_points(path_2d, umap_2d, pca_50d, k=1):
    """找到路径上每个点最近的实际数据点"""
    tree = cKDTree(umap_2d)
    distances, indices = tree.query(path_2d, k=k)
    return indices, distances, pca_50d[indices]

# 从相边界候选点计算中心
pb_candidates = pd.read_csv('data/phase_boundaries/phase_boundary_candidates.csv')
pb_boundary_points = pb_candidates[pb_candidates['is_boundary'] == True]

if len(pb_boundary_points) > 0:
    pb_center = np.array([pb_boundary_points['umap1'].mean(),
                          pb_boundary_points['umap2'].mean()])
else:
    # 回退：使用所有候选点的质心
    pb_center = np.array([pb_candidates['umap1'].mean(),
                          pb_candidates['umap2'].mean()])

# 定义多个代表性路径
paths_config = [
    # 路径1: 围绕流形中心的大圆
    {'name': 'center_large', 'center': umap_2d.mean(axis=0), 'radius': 10.0},
    # 路径2: 围绕流形中心的小圆
    {'name': 'center_small', 'center': umap_2d.mean(axis=0), 'radius': 5.0},
    # 路径3: 围绕相边界区域（IDP区域）
    {'name': 'phase_boundary', 'center': pb_center, 'radius': 3.0},
    # 路径4: 围绕anchor高密度区域（修正：使用anchor中心而非extreme）
    {'name': 'high_density', 'center': umap_2d[metadata['category'] == 'anchor'].mean(axis=0), 'radius': 2.0},
    # 路径5: 围绕anchor类别中心
    {'name': 'anchor_center', 'center': umap_2d[metadata['category'] == 'anchor'].mean(axis=0), 'radius': 5.0},
]

print(f"  - 定义了{len(paths_config)}条闭合路径")
for cfg in paths_config:
    print(f"    * {cfg['name']}: 中心({cfg['center'][0]:.2f}, {cfg['center'][1]:.2f}), 半径{cfg['radius']:.1f}")

# 3. 计算Wilson环
print("\n[3/6] 计算Wilson环...")

def compute_parallel_transport(vectors, closed=True):
    """
    计算沿路径的平行输运

    平行输运条件: ∇_v V = 0 (向量V沿切向量v的协变导数为0)
    在离散情况下，通过Gram-Schmidt正交化实现平行输运

    返回: 累积相位 (Wilson环值)
    """
    n_points = len(vectors)
    phases = []

    # 初始向量
    V = vectors[0].copy()
    V = V / np.linalg.norm(V)  # 归一化

    for i in range(n_points):
        # 当前点和下一点
        curr = vectors[i]
        next_idx = (i + 1) % n_points if closed else min(i + 1, n_points - 1)
        next_vec = vectors[next_idx]

        # 切向量 (连接当前点和下一点)
        tangent = next_vec - curr
        tangent_norm = np.linalg.norm(tangent)

        if tangent_norm < 1e-10:
            phases.append(0.0)
            continue

        tangent = tangent / tangent_norm

        # 平行输运: 将V投影到垂直于切向量的超平面
        V_parallel = V - np.dot(V, tangent) * tangent
        V_parallel_norm = np.linalg.norm(V_parallel)

        if V_parallel_norm < 1e-10:
            phases.append(0.0)
            V = next_vec / np.linalg.norm(next_vec)
            continue

        V_parallel = V_parallel / V_parallel_norm

        # 计算相位变化 (通过内积)
        phase = np.arccos(np.clip(np.dot(V, V_parallel), -1.0, 1.0))
        phases.append(phase)

        # 更新V为平行输运后的向量
        V = V_parallel

    # Wilson环值 = 累积相位
    wilson_loop = np.sum(phases)

    return wilson_loop, phases

wilson_results = []
path_data = []

for cfg in paths_config:
    # 创建路径
    path_2d = create_circular_path(cfg['center'], cfg['radius'], n_points=100)

    # 找到最近的实际数据点
    indices, distances, path_50d = find_nearest_points(path_2d, umap_2d, pca_50d)

    # 计算Wilson环
    wilson_value, phases = compute_parallel_transport(path_50d, closed=True)

    # 统计信息
    avg_distance = distances.mean()
    max_distance = distances.max()

    # 路径上的类别分布
    path_categories = metadata.iloc[indices]['category'].value_counts()

    wilson_results.append({
        'path_name': cfg['name'],
        'center_x': cfg['center'][0],
        'center_y': cfg['center'][1],
        'radius': cfg['radius'],
        'wilson_loop': wilson_value,
        'avg_phase': np.mean(phases),
        'std_phase': np.std(phases),
        'max_phase': np.max(phases),
        'avg_distance_to_path': avg_distance,
        'max_distance_to_path': max_distance,
        'n_points': len(indices),
        'dominant_category': path_categories.index[0] if len(path_categories) > 0 else 'N/A',
    })

    path_data.append({
        'name': cfg['name'],
        'path_2d': path_2d,
        'path_50d': path_50d,
        'indices': indices,
        'phases': phases,
    })

    print(f"  - {cfg['name']}: Wilson环 = {wilson_value:.4f}, 平均相位 = {np.mean(phases):.4f}")

# 保存结果
df_wilson = pd.DataFrame(wilson_results)
df_wilson.to_csv(output_dir / 'wilson_loop_values.csv', index=False)
print(f"\n  ✓ Wilson环值已保存: {output_dir / 'wilson_loop_values.csv'}")

# 保存路径数据
np.save(output_dir / 'wilson_loop_paths.npy', path_data, allow_pickle=True)
print(f"  ✓ 路径数据已保存: {output_dir / 'wilson_loop_paths.npy'}")

# 4. 检测非平凡拓扑相位
print("\n[4/6] 检测非平凡拓扑相位...")

# 非平凡拓扑的判据:
# 1. Wilson环值显著偏离0 (相位累积)
# 2. 不同路径的Wilson环值差异显著
# 3. Wilson环值与路径半径的依赖关系

# 计算统计量
wilson_values = df_wilson['wilson_loop'].values
mean_wilson = wilson_values.mean()
std_wilson = wilson_values.std()
max_wilson = wilson_values.max()
min_wilson = wilson_values.min()

print(f"  - Wilson环统计:")
print(f"    * 平均值: {mean_wilson:.4f}")
print(f"    * 标准差: {std_wilson:.4f}")
print(f"    * 范围: [{min_wilson:.4f}, {max_wilson:.4f}]")

# 判断是否存在非平凡拓扑
is_nontrivial = (std_wilson > 0.1) or (abs(mean_wilson) > 0.5)
print(f"\n  - 拓扑相位判断: {'非平凡' if is_nontrivial else '平凡'}")

if is_nontrivial:
    print(f"    * 理由: Wilson环值存在显著变化 (std={std_wilson:.4f})")
else:
    print(f"    * 理由: Wilson环值接近0且变化小 (mean={mean_wilson:.4f}, std={std_wilson:.4f})")

# 5. 可视化
print("\n[5/6] 生成可视化...")

# 第一组图: 路径和Wilson环值
fig1 = plt.figure(figsize=(18, 6))

# 子图1: UMAP空间中的路径
ax1 = plt.subplot(1, 3, 1)
scatter = ax1.scatter(umap_2d[:, 0], umap_2d[:, 1],
                     c=metadata['category'].astype('category').cat.codes,
                     cmap='tab10', s=1, alpha=0.3)
for i, pdata in enumerate(path_data):
    ax1.plot(pdata['path_2d'][:, 0], pdata['path_2d'][:, 1],
            linewidth=2, label=pdata['name'])
ax1.set_xlabel('UMAP 1')
ax1.set_ylabel('UMAP 2')
ax1.set_title('闭合路径在UMAP空间中的位置')
ax1.legend(fontsize=8)
ax1.grid(True, alpha=0.3)

# 子图2: Wilson环值对比
ax2 = plt.subplot(1, 3, 2)
bars = ax2.bar(range(len(df_wilson)), df_wilson['wilson_loop'],
              color=plt.cm.viridis(np.linspace(0, 1, len(df_wilson))))
ax2.set_xticks(range(len(df_wilson)))
ax2.set_xticklabels(df_wilson['path_name'], rotation=45, ha='right')
ax2.set_ylabel('Wilson环值')
ax2.set_title('不同路径的Wilson环值')
ax2.axhline(y=0, color='red', linestyle='--', linewidth=1, alpha=0.5)
ax2.grid(True, alpha=0.3, axis='y')

# 子图3: Wilson环 vs 路径半径
ax3 = plt.subplot(1, 3, 3)
ax3.scatter(df_wilson['radius'], df_wilson['wilson_loop'],
           s=100, c=range(len(df_wilson)), cmap='viridis')
for i, row in df_wilson.iterrows():
    ax3.annotate(row['path_name'], (row['radius'], row['wilson_loop']),
                fontsize=8, ha='left', va='bottom')
ax3.set_xlabel('路径半径')
ax3.set_ylabel('Wilson环值')
ax3.set_title('Wilson环值与路径半径的关系')
ax3.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(output_dir / 'wilson_loop_visualization.png', dpi=300, bbox_inches='tight')
print(f"  ✓ 可视化已保存: {output_dir / 'wilson_loop_visualization.png'}")
plt.close()

# 第二组图: 每条路径的相位累积
fig2 = plt.figure(figsize=(20, 8))
for i, pdata in enumerate(path_data):
    ax = plt.subplot(2, 3, i + 1)
    phases = pdata['phases']
    cumulative_phase = np.cumsum(phases)
    ax.plot(cumulative_phase, linewidth=2, color=plt.cm.viridis(i / len(path_data)))
    ax.set_xlabel('路径点索引')
    ax.set_ylabel('累积相位')
    ax.set_title(f'{pdata["name"]}\nWilson环 = {cumulative_phase[-1]:.4f}')
    ax.grid(True, alpha=0.3)
    ax.axhline(y=cumulative_phase[-1], color='red', linestyle='--',
              linewidth=1, alpha=0.5, label=f'最终值: {cumulative_phase[-1]:.4f}')
    ax.legend(fontsize=8)

plt.tight_layout()
plt.savefig(output_dir / 'wilson_loop_phase_accumulation.png', dpi=300, bbox_inches='tight')
print(f"  ✓ 相位累积图已保存: {output_dir / 'wilson_loop_phase_accumulation.png'}")
plt.close()

# 6. 生成分析报告
print("\n[6/6] 生成分析报告...")

report = f"""
{'='*80}
Wilson环计算 - 分析报告
{'='*80}

1. 数据概览
-----------
- 数据点数: {len(pca_50d)}
- 嵌入维度: 50D (PCA降维)
- 路径数量: {len(paths_config)}
- 每条路径点数: 100

2. Wilson环结果
---------------
"""

for _, row in df_wilson.iterrows():
    report += f"""
路径: {row['path_name']}
  - 中心坐标: ({row['center_x']:.2f}, {row['center_y']:.2f})
  - 半径: {row['radius']:.1f}
  - Wilson环值: {row['wilson_loop']:.4f}
  - 平均相位: {row['avg_phase']:.4f} ± {row['std_phase']:.4f}
  - 最大相位: {row['max_phase']:.4f}
  - 主要类别: {row['dominant_category']}
  - 平均偏离距离: {row['avg_distance_to_path']:.4f}
"""

report += f"""
3. 统计摘要
-----------
- Wilson环平均值: {mean_wilson:.4f}
- Wilson环标准差: {std_wilson:.4f}
- Wilson环范围: [{min_wilson:.4f}, {max_wilson:.4f}]

4. 拓扑相位判断
---------------
- 结论: {'非平凡拓扑相位' if is_nontrivial else '平凡拓扑相位'}
- 判据:
  * Wilson环标准差 > 0.1: {std_wilson > 0.1} (实际值: {std_wilson:.4f})
  * |平均Wilson环| > 0.5: {abs(mean_wilson) > 0.5} (实际值: {abs(mean_wilson):.4f})

5. 物理解释
-----------
Wilson环是规范场论中的基本观测量，用于检测非平凡拓扑相位。
在ESM-2隐空间中:
- Wilson环 ≈ 0: 平凡拓扑，无规范场结构
- Wilson环 ≠ 0: 非平凡拓扑，可能存在涌现的规范对称性

本分析中的Wilson环值{'显示' if is_nontrivial else '未显示'}显著的非平凡拓扑特征。

6. 关键发现
-----------
"""

# 找到最大和最小Wilson环值的路径
max_idx = df_wilson['wilson_loop'].idxmax()
min_idx = df_wilson['wilson_loop'].idxmin()

report += f"""
- 最大Wilson环: {df_wilson.loc[max_idx, 'path_name']} = {df_wilson.loc[max_idx, 'wilson_loop']:.4f}
- 最小Wilson环: {df_wilson.loc[min_idx, 'path_name']} = {df_wilson.loc[min_idx, 'wilson_loop']:.4f}
- Wilson环变化范围: {max_wilson - min_wilson:.4f}

7. 与物理指标的关联
-------------------
"""

# 检查路径上的物理指标分布
if 'E_n' in metadata.columns:
    for pdata in path_data:
        path_metadata = metadata.iloc[pdata['indices']]
        en_values = path_metadata['E_n'].dropna()
        if len(en_values) > 0:
            report += f"\n路径 {pdata['name']}:"
            report += f"\n  - E[n]均值: {en_values.mean():.4f} ± {en_values.std():.4f}"
            report += f"\n  - E[n]范围: [{en_values.min():.4f}, {en_values.max():.4f}]"

report += f"""

{'='*80}
报告生成时间: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}
{'='*80}
"""

with open(output_dir / 'wilson_loop_report.txt', 'w') as f:
    f.write(report)

print(f"  ✓ 分析报告已保存: {output_dir / 'wilson_loop_report.txt'}")

print("\n" + "="*80)
print("Wilson环计算完成!")
print("="*80)
print(f"\n核心结果:")
print(f"  - Wilson环平均值: {mean_wilson:.4f}")
print(f"  - Wilson环标准差: {std_wilson:.4f}")
print(f"  - 拓扑相位: {'非平凡' if is_nontrivial else '平凡'}")
print(f"\n输出文件:")
print(f"  - {output_dir / 'wilson_loop_values.csv'}")
print(f"  - {output_dir / 'wilson_loop_paths.npy'}")
print(f"  - {output_dir / 'wilson_loop_visualization.png'}")
print(f"  - {output_dir / 'wilson_loop_report.txt'}")
