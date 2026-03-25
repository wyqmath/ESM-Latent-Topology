#!/usr/bin/env python3
# -- Project root setup --
import os as _os
from pathlib import Path as _Path
_os.chdir(_Path(__file__).resolve().parents[2])
"""
持续同调计算 (Persistent Homology Analysis)

输入文件:
- data/pca/pca_embeddings_50d.npy (11068×50 PCA降维结果)
- data/metadata_final_with_en.csv (序列元数据)

输出文件:
- data/persistent_homology/persistence_diagrams.pkl (持续图数据)
- data/persistent_homology/betti_numbers.json (Betti数统计)
- data/persistent_homology/persistence_diagram_H0.png (β0持续图)
- data/persistent_homology/persistence_diagram_H1.png (β1持续图)
- data/persistent_homology/persistence_diagram_H2.png (β2持续图)
- data/persistent_homology/barcodes.png (条形码图)
- data/persistent_homology/ph_report.txt (分析报告)

功能描述:
使用Ripser计算持续同调，提取拓扑特征(连通分支、环、空腔)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from ripser import ripser
from persim import plot_diagrams
import json
import pickle
from pathlib import Path
import time

# 设置随机种子
np.random.seed(42)

# 创建输出目录
output_dir = Path("data/persistent_homology")
output_dir.mkdir(parents=True, exist_ok=True)

print("=" * 80)
print("持续同调计算 (Persistent Homology Analysis)")
print("=" * 80)

# 1. 加载数据
print("\n[1/6] 加载数据...")
pca_50d = np.load("data/pca/pca_embeddings_50d.npy")
metadata = pd.read_csv("data/metadata_final_with_en.csv")

print(f"  - PCA 50D嵌入: {pca_50d.shape}")
print(f"  - 元数据: {metadata.shape}")

# 2. 下采样到1500点 (推荐配置，平衡计算时间和拓扑精度)
print("\n[2/6] 下采样数据...")
n_samples = min(1500, len(pca_50d))
sample_indices = np.random.choice(len(pca_50d), size=n_samples, replace=False)
pca_sampled = pca_50d[sample_indices]
metadata_sampled = metadata.iloc[sample_indices].reset_index(drop=True)

print(f"  - 采样点数: {n_samples}")
print(f"  - 采样后形状: {pca_sampled.shape}")

# 统计采样后的类别分布
category_counts = metadata_sampled['category'].value_counts()
print(f"  - 类别分布:")
for cat, count in category_counts.items():
    print(f"    {cat}: {count} ({count/n_samples*100:.1f}%)")

# 3. 计算持续同调
print("\n[3/6] 计算持续同调 (maxdim=2)...")
print("  - 这可能需要10-30分钟，请耐心等待...")
start_time = time.time()

# 使用Ripser计算持续同调 (maxdim=2: 计算H0, H1, H2)
result = ripser(pca_sampled, maxdim=2, thresh=np.inf)
diagrams = result['dgms']

elapsed_time = time.time() - start_time
print(f"  ✓ 计算完成，耗时: {elapsed_time:.2f}秒 ({elapsed_time/60:.2f}分钟)")

# 4. 提取Betti数
print("\n[4/6] 提取Betti数...")

def count_features_at_scale(diagram, scale):
    """计算给定尺度下的拓扑特征数量"""
    if len(diagram) == 0:
        return 0
    birth = diagram[:, 0]
    death = diagram[:, 1]
    # 特征在scale时存在: birth <= scale < death
    alive = (birth <= scale) & (death > scale)
    return np.sum(alive)

# 选择多个尺度计算Betti数
scales = [0.5, 1.0, 2.0, 5.0, 10.0]
betti_numbers = {}

for scale in scales:
    beta0 = count_features_at_scale(diagrams[0], scale)
    beta1 = count_features_at_scale(diagrams[1], scale) if len(diagrams) > 1 else 0
    beta2 = count_features_at_scale(diagrams[2], scale) if len(diagrams) > 2 else 0
    betti_numbers[f"scale_{scale}"] = {
        "beta0": int(beta0),
        "beta1": int(beta1),
        "beta2": int(beta2)
    }
    print(f"  - 尺度 {scale:.1f}: β₀={beta0}, β₁={beta1}, β₂={beta2}")

# 计算持续性最长的特征
def get_top_persistent_features(diagram, top_k=10):
    """获取持续性最长的前k个特征"""
    if len(diagram) == 0:
        return []
    persistence = diagram[:, 1] - diagram[:, 0]
    # 过滤掉无穷大的点
    finite_mask = np.isfinite(persistence)
    if not np.any(finite_mask):
        return []
    top_indices = np.argsort(persistence[finite_mask])[-top_k:][::-1]
    return diagram[finite_mask][top_indices].tolist()

top_features = {
    "H0": get_top_persistent_features(diagrams[0], top_k=10),
    "H1": get_top_persistent_features(diagrams[1], top_k=10) if len(diagrams) > 1 else [],
    "H2": get_top_persistent_features(diagrams[2], top_k=10) if len(diagrams) > 2 else []
}

print(f"\n  - 持续性最长的特征:")
print(f"    H0 (连通分支): {len(top_features['H0'])}个")
print(f"    H1 (环): {len(top_features['H1'])}个")
print(f"    H2 (空腔): {len(top_features['H2'])}个")

# 5. 保存结果
print("\n[5/6] 保存结果...")

# 保存持续图数据
with open(output_dir / "persistence_diagrams.pkl", "wb") as f:
    pickle.dump({
        "diagrams": diagrams,
        "sample_indices": sample_indices,
        "n_samples": n_samples
    }, f)
print(f"  ✓ 持续图数据: {output_dir / 'persistence_diagrams.pkl'}")

# 保存Betti数
betti_output = {
    "betti_numbers_by_scale": betti_numbers,
    "top_persistent_features": top_features,
    "computation_time_seconds": elapsed_time,
    "n_samples": n_samples
}
with open(output_dir / "betti_numbers.json", "w") as f:
    json.dump(betti_output, f, indent=2)
print(f"  ✓ Betti数统计: {output_dir / 'betti_numbers.json'}")

# 6. 可视化
print("\n[6/6] 生成可视化...")

# 6.1 持续图 (Persistence Diagrams)
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

for i, (ax, title) in enumerate(zip(axes, ['H0 (连通分支)', 'H1 (环)', 'H2 (空腔)'])):
    if i < len(diagrams) and len(diagrams[i]) > 0:
        diagram = diagrams[i]
        # 过滤掉无穷大的点
        finite_mask = np.isfinite(diagram[:, 1])
        diagram_finite = diagram[finite_mask]

        if len(diagram_finite) > 0:
            ax.scatter(diagram_finite[:, 0], diagram_finite[:, 1],
                      alpha=0.6, s=20, c='blue')
            # 绘制对角线
            max_val = max(diagram_finite[:, 1].max(), diagram_finite[:, 0].max())
            ax.plot([0, max_val], [0, max_val], 'k--', alpha=0.3, linewidth=1)
            ax.set_xlabel('Birth', fontsize=12)
            ax.set_ylabel('Death', fontsize=12)
            ax.set_title(title, fontsize=14, fontweight='bold')
            ax.grid(True, alpha=0.3)
            ax.set_aspect('equal')
        else:
            ax.text(0.5, 0.5, 'No finite features',
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title(title, fontsize=14, fontweight='bold')
    else:
        ax.text(0.5, 0.5, 'No features',
               ha='center', va='center', transform=ax.transAxes)
        ax.set_title(title, fontsize=14, fontweight='bold')

plt.tight_layout()
plt.savefig(output_dir / "persistence_diagrams.png", dpi=300, bbox_inches='tight')
plt.close()
print(f"  ✓ 持续图: {output_dir / 'persistence_diagrams.png'}")

# 6.2 条形码图 (Barcodes)
fig, axes = plt.subplots(3, 1, figsize=(12, 10))

for i, (ax, title) in enumerate(zip(axes, ['H0 (连通分支)', 'H1 (环)', 'H2 (空腔)'])):
    if i < len(diagrams) and len(diagrams[i]) > 0:
        diagram = diagrams[i]
        # 过滤掉无穷大的点
        finite_mask = np.isfinite(diagram[:, 1])
        diagram_finite = diagram[finite_mask]

        if len(diagram_finite) > 0:
            # 按持续性排序
            persistence = diagram_finite[:, 1] - diagram_finite[:, 0]
            sorted_indices = np.argsort(persistence)[::-1]
            diagram_sorted = diagram_finite[sorted_indices]

            # 只显示前100个最持久的特征
            n_show = min(100, len(diagram_sorted))
            for j in range(n_show):
                birth, death = diagram_sorted[j]
                ax.plot([birth, death], [j, j], 'b-', linewidth=2, alpha=0.6)

            ax.set_xlabel('Scale', fontsize=12)
            ax.set_ylabel('Feature Index', fontsize=12)
            ax.set_title(title, fontsize=14, fontweight='bold')
            ax.grid(True, alpha=0.3, axis='x')
            ax.set_ylim(-1, n_show)
        else:
            ax.text(0.5, 0.5, 'No finite features',
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title(title, fontsize=14, fontweight='bold')
    else:
        ax.text(0.5, 0.5, 'No features',
               ha='center', va='center', transform=ax.transAxes)
        ax.set_title(title, fontsize=14, fontweight='bold')

plt.tight_layout()
plt.savefig(output_dir / "barcodes.png", dpi=300, bbox_inches='tight')
plt.close()
print(f"  ✓ 条形码图: {output_dir / 'barcodes.png'}")

# 6.3 Betti数随尺度变化
scales_fine = np.linspace(0, 10, 100)
betti_curves = {
    'beta0': [],
    'beta1': [],
    'beta2': []
}

for scale in scales_fine:
    betti_curves['beta0'].append(count_features_at_scale(diagrams[0], scale))
    betti_curves['beta1'].append(count_features_at_scale(diagrams[1], scale) if len(diagrams) > 1 else 0)
    betti_curves['beta2'].append(count_features_at_scale(diagrams[2], scale) if len(diagrams) > 2 else 0)

fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(scales_fine, betti_curves['beta0'], label='β₀ (连通分支)', linewidth=2)
ax.plot(scales_fine, betti_curves['beta1'], label='β₁ (环)', linewidth=2)
ax.plot(scales_fine, betti_curves['beta2'], label='β₂ (空腔)', linewidth=2)
ax.set_xlabel('Scale', fontsize=12)
ax.set_ylabel('Betti Number', fontsize=12)
ax.set_title('Betti数随尺度变化', fontsize=14, fontweight='bold')
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(output_dir / "betti_curves.png", dpi=300, bbox_inches='tight')
plt.close()
print(f"  ✓ Betti曲线: {output_dir / 'betti_curves.png'}")

# 7. 生成分析报告
print("\n[7/7] 生成分析报告...")

report_lines = [
    "=" * 80,
    "持续同调分析报告 (Persistent Homology Analysis Report)",
    "=" * 80,
    "",
    "1. 数据概览",
    "-" * 80,
    f"  - 原始数据点数: {len(pca_50d)}",
    f"  - 采样点数: {n_samples}",
    f"  - 特征维度: {pca_50d.shape[1]}D (PCA降维后)",
    f"  - 计算时间: {elapsed_time:.2f}秒 ({elapsed_time/60:.2f}分钟)",
    "",
    "2. 采样后类别分布",
    "-" * 80
]

for cat, count in category_counts.items():
    report_lines.append(f"  - {cat}: {count} ({count/n_samples*100:.1f}%)")

report_lines.extend([
    "",
    "3. Betti数统计 (不同尺度)",
    "-" * 80
])

for scale_key, betti in betti_numbers.items():
    scale = float(scale_key.split('_')[1])
    report_lines.append(
        f"  - 尺度 {scale:.1f}: β₀={betti['beta0']}, β₁={betti['beta1']}, β₂={betti['beta2']}"
    )

report_lines.extend([
    "",
    "4. 持续性最长的特征",
    "-" * 80,
    f"  - H0 (连通分支): {len(top_features['H0'])}个显著特征",
    f"  - H1 (环): {len(top_features['H1'])}个显著特征",
    f"  - H2 (空腔): {len(top_features['H2'])}个显著特征",
    "",
    "5. 拓扑解释",
    "-" * 80,
    "  - β₀ (H0): 连通分支数量，反映流形的分离程度",
    "  - β₁ (H1): 环的数量，反映流形的孔洞结构",
    "  - β₂ (H2): 空腔的数量，反映流形的三维拓扑复杂性",
    "",
    "6. 输出文件",
    "-" * 80,
    f"  - 持续图数据: {output_dir / 'persistence_diagrams.pkl'}",
    f"  - Betti数统计: {output_dir / 'betti_numbers.json'}",
    f"  - 持续图可视化: {output_dir / 'persistence_diagrams.png'}",
    f"  - 条形码图: {output_dir / 'barcodes.png'}",
    f"  - Betti曲线: {output_dir / 'betti_curves.png'}",
    "",
    "=" * 80
])

report_text = "\n".join(report_lines)
print(report_text)

with open(output_dir / "ph_report.txt", "w") as f:
    f.write(report_text)
print(f"\n  ✓ 分析报告: {output_dir / 'ph_report.txt'}")

print("\n" + "=" * 80)
print("持续同调计算完成!")
print("=" * 80)
