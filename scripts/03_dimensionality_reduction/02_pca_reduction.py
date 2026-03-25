#!/usr/bin/env python3
# -- Project root setup --
import os as _os
from pathlib import Path as _Path
_os.chdir(_Path(__file__).resolve().parents[2])
"""
PCA降维分析脚本

输入文件:
- data/embeddings/sequence_embeddings_final.pt (11068×2560 FP16嵌入)
- data/metadata_final_with_en.csv (序列元数据)

输出文件:
- data/pca/pca_embeddings_2d.npy (11068×2降维结果)
- data/pca/pca_embeddings_50d.npy (11068×50降维结果，用于几何计算)
- data/pca/pca_model.pkl (PCA模型)
- data/pca/explained_variance.npy (解释方差比)
- data/pca/pca_visualization.png (静态图)
- data/pca/pca_visualization.html (交互式图)
- data/pca/variance_explained_plot.png (方差解释图)

功能描述:
使用PCA进行线性降维，计算前50个主成分用于几何分析，
前2个主成分用于可视化，分析解释方差比。
"""

import torch
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path
import pickle
import time

# 设置随机种子
np.random.seed(42)

# 路径配置
DATA_DIR = Path("data")
EMBEDDING_FILE = DATA_DIR / "embeddings" / "sequence_embeddings_final.pt"
METADATA_FILE = DATA_DIR / "metadata_final_with_en.csv"
OUTPUT_DIR = DATA_DIR / "pca"
OUTPUT_DIR.mkdir(exist_ok=True)

print("=" * 80)
print("PCA降维分析")
print("=" * 80)

# 1. 加载嵌入和元数据
print("\n[1/6] 加载数据...")
embeddings = torch.load(EMBEDDING_FILE).numpy().astype(np.float32)
metadata = pd.read_csv(METADATA_FILE)

print(f"  嵌入形状: {embeddings.shape}")
print(f"  元数据行数: {len(metadata)}")
assert embeddings.shape[0] == len(metadata), "嵌入和元数据数量不匹配"

# 2. 标准化数据
print("\n[2/6] 标准化数据...")
scaler = StandardScaler()
embeddings_scaled = scaler.fit_transform(embeddings)
print(f"  标准化后均值: {embeddings_scaled.mean():.6f}")
print(f"  标准化后标准差: {embeddings_scaled.std():.6f}")

# 3. PCA降维到50维
print("\n[3/6] PCA降维到50维...")
start_time = time.time()

pca_50d = PCA(n_components=50, random_state=42)
embeddings_50d = pca_50d.fit_transform(embeddings_scaled)

elapsed_50d = time.time() - start_time
print(f"  降维时间: {elapsed_50d:.2f}秒 ({elapsed_50d/len(embeddings)*1000:.2f}毫秒/序列)")
print(f"  输出形状: {embeddings_50d.shape}")
print(f"  累积解释方差比: {pca_50d.explained_variance_ratio_.sum():.4f}")
print(f"  前10个主成分解释方差比: {pca_50d.explained_variance_ratio_[:10].sum():.4f}")

# 保存50维结果
np.save(OUTPUT_DIR / "pca_embeddings_50d.npy", embeddings_50d)

# 4. PCA降维到2维
print("\n[4/6] PCA降维到2维...")
start_time = time.time()

pca_2d = PCA(n_components=2, random_state=42)
embeddings_2d = pca_2d.fit_transform(embeddings_scaled)

elapsed_2d = time.time() - start_time
print(f"  降维时间: {elapsed_2d:.2f}秒 ({elapsed_2d/len(embeddings)*1000:.2f}毫秒/序列)")
print(f"  输出形状: {embeddings_2d.shape}")
print(f"  PC1解释方差比: {pca_2d.explained_variance_ratio_[0]:.4f}")
print(f"  PC2解释方差比: {pca_2d.explained_variance_ratio_[1]:.4f}")
print(f"  累积解释方差比: {pca_2d.explained_variance_ratio_.sum():.4f}")
print(f"  坐标范围: X[{embeddings_2d[:, 0].min():.2f}, {embeddings_2d[:, 0].max():.2f}], "
      f"Y[{embeddings_2d[:, 1].min():.2f}, {embeddings_2d[:, 1].max():.2f}]")

# 保存2维结果
np.save(OUTPUT_DIR / "pca_embeddings_2d.npy", embeddings_2d)

# 5. 保存模型和方差数据
print("\n[5/6] 保存模型和方差数据...")
with open(OUTPUT_DIR / "pca_model.pkl", 'wb') as f:
    pickle.dump({'pca_50d': pca_50d, 'pca_2d': pca_2d, 'scaler': scaler}, f)

np.save(OUTPUT_DIR / "explained_variance.npy", pca_50d.explained_variance_ratio_)

# 6. 生成可视化
print("\n[6/6] 生成可视化...")

# 6.1 PCA 2D散点图
fig, ax = plt.subplots(figsize=(12, 10))

categories = metadata['category'].unique()
colors = plt.cm.tab10(np.linspace(0, 1, len(categories)))

for i, cat in enumerate(categories):
    mask = metadata['category'] == cat
    ax.scatter(
        embeddings_2d[mask, 0],
        embeddings_2d[mask, 1],
        c=[colors[i]],
        label=cat,
        alpha=0.6,
        s=10
    )

ax.set_xlabel(f'PC1 ({pca_2d.explained_variance_ratio_[0]:.2%} variance)', fontsize=12)
ax.set_ylabel(f'PC2 ({pca_2d.explained_variance_ratio_[1]:.2%} variance)', fontsize=12)
ax.set_title(f'PCA Dimensionality Reduction (Total: {pca_2d.explained_variance_ratio_.sum():.2%} variance)',
             fontsize=14, fontweight='bold')
ax.legend(loc='best', fontsize=10)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "pca_visualization.png", dpi=150, bbox_inches='tight')
plt.close()

# 6.2 交互式图
fig = px.scatter(
    x=embeddings_2d[:, 0],
    y=embeddings_2d[:, 1],
    color=metadata['category'],
    hover_data={
        'seq_id': metadata['seq_id'],
        'length': metadata['length'],
        'category': metadata['category']
    },
    labels={
        'x': f'PC1 ({pca_2d.explained_variance_ratio_[0]:.2%} variance)',
        'y': f'PC2 ({pca_2d.explained_variance_ratio_[1]:.2%} variance)'
    },
    title='PCA Dimensionality Reduction',
    width=1200,
    height=900
)

fig.update_traces(marker=dict(size=5, opacity=0.6))
fig.write_html(OUTPUT_DIR / "pca_visualization.html")

# 6.3 解释方差图
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# 左图: 前50个主成分的解释方差比
ax1 = axes[0]
ax1.bar(range(1, 51), pca_50d.explained_variance_ratio_, alpha=0.7, color='steelblue')
ax1.set_xlabel('Principal Component', fontsize=12)
ax1.set_ylabel('Explained Variance Ratio', fontsize=12)
ax1.set_title('Explained Variance by Component (Top 50)', fontsize=13, fontweight='bold')
ax1.grid(True, alpha=0.3, axis='y')

# 右图: 累积解释方差比
ax2 = axes[1]
cumsum = np.cumsum(pca_50d.explained_variance_ratio_)
ax2.plot(range(1, 51), cumsum, marker='o', markersize=4, linewidth=2, color='darkred')
ax2.axhline(y=0.9, color='green', linestyle='--', linewidth=1.5, label='90% threshold')
ax2.axhline(y=0.95, color='orange', linestyle='--', linewidth=1.5, label='95% threshold')
ax2.set_xlabel('Number of Components', fontsize=12)
ax2.set_ylabel('Cumulative Explained Variance', fontsize=12)
ax2.set_title('Cumulative Explained Variance', fontsize=13, fontweight='bold')
ax2.legend(fontsize=10)
ax2.grid(True, alpha=0.3)

# 标注关键点
n_90 = np.argmax(cumsum >= 0.9) + 1
n_95 = np.argmax(cumsum >= 0.95) + 1
ax2.scatter([n_90, n_95], [cumsum[n_90-1], cumsum[n_95-1]],
            s=100, c=['green', 'orange'], zorder=5, edgecolors='black')
ax2.text(n_90, cumsum[n_90-1] - 0.03, f'n={n_90}', ha='center', fontsize=10, fontweight='bold')
ax2.text(n_95, cumsum[n_95-1] + 0.02, f'n={n_95}', ha='center', fontsize=10, fontweight='bold')

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "variance_explained_plot.png", dpi=150, bbox_inches='tight')
plt.close()

# 7. 总结报告
print("\n" + "=" * 80)
print("PCA降维完成!")
print("=" * 80)
print(f"\n输出文件:")
print(f"  - 2D降维结果: {OUTPUT_DIR}/pca_embeddings_2d.npy")
print(f"  - 50D降维结果: {OUTPUT_DIR}/pca_embeddings_50d.npy")
print(f"  - PCA模型: {OUTPUT_DIR}/pca_model.pkl")
print(f"  - 解释方差: {OUTPUT_DIR}/explained_variance.npy")
print(f"  - 静态图: {OUTPUT_DIR}/pca_visualization.png")
print(f"  - 交互式图: {OUTPUT_DIR}/pca_visualization.html")
print(f"  - 方差解释图: {OUTPUT_DIR}/variance_explained_plot.png")

print(f"\n关键统计:")
print(f"  - PC1解释方差: {pca_2d.explained_variance_ratio_[0]:.4f} ({pca_2d.explained_variance_ratio_[0]*100:.2f}%)")
print(f"  - PC2解释方差: {pca_2d.explained_variance_ratio_[1]:.4f} ({pca_2d.explained_variance_ratio_[1]*100:.2f}%)")
print(f"  - 前2个PC累积方差: {pca_2d.explained_variance_ratio_.sum():.4f} ({pca_2d.explained_variance_ratio_.sum()*100:.2f}%)")
print(f"  - 前50个PC累积方差: {pca_50d.explained_variance_ratio_.sum():.4f} ({pca_50d.explained_variance_ratio_.sum()*100:.2f}%)")
print(f"  - 达到90%方差需要: {n_90}个主成分")
print(f"  - 达到95%方差需要: {n_95}个主成分")

print("\n下一步建议:")
print("  1. 使用50维PCA结果进行几何指标计算(避免维度灾难)")
print("  2. 对比PCA、UMAP、t-SNE三种降维方法的聚类效果")
print("  3. 继续可视化散点图 (Layer 3.1.4)")
