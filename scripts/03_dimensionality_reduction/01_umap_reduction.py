#!/usr/bin/env python3
# -- Project root setup --
import os as _os
from pathlib import Path as _Path
_os.chdir(_Path(__file__).resolve().parents[2])
"""
UMAP降维分析

输入文件:
- data/embeddings/sequence_embeddings_final.pt (11068×2560 FP16)
- data/metadata_final_with_en.csv (11068行元数据)

输出文件:
- data/umap/umap_embeddings_2d.npy (11068×2降维结果)
- data/umap/umap_params.json (UMAP参数)
- data/umap/umap_visualization.png (可视化图表)
- data/umap/umap_visualization.html (交互式图表)

功能描述:
使用UMAP将11068×2560的ESM-2嵌入降维到2D空间，用于可视化和后续分析
"""

import numpy as np
import pandas as pd
import torch
import umap
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import json
from pathlib import Path
import time

# 设置路径
DATA_DIR = Path("data")
EMBEDDING_DIR = DATA_DIR / "embeddings"
UMAP_DIR = DATA_DIR / "umap"
UMAP_DIR.mkdir(exist_ok=True)

# 文件路径
EMBEDDING_FILE = EMBEDDING_DIR / "sequence_embeddings_final.pt"
METADATA_FILE = DATA_DIR / "metadata_final_with_en.csv"
UMAP_2D_FILE = UMAP_DIR / "umap_embeddings_2d.npy"
UMAP_PARAMS_FILE = UMAP_DIR / "umap_params.json"
UMAP_VIS_FILE = UMAP_DIR / "umap_visualization.png"
UMAP_HTML_FILE = UMAP_DIR / "umap_visualization.html"

print("=" * 80)
print("UMAP降维分析")
print("=" * 80)

# 1. 加载嵌入数据
print("\n[1/5] 加载嵌入数据...")
embeddings = torch.load(EMBEDDING_FILE, map_location='cpu', weights_only=True)
print(f"  嵌入形状: {embeddings.shape}")
print(f"  数据类型: {embeddings.dtype}")

# 转换为FP32 numpy数组（UMAP需要FP32）
embeddings_np = embeddings.float().numpy()
print(f"  转换为numpy: {embeddings_np.shape}, {embeddings_np.dtype}")

# 2. 加载元数据
print("\n[2/5] 加载元数据...")
metadata = pd.read_csv(METADATA_FILE)
print(f"  元数据行数: {len(metadata)}")
print(f"  类别分布:")
category_counts = metadata['category'].value_counts()
for cat, count in category_counts.items():
    print(f"    {cat}: {count}")

# 3. UMAP降维
print("\n[3/5] 执行UMAP降维...")
umap_params = {
    'n_neighbors': 15,
    'min_dist': 0.1,
    'n_components': 2,
    'metric': 'cosine',
    'random_state': 42,
    'n_jobs': -1,
    'verbose': True
}

print(f"  UMAP参数: {umap_params}")
start_time = time.time()

reducer = umap.UMAP(**umap_params)
umap_2d = reducer.fit_transform(embeddings_np)

elapsed_time = time.time() - start_time
print(f"  降维完成，耗时: {elapsed_time:.2f}秒")
print(f"  降维结果形状: {umap_2d.shape}")
print(f"  X范围: [{umap_2d[:, 0].min():.2f}, {umap_2d[:, 0].max():.2f}]")
print(f"  Y范围: [{umap_2d[:, 1].min():.2f}, {umap_2d[:, 1].max():.2f}]")

# 4. 保存结果
print("\n[4/5] 保存降维结果...")
np.save(UMAP_2D_FILE, umap_2d)
print(f"  ✓ 保存降维结果: {UMAP_2D_FILE}")

# 保存参数（包含运行时间）
umap_params['elapsed_time_seconds'] = elapsed_time
with open(UMAP_PARAMS_FILE, 'w') as f:
    json.dump(umap_params, f, indent=2)
print(f"  ✓ 保存UMAP参数: {UMAP_PARAMS_FILE}")

# 5. 可视化
print("\n[5/5] 生成可视化...")

# 准备数据
df_vis = pd.DataFrame({
    'UMAP1': umap_2d[:, 0],
    'UMAP2': umap_2d[:, 1],
    'category': metadata['category'],
    'seq_id': metadata['seq_id'],
    'length': metadata['length']
})

# 定义颜色映射
color_map = {
    'anchor': '#1f77b4',      # 蓝色
    'astral95': '#ff7f0e',    # 橙色
    'control': '#2ca02c',     # 绿色
    'extreme': '#d62728',     # 红色
    'integrable': '#9467bd'   # 紫色
}

# 5.1 Matplotlib静态图
fig, ax = plt.subplots(figsize=(12, 10))

for cat in df_vis['category'].unique():
    mask = df_vis['category'] == cat
    ax.scatter(
        df_vis.loc[mask, 'UMAP1'],
        df_vis.loc[mask, 'UMAP2'],
        c=color_map.get(cat, '#7f7f7f'),
        label=f"{cat} (n={mask.sum()})",
        alpha=0.6,
        s=20,
        edgecolors='none'
    )

ax.set_xlabel('UMAP1', fontsize=12)
ax.set_ylabel('UMAP2', fontsize=12)
ax.set_title('UMAP 2D Projection of ESM-2 Embeddings (11068 sequences)', fontsize=14, fontweight='bold')
ax.legend(loc='best', framealpha=0.9)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(UMAP_VIS_FILE, dpi=300, bbox_inches='tight')
print(f"  ✓ 保存静态图: {UMAP_VIS_FILE}")
plt.close()

# 5.2 Plotly交互式图
fig_interactive = px.scatter(
    df_vis,
    x='UMAP1',
    y='UMAP2',
    color='category',
    hover_data=['seq_id', 'length'],
    title='UMAP 2D Projection of ESM-2 Embeddings (Interactive)',
    color_discrete_map=color_map,
    opacity=0.7
)

fig_interactive.update_traces(marker=dict(size=5))
fig_interactive.update_layout(
    width=1200,
    height=900,
    font=dict(size=12),
    hovermode='closest'
)

fig_interactive.write_html(UMAP_HTML_FILE)
print(f"  ✓ 保存交互式图: {UMAP_HTML_FILE}")

# 6. 统计分析
print("\n" + "=" * 80)
print("降维结果统计")
print("=" * 80)

print(f"\n数据集大小: {len(umap_2d)} 条序列")
print(f"降维维度: {embeddings_np.shape[1]} → {umap_2d.shape[1]}")
print(f"计算时间: {elapsed_time:.2f} 秒")
print(f"平均每序列: {elapsed_time / len(umap_2d) * 1000:.2f} 毫秒")

print("\n类别分布:")
for cat, count in category_counts.items():
    print(f"  {cat}: {count} ({count / len(metadata) * 100:.1f}%)")

print("\nUMAP坐标统计:")
print(f"  UMAP1: mean={umap_2d[:, 0].mean():.2f}, std={umap_2d[:, 0].std():.2f}")
print(f"  UMAP2: mean={umap_2d[:, 1].mean():.2f}, std={umap_2d[:, 1].std():.2f}")

print("\n" + "=" * 80)
print("✓ UMAP降维完成")
print("=" * 80)
