#!/usr/bin/env python3
# -- Project root setup --
import os as _os
from pathlib import Path as _Path
_os.chdir(_Path(__file__).resolve().parents[2])
"""
Supplementary Figure S7: Density Landscape and Entropy Association

Panel layout (1x3):
- (a) UMAP colored by log density
- (b) Smoothed density contour map over UMAP plane
- (c) E[n] vs log density, colored by category with Spearman rho

Input files:
- data/density/density_values.npy
- data/umap/umap_embeddings_2d.npy
- data/metadata_final_with_en.csv
- data/embeddings/embedding_index_final.csv

Output: manuscript/supp7.png
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.interpolate import griddata
from scipy.stats import spearmanr
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

# Style
plt.style.use('seaborn-v0_8-paper')
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 10

print("=" * 70)
print("Supplementary Figure S7: Density Landscape and Entropy Association")
print("=" * 70)

# Load core arrays
print("\n[1/4] Loading density and UMAP data...")
density = np.load('data/density/density_values.npy')
umap_coords = np.load('data/umap/umap_embeddings_2d.npy')
log_density = np.log10(density)

print(f"  Total samples: {len(density)}")
print(f"  Density shape: {density.shape}")
print(f"  UMAP shape: {umap_coords.shape}")

if density.shape[0] != umap_coords.shape[0]:
    raise ValueError(f"Shape mismatch: density={density.shape[0]}, umap={umap_coords.shape[0]}")

# Load metadata and index mapping for panel (c)
print("\n[2/4] Loading metadata alignment for panel (c)...")
metadata = pd.read_csv('data/metadata_final_with_en.csv')
embedding_index = pd.read_csv('data/embeddings/embedding_index_final.csv')

if 'seq_id' not in metadata.columns or 'E_n' not in metadata.columns or 'category' not in metadata.columns:
    raise KeyError("metadata_final_with_en.csv must contain seq_id, category, and E_n columns")
if 'seq_id' not in embedding_index.columns or 'index' not in embedding_index.columns:
    raise KeyError("embedding_index_final.csv must contain seq_id and index columns")

merged = embedding_index.merge(
    metadata[['seq_id', 'category', 'E_n']],
    on='seq_id',
    how='left',
    sort=False,
)

indices = pd.to_numeric(merged['index'], errors='coerce')
valid_index_mask = indices.notna() & (indices >= 0) & (indices < len(density))
indices = indices[valid_index_mask].astype(int).to_numpy()
merged = merged.loc[valid_index_mask].copy()

E_n_all = np.full(len(density), np.nan)
category_all = np.full(len(density), 'unknown', dtype=object)

E_n_values = pd.to_numeric(merged['E_n'], errors='coerce').to_numpy()
category_values = merged['category'].fillna('unknown').astype(str).to_numpy()

E_n_all[indices] = E_n_values
category_all[indices] = category_values

valid_en_mask = np.isfinite(E_n_all) & np.isfinite(log_density)
E_n_valid = E_n_all[valid_en_mask]
log_density_valid = log_density[valid_en_mask]
category_valid = category_all[valid_en_mask]

rho, p_value = spearmanr(E_n_valid, log_density_valid)
print(f"  Panel (c) valid E[n] samples: {len(E_n_valid)}")
print(f"  Spearman rho: {rho:.4f} (p={p_value:.3e})")

# Figure canvas (1x3)
print("\n[3/4] Generating 1x3 panels...")
fig = plt.figure(figsize=(16, 5.6))
gs = fig.add_gridspec(1, 3, wspace=0.45)

# Panel (a): 2D UMAP density
ax1 = fig.add_subplot(gs[0, 0])
scatter = ax1.scatter(
    umap_coords[:, 0],
    umap_coords[:, 1],
    c=log_density,
    s=3,
    alpha=0.5,
    cmap='viridis',
    rasterized=True,
)
ax1.set_box_aspect(1)
cax1 = inset_axes(
    ax1,
    width="4%",
    height="60%",
    loc='lower left',
    bbox_to_anchor=(1.05, 0.20, 1, 1),
    bbox_transform=ax1.transAxes,
    borderpad=0,
)
cbar1 = fig.colorbar(scatter, cax=cax1)
cbar1.set_label(r'$\log_{10}$(Density)', fontsize=9)
ax1.set_xlabel('UMAP 1')
ax1.set_ylabel('UMAP 2')
ax1.set_title('a', fontsize=13, fontweight='bold')

# Panel (b): Smoothed density contour map over UMAP plane
ax2 = fig.add_subplot(gs[0, 1])

x = umap_coords[:, 0]
y = umap_coords[:, 1]
xi = np.linspace(x.min(), x.max(), 200)
yi = np.linspace(y.min(), y.max(), 200)
XI, YI = np.meshgrid(xi, yi)

ZI_linear = griddata((x, y), log_density, (XI, YI), method='linear')
ZI_nearest = griddata((x, y), log_density, (XI, YI), method='nearest')
ZI = np.where(np.isnan(ZI_linear), ZI_nearest, ZI_linear)
print(f"  Panel (b) grid: {XI.shape[0]}x{XI.shape[1]}")
print(f"  Panel (b) log10(density) range: [{np.nanmin(ZI):.3f}, {np.nanmax(ZI):.3f}]")
print(f"  Panel (b) NaN fill ratio: {np.isnan(ZI_linear).mean():.3%} (linear -> nearest)")

cont = ax2.contourf(
    XI,
    YI,
    ZI,
    levels=18,
    cmap='viridis',
)
ax2.contour(
    XI,
    YI,
    ZI,
    levels=8,
    colors='white',
    linewidths=0.35,
    alpha=0.45,
)
ax2.set_box_aspect(1)
ax2.set_xlabel('UMAP 1')
ax2.set_ylabel('UMAP 2')
ax2.set_title('b', fontsize=13, fontweight='bold')
cax2 = inset_axes(
    ax2,
    width="4%",
    height="60%",
    loc='lower left',
    bbox_to_anchor=(1.05, 0.20, 1, 1),
    bbox_transform=ax2.transAxes,
    borderpad=0,
)
cbar2 = fig.colorbar(cont, cax=cax2)
cbar2.set_label(r'$\log_{10}$(Density)', fontsize=9)

# Panel (c): E[n] vs log10(density)
ax3 = fig.add_subplot(gs[0, 2])
unique_categories = sorted(pd.unique(category_valid))
category_cmap = plt.get_cmap('tab10')

for i, cat in enumerate(unique_categories):
    cat_mask = category_valid == cat
    ax3.scatter(
        E_n_valid[cat_mask],
        log_density_valid[cat_mask],
        s=8,
        alpha=0.65,
        color=category_cmap(i % 10),
        label=cat,
        rasterized=True,
    )

ax3.set_box_aspect(1)
ax3.set_xlabel('E[n]')
ax3.set_ylabel(r'$\log_{10}$(Density)')
ax3.set_title('c', fontsize=13, fontweight='bold')
ax3.text(
    0.97,
    0.06,
    f"Spearman $\\rho$ = {rho:.3f}\np = {p_value:.2e}\nn = {len(E_n_valid)}",
    transform=ax3.transAxes,
    va='bottom',
    ha='right',
    fontsize=8,
    bbox=dict(facecolor='white', edgecolor='none', alpha=0.85),
)

if len(unique_categories) <= 12:
    ax3.legend(
        title='Category',
        fontsize=6,
        title_fontsize=7,
        frameon=False,
        loc='upper right',
    )

fig.subplots_adjust(left=0.05, right=0.95, top=0.90, bottom=0.15)

# Save output
print("\n[4/4] Saving figure...")
output_path = 'manuscript/supp7.png'
plt.savefig(output_path, dpi=600, bbox_inches='tight', facecolor='white')
plt.close()
print(f"  Saved: {output_path}")

print("\n" + "=" * 70)
print("SUPP 7 TECHNICAL CHECK")
print("=" * 70)
print(f"  Total samples: {len(density)}")
print(f"  density vs umap shape match: {density.shape[0] == umap_coords.shape[0]}")
print(f"  Panel (c) valid E[n] samples: {len(E_n_valid)}")
print(f"  Spearman rho: {rho:.4f} (p={p_value:.3e})")
print("=" * 70)
print("Done.")
