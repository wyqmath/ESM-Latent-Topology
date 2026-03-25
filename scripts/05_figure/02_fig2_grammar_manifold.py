# -- Project root setup --
import os as _os
from pathlib import Path as _Path
_os.chdir(_Path(__file__).resolve().parents[2])
"""
Figure 2: PCA Grammar Manifold (3 panels)
Panel A: PCA 2D colored by 7 categories (Random separation highlighted)
Panel B: PCA colored by sequence length + ρ(PC1, length)
Panel C: PCA vs UMAP pairwise distance correlation
Output: manuscript/figure2.png
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from scipy.spatial.distance import pdist
from sklearn.metrics import silhouette_score

print("=" * 70)
print("Figure 2: PCA Grammar Manifold")
print("=" * 70)

# Publication style
plt.rcParams['figure.dpi'] = 600
plt.rcParams['savefig.dpi'] = 600
plt.rcParams['font.size'] = 8
plt.rcParams['axes.labelsize'] = 9
plt.rcParams['axes.titlesize'] = 10
plt.rcParams['xtick.labelsize'] = 8
plt.rcParams['ytick.labelsize'] = 8
plt.rcParams['legend.fontsize'] = 7
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']

# Load data
print("\nLoading data...")
pca_2d = np.load('data/pca/pca_embeddings_2d.npy')
pca_50d = np.load('data/pca/pca_embeddings_50d.npy')
umap_2d = np.load('data/umap/umap_embeddings_2d.npy')
metadata = pd.read_csv('data/metadata_final_with_en.csv')
explained_var = np.load('data/pca/explained_variance.npy')

print(f"  Sequences: {len(pca_2d)}")
assert len(pca_2d) == len(metadata), f"Shape mismatch: {len(pca_2d)} vs {len(metadata)}"

# Assign 7-category labels
def assign_cat7(row):
    cat, sub = row['category'], row['subcategory']
    if cat == 'anchor': return 'Anchor'
    elif cat == 'astral95': return 'Astral95'
    elif cat == 'integrable': return 'Integrable'
    elif cat == 'control': return 'Random'
    elif cat == 'extreme':
        if sub == 'fold_switching': return 'Fold-switching'
        elif sub == 'idp': return 'IDP'
        else: return 'Knotted'
    return 'Unknown'

metadata['cat7'] = metadata.apply(assign_cat7, axis=1)

# Color scheme
COLORS = {
    'Astral95': '#ff7f0e', 'Anchor': '#1f77b4', 'Random': '#8c564b',
    'Integrable': '#2ca02c', 'IDP': '#9467bd', 'Fold-switching': '#d62728',
    'Knotted': '#e377c2',
}
PLOT_ORDER = ['Astral95', 'Anchor', 'Random', 'Integrable', 'Knotted', 'IDP', 'Fold-switching']
SIZES = {'Astral95': 1, 'Anchor': 3, 'Random': 6, 'Integrable': 8, 'IDP': 8, 'Fold-switching': 8, 'Knotted': 8}
ALPHAS = {'Astral95': 0.3, 'Anchor': 0.5, 'Random': 0.7, 'Integrable': 0.8, 'IDP': 0.8, 'Fold-switching': 0.8, 'Knotted': 0.8}

# Compute Silhouette for Random vs rest (binary)
is_random = (metadata['cat7'] == 'Random').astype(int).values
sil_random_binary = silhouette_score(pca_50d, is_random)

# Compute Silhouette per category (5-class: by original category)
sil_5class = silhouette_score(pca_50d, metadata['category'].values)

# Spearman correlations with length
lengths = metadata['length'].values
rho_pc1_len, p_pc1_len = spearmanr(pca_2d[:, 0], lengths)
rho_pc2_len, p_pc2_len = spearmanr(pca_2d[:, 1], lengths)

# ============ Figure ============
fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))

# ============ Panel A: PCA by category ============
ax = axes[0]
for cat in PLOT_ORDER:
    mask = metadata['cat7'].values == cat
    n = mask.sum()
    if n == 0:
        continue
    ax.scatter(pca_2d[mask, 0], pca_2d[mask, 1],
               c=COLORS[cat], s=SIZES[cat], alpha=ALPHAS[cat],
               edgecolors='none', rasterized=True)

ax.set_xlabel(f'PC1 ({explained_var[0]*100:.1f}% var)')
ax.set_ylabel(f'PC2 ({explained_var[1]*100:.1f}% var)')
ax.set_title('a', fontsize=12, fontweight='bold', loc='center', pad=10)
ax.grid(True, alpha=0.2, linewidth=0.5)

# Uniform-size legend handles
from matplotlib.lines import Line2D
legend_handles = []
for cat in PLOT_ORDER:
    n = (metadata['cat7'] == cat).sum()
    if n > 0:
        legend_handles.append(Line2D([0], [0], marker='o', color='w',
                                     markerfacecolor=COLORS[cat], markersize=6,
                                     label=f'{cat} (n={n})', linestyle='None'))
ax.legend(handles=legend_handles, loc='upper left', frameon=True, fontsize=6,
          framealpha=0.9, edgecolor='gray')

# Silhouette stats below legend
props = dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='gray', linewidth=0.8)
ax.text(0.02, 0.02,
        f'Silhouette (Random vs rest) = {sil_random_binary:.3f}\nSilhouette (5-class) = {sil_5class:.3f}',
        transform=ax.transAxes, fontsize=6.5, va='bottom', ha='left', bbox=props)

# ============ Panel B: PCA by sequence length ============
ax = axes[1]
len_clip_lo = np.percentile(lengths, 1)
len_clip_hi = np.percentile(lengths, 99)
sc = ax.scatter(pca_2d[:, 0], pca_2d[:, 1],
                c=lengths, cmap='viridis', s=1, alpha=0.5,
                vmin=len_clip_lo, vmax=len_clip_hi,
                edgecolors='none', rasterized=True)
cbar = plt.colorbar(sc, ax=ax, shrink=0.8)
cbar.set_label('Sequence length (aa)', fontsize=8)
cbar.ax.tick_params(labelsize=7)

ax.set_xlabel(f'PC1 ({explained_var[0]*100:.1f}% var)')
ax.set_ylabel(f'PC2 ({explained_var[1]*100:.1f}% var)')
ax.set_title('b', fontsize=12, fontweight='bold', loc='center', pad=10)
ax.grid(True, alpha=0.2, linewidth=0.5)

props = dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='gray', linewidth=0.8)
ax.text(0.98, 0.02,
        f'ρ(PC1, length) = {rho_pc1_len:.3f} (p={p_pc1_len:.1e})\n'
        f'ρ(PC2, length) = {rho_pc2_len:.3f} (p={p_pc2_len:.1e})\n'
        f'Color clipped to [{len_clip_lo:.0f}, {len_clip_hi:.0f}] aa (1st–99th pctl)',
        transform=ax.transAxes, fontsize=7, va='bottom', ha='right', bbox=props)

# ============ Panel C: Distance correlation PCA vs UMAP ============
ax = axes[2]
print("\nComputing pairwise distance correlations (subsample 2000)...")
np.random.seed(42)
n_sub = min(2000, len(pca_2d))
idx_sub = np.random.choice(len(pca_2d), n_sub, replace=False)

pca_sub = pca_2d[idx_sub]
umap_sub = umap_2d[idx_sub]
hd_sub = pca_50d[idx_sub]

dist_hd = pdist(hd_sub, metric='euclidean')
dist_pca = pdist(pca_sub, metric='euclidean')
dist_umap = pdist(umap_sub, metric='euclidean')

rho_pca_hd, p_pca_hd = spearmanr(dist_hd, dist_pca)
rho_umap_hd, p_umap_hd = spearmanr(dist_hd, dist_umap)

# Subsample distances for plotting (too many pairs)
n_pairs = len(dist_hd)
plot_idx = np.random.choice(n_pairs, min(50000, n_pairs), replace=False)

ax.scatter(dist_hd[plot_idx], dist_pca[plot_idx],
           c='#3498db', s=0.5, alpha=0.15, label=f'PCA (ρ={rho_pca_hd:.3f})',
           rasterized=True)
ax.scatter(dist_hd[plot_idx], dist_umap[plot_idx],
           c='#e74c3c', s=0.5, alpha=0.15, label=f'UMAP (ρ={rho_umap_hd:.3f})',
           rasterized=True)

ax.set_xlabel('50D PCA pairwise distance')
ax.set_ylabel('2D embedding pairwise distance')
ax.set_title('c', fontsize=12, fontweight='bold', loc='center', pad=10)
ax.legend(loc='upper left', frameon=True, fontsize=7, framealpha=0.9,
          edgecolor='gray', markerscale=5)
ax.grid(True, alpha=0.2, linewidth=0.5)

props = dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='gray', linewidth=0.8)
ax.text(0.98, 0.02,
        f'PCA: ρ={rho_pca_hd:.3f} (p={p_pca_hd:.1e})\n'
        f'UMAP: ρ={rho_umap_hd:.3f} (p={p_umap_hd:.1e})\n'
        f'n={n_sub} subsampled\n({n_sub}×{n_sub-1}÷2={n_pairs:,} pairs)',
        transform=ax.transAxes, fontsize=7, va='bottom', ha='right', bbox=props)

plt.tight_layout()
plt.savefig('manuscript/figure2.png', dpi=600, bbox_inches='tight')
plt.close()

# ============ Summary Statistics (for figure caption) ============
print("\n" + "=" * 70)
print("FIGURE 2 — STATISTICS FOR CAPTION")
print("=" * 70)

print(f"\n{'─'*70}")
print(f"[Panel A] PCA 2D by 7-category")
print(f"{'─'*70}")
for cat in PLOT_ORDER:
    n = (metadata['cat7'] == cat).sum()
    print(f"  {cat:<20s}: n = {n}")
print(f"  {'Total':<20s}: n = {len(metadata)}")
print(f"  PC1 explained variance: {explained_var[0]*100:.2f}%")
print(f"  PC2 explained variance: {explained_var[1]*100:.2f}%")
print(f"  Silhouette (Random vs rest, 50D PCA): {sil_random_binary:.4f}")
print(f"  Silhouette (5-class, 50D PCA): {sil_5class:.4f}")
print(f"  Method: sklearn silhouette_score, Euclidean distance in 50D PCA space")

print(f"\n{'─'*70}")
print(f"[Panel B] PCA colored by sequence length")
print(f"{'─'*70}")
print(f"  n = {len(metadata)}")
print(f"  Length range: [{lengths.min()}, {lengths.max()}] aa")
print(f"  Length mean ± std: {lengths.mean():.1f} ± {lengths.std():.1f} aa")
print(f"  Length median: {np.median(lengths):.0f} aa")
print(f"  Color clip: [{len_clip_lo:.0f}, {len_clip_hi:.0f}] aa (1st–99th percentile)")
print(f"  ρ(PC1, length) = {rho_pc1_len:.4f}, p = {p_pc1_len:.2e}")
print(f"  ρ(PC2, length) = {rho_pc2_len:.4f}, p = {p_pc2_len:.2e}")
print(f"  → PC1 is NOT a length counter (ρ ≈ 0)")
print(f"  → PC2 shows weak negative correlation with length")
print(f"  Method: two-tailed Spearman rank correlation")

print(f"\n{'─'*70}")
print(f"[Panel C] Pairwise distance correlation: PCA-2D vs UMAP-2D vs 50D")
print(f"{'─'*70}")
print(f"  Subsample: n = {n_sub} sequences (random seed=42)")
print(f"  Total pairs: {n_pairs:,} ({n_sub}×{n_sub-1}÷2)")
print(f"  Plotted pairs: {min(50000, n_pairs):,} (random subsample for visualization)")
print(f"  ρ(50D, PCA-2D) = {rho_pca_hd:.4f}, p = {p_pca_hd:.2e}")
print(f"  ρ(50D, UMAP-2D) = {rho_umap_hd:.4f}, p = {p_umap_hd:.2e}")
print(f"  PCA/UMAP ratio: {rho_pca_hd/rho_umap_hd:.2f}× better distance preservation")
print(f"  → PCA preserves global pairwise distance structure; UMAP distorts it")
print(f"  → Justifies using PCA (not UMAP) for manifold geometry analysis")
print(f"  Method: Spearman rank correlation on Euclidean pairwise distances")

print(f"\n{'─'*70}")
print(f"KEY MESSAGES:")
print(f"  1. Random sequences are cleanly separated (Silhouette = {sil_random_binary:.3f})")
print(f"     → ESM-2 recognizes evolutionary sequence grammar")
print(f"  2. PC1 ≠ sequence length (ρ = {rho_pc1_len:.4f}, p = {p_pc1_len:.2e})")
print(f"     → The horseshoe is NOT a trivial length artifact")
print(f"  3. PCA preserves {rho_pca_hd/rho_umap_hd:.1f}× more distance structure than UMAP")
print(f"     → PCA is the correct embedding for geometric analysis")
print(f"{'─'*70}")

print(f"\nSaved: manuscript/figure2.png")
print("=" * 70)
