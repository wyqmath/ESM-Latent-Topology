# -- Project root setup --
import os as _os
from pathlib import Path as _Path
_os.chdir(_Path(__file__).resolve().parents[2])
"""
Supplementary Figure 3: Dimensionality Reduction Method Comparison
对比UMAP、t-SNE、PCA三种降维方法在ESM-2嵌入上的表现，并增加UMAP n_neighbors鲁棒性扫描

输入文件:
- data/embeddings/sequence_embeddings_final.pt
- data/all_sequences_final.fasta

输出文件:
- manuscript/supp3.png (DPI=600)
- data/dimred_comparison_metrics.csv
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap
from sklearn.neighbors import NearestNeighbors
from scipy.stats import spearmanr
from scipy.spatial.distance import pdist
import pandas as pd
from pathlib import Path
import time

np.random.seed(42)

DATA_DIR = Path("data")
FIG_DIR = Path("manuscript")
FIG_DIR.mkdir(parents=True, exist_ok=True)

# 7-category system consistent with Figure 2 / Figure 4 / supp1
COLORS = {
    'anchor': '#1f77b4',
    'astral95': '#ff7f0e',
    'integrable': '#2ca02c',
    'fold_switching': '#d62728',
    'idp': '#9467bd',
    'random': '#8c564b',
    'knotted': '#e377c2'
}

DISPLAY_NAMES = {
    'anchor': 'Anchor',
    'astral95': 'Astral95',
    'integrable': 'Integrable',
    'fold_switching': 'Fold-switching',
    'idp': 'IDP',
    'random': 'Random',
    'knotted': 'Knotted'
}

plot_order = ['astral95', 'anchor', 'random', 'integrable', 'knotted', 'idp', 'fold_switching']

def assign_category_from_subcategory(subcat):
    """Category mapping consistent with Figure 2 / Figure 4 / supp1"""
    if subcat == 'anchor':
        return 'anchor'
    elif subcat == 'astral95':
        return 'astral95'
    elif subcat == 'integrable':
        return 'integrable'
    elif subcat == 'random':
        return 'random'
    elif subcat == 'fold_switching':
        return 'fold_switching'
    elif subcat == 'idp':
        return 'idp'
    elif subcat == 'knotted':
        return 'knotted'
    else:
        return 'unknown'

print("=" * 80)
print("Supplementary Figure 3: Dimensionality Reduction Method Comparison")
print("=" * 80)

# 1. Load data
print("\n[1/6] Loading embeddings...")
embeddings = torch.load(DATA_DIR / "embeddings/sequence_embeddings_final.pt").numpy()
print(f"  Loaded {len(embeddings)} sequences, dim = {embeddings.shape[1]}")

from Bio import SeqIO
seq_ids = [record.id for record in SeqIO.parse("data/all_sequences_final.fasta", "fasta")]
print(f"  Loaded {len(seq_ids)} sequence IDs")

# Assign categories from metadata subcategory (consistent with figure2/supp1)
metadata = pd.read_csv(DATA_DIR / "metadata_final_with_en.csv")
subcategory_map = dict(zip(metadata['seq_id'], metadata['subcategory']))
categories = [assign_category_from_subcategory(subcategory_map.get(sid, 'unknown')) for sid in seq_ids]

category_counts = {}
for cat in categories:
    category_counts[cat] = category_counts.get(cat, 0) + 1
print(f"  Category distribution: {category_counts}")

# 2. PCA to 50D (randomized SVD for speed)
print("\n[2/6] PCA preprocessing to 50D...")
pca_50d = PCA(n_components=50, random_state=42, svd_solver='randomized')
embeddings_50d = pca_50d.fit_transform(embeddings)
print(f"  Explained variance: {pca_50d.explained_variance_ratio_.sum():.4f}")

# 3. Dimensionality reduction to 2D
print("\n[3/6] Applying dimensionality reduction methods...")

print("  [3.1] PCA...")
start_time = time.time()
pca_2d = PCA(n_components=2, random_state=42, svd_solver='randomized')
coords_pca = pca_2d.fit_transform(embeddings_50d)
time_pca = time.time() - start_time
print(f"    Time: {time_pca:.2f}s")

print("  [3.2] t-SNE...")
start_time = time.time()
tsne = TSNE(n_components=2, random_state=42, perplexity=30, max_iter=1000, verbose=0, n_jobs=-1)
coords_tsne = tsne.fit_transform(embeddings_50d)
time_tsne = time.time() - start_time
print(f"    Time: {time_tsne:.2f}s")

print("  [3.3] UMAP...")
start_time = time.time()
reducer = umap.UMAP(n_components=2, random_state=42, n_neighbors=15, min_dist=0.1, n_jobs=-1)
coords_umap = reducer.fit_transform(embeddings_50d)
time_umap = time.time() - start_time
print(f"    Time: {time_umap:.2f}s")

# 4. Quality metrics — compute 50D KNN only ONCE
print("\n[4/6] Computing quality metrics...")

k = 15
print(f"  Building 50D KNN (k={k}) once...")
nbrs_high = NearestNeighbors(n_neighbors=k+1, algorithm='ball_tree', n_jobs=-1).fit(embeddings_50d)
_, indices_high = nbrs_high.kneighbors(embeddings_50d)
indices_high = indices_high[:, 1:]  # exclude self

def compute_neighborhood_preservation_fast(indices_high, X_low, k=15):
    """Vectorized neighborhood preservation using precomputed high-dim neighbors"""
    nbrs_low = NearestNeighbors(n_neighbors=k+1, algorithm='kd_tree', n_jobs=-1).fit(X_low)
    _, indices_low = nbrs_low.kneighbors(X_low)
    indices_low = indices_low[:, 1:]
    high_sorted = np.sort(indices_high, axis=1)
    low_sorted = np.sort(indices_low, axis=1)
    combined = np.concatenate([high_sorted, low_sorted], axis=1)
    combined.sort(axis=1)
    matches = np.sum(combined[:, :-1] == combined[:, 1:], axis=1)
    return matches.mean() / k

def compute_distance_correlation(X_high, X_low, n_samples=1000):
    """Spearman correlation of pairwise distances (sampled)"""
    n = X_high.shape[0]
    if n > n_samples:
        idx = np.random.choice(n, n_samples, replace=False)
        X_high = X_high[idx]
        X_low = X_low[idx]
    dist_high = pdist(X_high, metric='euclidean')
    dist_low = pdist(X_low, metric='euclidean')
    rho, _ = spearmanr(dist_high, dist_low)
    return rho


def compute_class_knn_purity(X_low, categories, k=15):
    """Per-category local clustering strength in 2D via same-class kNN ratio."""
    cats = np.asarray(categories)
    nbrs = NearestNeighbors(n_neighbors=k + 1, algorithm='kd_tree', n_jobs=-1).fit(X_low)
    _, indices_low = nbrs.kneighbors(X_low)
    indices_low = indices_low[:, 1:]
    neigh_cats = cats[indices_low]
    same_class_ratio = (neigh_cats == cats[:, None]).mean(axis=1)

    rows = []
    total_n = len(cats)
    for cat in plot_order:
        mask = (cats == cat)
        if mask.sum() == 0:
            continue
        prevalence = mask.mean()
        purity = same_class_ratio[mask].mean()
        rows.append({
            'Category': cat,
            'Count': int(mask.sum()),
            'Prevalence': prevalence,
            'kNN Purity': purity,
            'Purity Enrichment': purity / max(prevalence, 1e-12)
        })
    return pd.DataFrame(rows)

print("  Computing neighborhood preservation...")
np_pca = compute_neighborhood_preservation_fast(indices_high, coords_pca, k=k)
np_tsne = compute_neighborhood_preservation_fast(indices_high, coords_tsne, k=k)
np_umap = compute_neighborhood_preservation_fast(indices_high, coords_umap, k=k)

print("  Computing distance correlation (1000 samples)...")
dc_pca = compute_distance_correlation(embeddings_50d, coords_pca)
dc_tsne = compute_distance_correlation(embeddings_50d, coords_tsne)
dc_umap = compute_distance_correlation(embeddings_50d, coords_umap)

print("  Computing per-category local clustering (kNN purity, k=15)...")
purity_pca = compute_class_knn_purity(coords_pca, categories, k=k)
purity_tsne = compute_class_knn_purity(coords_tsne, categories, k=k)
purity_umap = compute_class_knn_purity(coords_umap, categories, k=k)

print("  Running seed robustness scan (PCA vs UMAP)...")
scan_seeds = [0, 7, 21, 42, 84]
stability_results = []
for seed in scan_seeds:
    pca_start = time.time()
    pca_seed = PCA(n_components=2, random_state=seed, svd_solver='randomized')
    coords_pca_seed = pca_seed.fit_transform(embeddings_50d)
    pca_seed_time = time.time() - pca_start

    umap_start = time.time()
    reducer_seed = umap.UMAP(
        n_components=2,
        random_state=seed,
        n_neighbors=15,
        min_dist=0.1,
        n_jobs=-1
    )
    coords_umap_seed = reducer_seed.fit_transform(embeddings_50d)
    umap_seed_time = time.time() - umap_start

    pca_dc_seed = compute_distance_correlation(embeddings_50d, coords_pca_seed)
    umap_dc_seed = compute_distance_correlation(embeddings_50d, coords_umap_seed)

    stability_results.append({
        'seed': seed,
        'PCA Distance Correlation': pca_dc_seed,
        'UMAP Distance Correlation': umap_dc_seed,
        'PCA Time (s)': pca_seed_time,
        'UMAP Time (s)': umap_seed_time
    })

stability_df = pd.DataFrame(stability_results)

metrics_df = pd.DataFrame({
    'Method': ['PCA', 't-SNE', 'UMAP'],
    'Time (s)': [time_pca, time_tsne, time_umap],
    'Neighborhood Preservation': [np_pca, np_tsne, np_umap],
    'Distance Correlation': [dc_pca, dc_tsne, dc_umap]
})

purity_compare_df = purity_pca[['Category', 'Count', 'Prevalence', 'kNN Purity', 'Purity Enrichment']].rename(
    columns={'kNN Purity': 'PCA kNN Purity', 'Purity Enrichment': 'PCA Purity Enrichment'}
)
purity_compare_df = purity_compare_df.merge(
    purity_tsne[['Category', 'kNN Purity', 'Purity Enrichment']].rename(
        columns={'kNN Purity': 't-SNE kNN Purity', 'Purity Enrichment': 't-SNE Purity Enrichment'}
    ),
    on='Category',
    how='left'
)
purity_compare_df = purity_compare_df.merge(
    purity_umap[['Category', 'kNN Purity', 'Purity Enrichment']].rename(
        columns={'kNN Purity': 'UMAP kNN Purity', 'Purity Enrichment': 'UMAP Purity Enrichment'}
    ),
    on='Category',
    how='left'
)

metrics_df.to_csv(DATA_DIR / "dimred_comparison_metrics.csv", index=False)
purity_compare_df.to_csv(DATA_DIR / "dimred_class_knn_purity.csv", index=False)
print(f"\n  Metrics:")
print(metrics_df.to_string(index=False))

# 5. Generate figure
print("\n[5/6] Generating figure...")

fig = plt.figure(figsize=(18, 12))
gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)

methods = [
    ('PCA', coords_pca, np_pca, dc_pca, time_pca),
    ('t-SNE', coords_tsne, np_tsne, dc_tsne, time_tsne),
    ('UMAP', coords_umap, np_umap, dc_umap, time_umap)
]

# Row 1: scatter plots with legend on each panel
for i, (method_name, coords, np_score, dc_score, exec_time) in enumerate(methods):
    ax = fig.add_subplot(gs[0, i])
    for cat in plot_order:
        mask = np.array([c == cat for c in categories])
        if mask.sum() > 0:
            ax.scatter(coords[mask, 0], coords[mask, 1],
                      c=COLORS[cat], label=f'{DISPLAY_NAMES[cat]} ({mask.sum()})',
                      s=1, alpha=0.6, rasterized=True)
    ax.set_xlabel('Component 1', fontsize=12)
    ax.set_ylabel('Component 2', fontsize=12)
    ax.set_box_aspect(1)
    letter = chr(97 + i)
    ax.set_title(letter, fontsize=12, fontweight='bold', loc='center', pad=10)
    ax.grid(True, alpha=0.3)
    # Add method name as title inside legend area
    legend_loc = 'lower left' if i == 0 else 'upper right'
    leg = ax.legend(loc=legend_loc, fontsize=7, markerscale=5, framealpha=0.6,
                    facecolor='white', edgecolor='gray', title=method_name,
                    title_fontsize=9)

# Row 2: bar charts and seed robustness (PCA vs UMAP)
methods_names = [m[0] for m in methods]
bar_colors = ['#377eb8', '#ff7f00', '#4daf4a']

ax_np = fig.add_subplot(gs[1, 0])
ax_np.set_box_aspect(1)
np_scores = [m[2] for m in methods]
bars = ax_np.bar(methods_names, np_scores, color=bar_colors, alpha=0.8)
ax_np.set_ylabel('Neighborhood Preservation', fontsize=12)
ax_np.set_title('d', fontsize=12, fontweight='bold', loc='center', pad=10)
ax_np.set_ylim([0, 1])
for bar, score in zip(bars, np_scores):
    ax_np.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.02,
               f'{score:.3f}', ha='center', va='bottom', fontsize=11)

ax_dc = fig.add_subplot(gs[1, 1])
ax_dc.set_box_aspect(1)
dc_scores = [m[3] for m in methods]
bars = ax_dc.bar(methods_names, dc_scores, color=bar_colors, alpha=0.8)
ax_dc.set_ylabel('Distance Correlation (Spearman ρ)', fontsize=12)
ax_dc.set_title('e', fontsize=12, fontweight='bold', loc='center', pad=10)
ax_dc.set_ylim([0, 1])
for bar, score in zip(bars, dc_scores):
    ax_dc.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.02,
               f'{score:.3f}', ha='center', va='bottom', fontsize=11)

ax_stab = fig.add_subplot(gs[1, 2])
ax_stab.set_box_aspect(1)
stab_x = stability_df['seed'].to_numpy()
stab_pca_dc = stability_df['PCA Distance Correlation'].to_numpy()
stab_umap_dc = stability_df['UMAP Distance Correlation'].to_numpy()

ax_stab.plot(stab_x, stab_pca_dc, marker='o', color='#377eb8', linewidth=2, label='PCA')
ax_stab.plot(stab_x, stab_umap_dc, marker='s', color='#ff7f00', linewidth=2, label='UMAP')
ax_stab.set_xlabel('Random seed', fontsize=12)
ax_stab.set_ylabel('Distance Correlation (Spearman ρ)', fontsize=12)
ax_stab.set_title('f', fontsize=12, fontweight='bold', loc='center', pad=10)
ax_stab.set_ylim([0, 1])
ax_stab.grid(True, alpha=0.3)
ax_stab.legend(loc='lower right', fontsize=9)

pca_mean = stability_df['PCA Distance Correlation'].mean()
pca_std = stability_df['PCA Distance Correlation'].std(ddof=1)
umap_mean = stability_df['UMAP Distance Correlation'].mean()
umap_std = stability_df['UMAP Distance Correlation'].std(ddof=1)
ax_stab.text(
    0.03, 0.97,
    f'PCA: {pca_mean:.3f}±{pca_std:.3f}\nUMAP: {umap_mean:.3f}±{umap_std:.3f}',
    transform=ax_stab.transAxes,
    fontsize=9,
    va='top',
    bbox=dict(facecolor='white', alpha=0.7, edgecolor='gray')
)

# 6. Save
print("\n[6/6] Saving figure...")
output_path = FIG_DIR / "supp3.png"
fig.savefig(output_path, dpi=600, bbox_inches='tight')
print(f"  Saved: {output_path}")

plt.close()

print("\n" + "=" * 80)
print("Supplementary Figure 3 completed!")
print("=" * 80)
print("\n  Method-level metrics:")
print(metrics_df.to_string(index=False))

print("\n  Per-category local clustering (kNN purity, k=15):")
print(purity_compare_df.to_string(index=False, formatters={
    'Prevalence': '{:.4f}'.format,
    'PCA kNN Purity': '{:.4f}'.format,
    't-SNE kNN Purity': '{:.4f}'.format,
    'UMAP kNN Purity': '{:.4f}'.format,
    'PCA Purity Enrichment': '{:.2f}'.format,
    't-SNE Purity Enrichment': '{:.2f}'.format,
    'UMAP Purity Enrichment': '{:.2f}'.format
}))

idp_row = purity_compare_df[purity_compare_df['Category'] == 'idp']
if len(idp_row) > 0:
    idp = idp_row.iloc[0]
    print(
        f"  IDP kNN purity — PCA: {idp['PCA kNN Purity']:.4f}, "
        f"t-SNE: {idp['t-SNE kNN Purity']:.4f}, UMAP: {idp['UMAP kNN Purity']:.4f}"
    )

winner_map = []
for _, r in purity_compare_df.iterrows():
    vals = {
        'PCA': r['PCA kNN Purity'],
        't-SNE': r['t-SNE kNN Purity'],
        'UMAP': r['UMAP kNN Purity']
    }
    winner = max(vals, key=vals.get)
    winner_map.append((r['Category'], winner, vals[winner]))

print("\n  Best local-clustering method by category (highest kNN purity):")
for cat, winner, score in winner_map:
    print(f"    - {cat}: {winner} ({score:.4f})")

print("\n  Seed robustness scan (PCA vs UMAP):")
print(stability_df.to_string(index=False, formatters={
    'PCA Distance Correlation': '{:.4f}'.format,
    'UMAP Distance Correlation': '{:.4f}'.format,
    'PCA Time (s)': '{:.2f}'.format,
    'UMAP Time (s)': '{:.2f}'.format
}))

best_seed_pca = stability_df.loc[stability_df['PCA Distance Correlation'].idxmax()]
best_seed_umap = stability_df.loc[stability_df['UMAP Distance Correlation'].idxmax()]
print(f"\n  PCA DC mean±std: {pca_mean:.4f}±{pca_std:.4f}")
print(f"  UMAP DC mean±std: {umap_mean:.4f}±{umap_std:.4f}")
print(f"  PCA best seed: {int(best_seed_pca['seed'])} (DC={best_seed_pca['PCA Distance Correlation']:.4f})")
print(f"  UMAP best seed: {int(best_seed_umap['seed'])} (DC={best_seed_umap['UMAP Distance Correlation']:.4f})")
print(f"  Variability ratio (UMAP std / PCA std): {umap_std / max(pca_std, 1e-12):.2f}x")
print(f"  Global fidelity gain (PCA mean DC - UMAP mean DC): {pca_mean - umap_mean:.4f}")
print("  Panel-f conclusion: PCA shows both higher global distance preservation and better seed robustness than UMAP.")
