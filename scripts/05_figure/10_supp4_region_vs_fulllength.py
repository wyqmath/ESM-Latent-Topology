#!/usr/bin/env python3
"""
Supp Figure 4: Single-sample micro-tracking of topological aliasing (1×3)
Panel (a): Raincloud distribution of per-sample silhouette (full-length vs region)
Panel (b): DeltaSilhouette scatter (x = full-length silhouette, y = delta)
Panel (c): Length attribution (x = sequence length, y = delta) with OLS fit
Output: manuscript/supp4.png
"""
# -- Project root setup --
import os as _os
from pathlib import Path as _Path
_os.chdir(_Path(__file__).resolve().parents[2])

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import linregress, t as student_t
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_samples


# -----------------------------
# 1) Load metadata and labels
# -----------------------------
metadata = pd.read_csv('data/metadata_final_with_en.csv')

CAT7_MAP = {
    'anchor': 'Anchor',
    'astral95': 'Astral95',
    'integrable': 'Integrable',
    'random': 'Random',
    'fold_switching': 'Fold-switching',
    'idp': 'IDP',
    'knotted': 'Knotted',
}
metadata['cat7'] = metadata['subcategory'].map(CAT7_MAP)
cat7_labels = metadata['cat7'].values

PCA_DIM = 50
EXTREME_CATS = ['Fold-switching', 'IDP', 'Knotted']
CAT_COLORS = {
    'Fold-switching': '#d62728',
    'IDP': '#9467bd',
    'Knotted': '#ff7f0e',
}
COLOR_FL = '#4c78a8'   # full-length
COLOR_RG = '#f58518'   # region-replaced


# -----------------------------------------
# 2) Compute per-sample silhouette (FL/RG)
# -----------------------------------------
print('Loading full-length embeddings...')
emb_fl = torch.load(
    'data/embeddings/sequence_embeddings_fulllength.pt',
    map_location='cpu',
    weights_only=True,
).float().numpy()

print('Loading region-replaced embeddings...')
emb_rr = torch.load(
    'data/embeddings/sequence_embeddings_region_replaced.pt',
    map_location='cpu',
    weights_only=True,
).float().numpy()

pca_fl = PCA(n_components=PCA_DIM, random_state=42).fit_transform(emb_fl)
pca_rr = PCA(n_components=PCA_DIM, random_state=42).fit_transform(emb_rr)

sil_fl = silhouette_samples(pca_fl, cat7_labels)
sil_rr = silhouette_samples(pca_rr, cat7_labels)
delta = sil_rr - sil_fl

metadata['sil_fl'] = sil_fl
metadata['sil_rr'] = sil_rr
metadata['delta'] = delta

extreme_mask = metadata['cat7'].isin(EXTREME_CATS).values
extreme_df = metadata.loc[extreme_mask].copy()

# For panels (b)(c): remove extreme delta outliers by 99th percentile of |delta|
delta_abs_q99 = extreme_df['delta'].abs().quantile(0.99)
plot_df = extreme_df.loc[extreme_df['delta'].abs() <= delta_abs_q99].copy()


# -----------------------------------------
# 3) Figure layout and styling (1x3)
# -----------------------------------------
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

fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))


# -----------------------------------------
# Panel (a): Raincloud distributions
# -----------------------------------------
ax = axes[0]
np.random.seed(42)

base_positions = np.arange(len(EXTREME_CATS), dtype=float)
offset = 0.16
violin_width = 0.24

for i, cat in enumerate(EXTREME_CATS):
    cat_mask = (metadata['cat7'] == cat).values
    vals_fl = sil_fl[cat_mask]
    vals_rg = sil_rr[cat_mask]

    # Full-length raincloud (left group)
    x_fl = base_positions[i] - offset
    parts_fl = ax.violinplot([vals_fl], positions=[x_fl], widths=violin_width,
                             showmeans=False, showmedians=False, showextrema=False)
    body_fl = parts_fl['bodies'][0]
    body_fl.set_facecolor(COLOR_FL)
    body_fl.set_edgecolor(COLOR_FL)
    body_fl.set_alpha(0.45)
    verts_fl = body_fl.get_paths()[0].vertices
    verts_fl[:, 0] = np.minimum(verts_fl[:, 0], x_fl)  # keep left half

    jitter_fl = np.random.uniform(0.005, 0.08, size=len(vals_fl))
    ax.scatter(np.full(len(vals_fl), x_fl) + jitter_fl, vals_fl,
               s=6, alpha=0.35, color=COLOR_FL, edgecolors='none', rasterized=True)

    # Region raincloud (right group)
    x_rg = base_positions[i] + offset
    parts_rg = ax.violinplot([vals_rg], positions=[x_rg], widths=violin_width,
                             showmeans=False, showmedians=False, showextrema=False)
    body_rg = parts_rg['bodies'][0]
    body_rg.set_facecolor(COLOR_RG)
    body_rg.set_edgecolor(COLOR_RG)
    body_rg.set_alpha(0.45)
    verts_rg = body_rg.get_paths()[0].vertices
    verts_rg[:, 0] = np.minimum(verts_rg[:, 0], x_rg)  # keep left half

    jitter_rg = np.random.uniform(0.005, 0.08, size=len(vals_rg))
    ax.scatter(np.full(len(vals_rg), x_rg) + jitter_rg, vals_rg,
               s=6, alpha=0.35, color=COLOR_RG, edgecolors='none', rasterized=True)

    # Mean markers
    ax.scatter([x_fl, x_rg], [vals_fl.mean(), vals_rg.mean()],
               s=24, c=[COLOR_FL, COLOR_RG], edgecolors='black', linewidths=0.4, zorder=6)

ax.axhline(0, color='gray', linewidth=0.8, linestyle='--', alpha=0.7)
ax.set_xticks(base_positions)
ax.set_xticklabels([f'{c}\n(n={(metadata["cat7"] == c).sum()})' for c in EXTREME_CATS])
ax.set_ylabel('Per-sample silhouette')
ax.set_title('a', fontweight='bold')
ax.grid(True, axis='y', alpha=0.2, linewidth=0.5)

legend_handles = [
    plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=COLOR_FL,
               markeredgecolor='black', markeredgewidth=0.4, markersize=5, label='Full-length'),
    plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=COLOR_RG,
               markeredgecolor='black', markeredgewidth=0.4, markersize=5, label='Region'),
]
ax.legend(handles=legend_handles, loc='lower left', frameon=True)


# -----------------------------------------
# Panel (b): DeltaSilhouette scatter
# -----------------------------------------
ax = axes[1]

for cat in EXTREME_CATS:
    sub = plot_df[plot_df['cat7'] == cat]
    ax.scatter(sub['sil_fl'], sub['delta'], s=10, alpha=0.45,
               color=CAT_COLORS[cat], edgecolors='none', label=cat, rasterized=True)

    # category mean marker
    mx = sub['sil_fl'].mean()
    my = sub['delta'].mean()
    ax.scatter(mx, my, s=80, marker='D', color=CAT_COLORS[cat],
               edgecolors='black', linewidths=0.6, zorder=6)

ax.axhline(0, color='gray', linewidth=0.8, linestyle='--', alpha=0.7)
ax.set_xlabel('Full-length silhouette')
ax.set_ylabel('Delta silhouette (region - full-length)')
ax.set_title('b', fontweight='bold')
ax.grid(True, alpha=0.2, linewidth=0.5)
ax.legend(loc='upper right', frameon=True)


# -----------------------------------------
# Panel (c): Length attribution with OLS
# -----------------------------------------
ax = axes[2]

for cat in EXTREME_CATS:
    sub = plot_df[plot_df['cat7'] == cat]
    ax.scatter(sub['length'], sub['delta'], s=10, alpha=0.45,
               color=CAT_COLORS[cat], edgecolors='none', label=cat, rasterized=True)

x = plot_df['length'].values
y = plot_df['delta'].values
ols = linregress(x, y)

x_line = np.linspace(x.min(), x.max(), 200)
y_line = ols.slope * x_line + ols.intercept

# 95% confidence interval for mean fitted line
n = len(x)
x_mean = x.mean()
ssx = np.sum((x - x_mean) ** 2)
residuals = y - (ols.slope * x + ols.intercept)
s_err = np.sqrt(np.sum(residuals ** 2) / (n - 2))
t_crit = student_t.ppf(0.975, df=n - 2)
se_fit = s_err * np.sqrt((1.0 / n) + ((x_line - x_mean) ** 2) / ssx)
ci_low = y_line - t_crit * se_fit
ci_high = y_line + t_crit * se_fit

FIT_COLOR = '#2ca02c'
ax.plot(x_line, y_line, color=FIT_COLOR, linewidth=1.4, label='OLS fit')
ax.fill_between(x_line, ci_low, ci_high, color=FIT_COLOR, alpha=0.18, linewidth=0, label='95% CI')
ax.axhline(0, color='gray', linewidth=0.8, linestyle='--', alpha=0.7)

ax.set_xlabel('Sequence length (aa)')
ax.set_ylabel('Delta silhouette (region - full-length)')
ax.set_title('c', fontweight='bold')
ax.grid(True, alpha=0.2, linewidth=0.5)
ax.legend(loc='upper right', frameon=True)

r2 = ols.rvalue ** 2
annot = f'slope = {ols.slope:.2e}\nR² = {r2:.4f}\np = {ols.pvalue:.2e}'
ax.text(0.04, 0.96, annot, transform=ax.transAxes, va='top', ha='left',
        fontsize=7, bbox=dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='gray'))


# -----------------------------------------
# Save figure
# -----------------------------------------
plt.tight_layout()
plt.savefig('manuscript/supp4.png', dpi=600, bbox_inches='tight')
plt.close()
print('\nSaved: manuscript/supp4.png')


# -----------------------------------------
# Terminal summary stats
# -----------------------------------------
print('\nDeltaSilhouette summary by category:')
print('=' * 86)
print(f"{'Category':<16s} {'n_total':>8s} {'n_plot':>8s} {'mean(fl)':>12s} {'mean(region)':>14s} {'mean(delta)':>14s} {'std(delta)':>12s}")
print('=' * 86)
for cat in EXTREME_CATS:
    sub_all = extreme_df[extreme_df['cat7'] == cat]
    sub_plot = plot_df[plot_df['cat7'] == cat]
    print(
        f"{cat:<16s} {len(sub_all):>8d} {len(sub_plot):>8d} "
        f"{sub_all['sil_fl'].mean():>12.6f} {sub_all['sil_rr'].mean():>14.6f} "
        f"{sub_all['delta'].mean():>+14.6f} {sub_all['delta'].std():>12.6f}"
    )
print('=' * 86)
print(f'Filtered for plotting by |delta| <= q99: q99={delta_abs_q99:.6f}; kept {len(plot_df)}/{len(extreme_df)} samples')

print('\nDelta quantiles (all extreme samples):')
qs = [0.00, 0.01, 0.05, 0.50, 0.95, 0.99, 1.00]
for q in qs:
    print(f'  q{int(q*100):>2d}: {np.quantile(extreme_df["delta"].values, q):+.6f}')

print('\nOLS (length -> delta) on filtered plotting samples:')
print(f'  n = {len(plot_df)}')
print(f'  slope = {ols.slope:+.6e}')
print(f'  intercept = {ols.intercept:+.6e}')
print(f'  R² = {r2:.6f}')
print(f'  p = {ols.pvalue:.6e}')

print('\nOLS (length -> delta) on all extreme samples (unfiltered):')
ols_all = linregress(extreme_df['length'].values, extreme_df['delta'].values)
print(f'  n = {len(extreme_df)}')
print(f'  slope = {ols_all.slope:+.6e}')
print(f'  intercept = {ols_all.intercept:+.6e}')
print(f'  R² = {ols_all.rvalue ** 2:.6f}')
print(f'  p = {ols_all.pvalue:.6e}')
