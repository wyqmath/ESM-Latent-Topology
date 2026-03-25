#!/usr/bin/env python3
# -- Project root setup --
import os as _os
from pathlib import Path as _Path
_os.chdir(_Path(__file__).resolve().parents[2])
"""
Figure 5: Density Inversion Analysis (4 panels)
Panel A: 3D density landscape over UMAP plane
Panel B: Density violin plot by category (7-class)
Panel C: Fold-change vs Astral95 (Mann-Whitney U, exact p-values)
Panel D: Density CDF by category (cumulative distribution)
Output: manuscript/figure5.png
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from pathlib import Path
from scipy.interpolate import griddata
from scipy.stats import mannwhitneyu, ks_2samp

print("=" * 70)
print("Figure 5: Density Inversion")
print("=" * 70)

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
print("\n[1/4] Loading data...")
density = np.load('data/density/density_values.npy')
umap_coords = np.load('data/umap/umap_embeddings_2d.npy')
metadata = pd.read_csv('data/metadata_final_with_en.csv')

if len(density) != len(umap_coords):
    raise ValueError(f"Shape mismatch: density={len(density)}, umap={len(umap_coords)}")

metadata['density'] = density
log_density = np.log10(density)
print(f"  Sequences: {len(metadata)}")

# 7-class mapping and colors (identical to Fig4)
CAT7_MAP = {
    'anchor': 'Anchor', 'astral95': 'Astral95', 'integrable': 'Integrable',
    'random': 'Random', 'fold_switching': 'Fold-switching',
    'idp': 'IDP', 'knotted': 'Knotted',
}
COLORS_7 = {
    'Astral95': '#bdbdbd', 'Anchor': '#1f77b4', 'Random': '#2ca02c',
    'Integrable': '#17becf', 'IDP': '#9467bd',
    'Fold-switching': '#d62728', 'Knotted': '#ff7f0e',
}
PLOT_ORDER = ['Random', 'Integrable', 'Anchor', 'Astral95', 'Fold-switching', 'IDP', 'Knotted']

metadata['cat7'] = metadata['subcategory'].map(CAT7_MAP)
metadata = metadata[metadata['cat7'].notna()].copy()
print(f"  After 7-class filter: {len(metadata)}")
for cat in PLOT_ORDER:
    print(f"    {cat}: {(metadata['cat7'] == cat).sum()}")

# Stats per category
stats = metadata.groupby('cat7')['density'].agg(['mean', 'median', 'std', 'count'])

# Fold-change vs Astral95 (reference: broadest structural coverage, n=8292)
ref_cat = 'Astral95'
ref_mean = stats.loc[ref_cat, 'mean']
fold_change = stats['mean'] / ref_mean

# Mann-Whitney U tests (each category vs Astral95)
mw_results = {}
ref_density = metadata[metadata['cat7'] == ref_cat]['density'].values
for cat in PLOT_ORDER:
    if cat == ref_cat:
        continue
    cat_density = metadata[metadata['cat7'] == cat]['density'].values
    u_stat, p_val = mannwhitneyu(cat_density, ref_density, alternative='two-sided')
    mw_results[cat] = (u_stat, p_val)

# ============ Figure ============
print("\n[2/4] Rendering figure panels...")
fig = plt.figure(figsize=(24, 6.4))
gs = fig.add_gridspec(1, 4, wspace=0.40)

# ============ Panel A: 3D density landscape ============
ax = fig.add_subplot(gs[0, 0], projection='3d')

x = umap_coords[:, 0]
y = umap_coords[:, 1]
xi = np.linspace(x.min(), x.max(), 100)
yi = np.linspace(y.min(), y.max(), 100)
XI, YI = np.meshgrid(xi, yi)

ZI_linear = griddata((x, y), log_density, (XI, YI), method='linear')
ZI_nearest = griddata((x, y), log_density, (XI, YI), method='nearest')
ZI = np.where(np.isnan(ZI_linear), ZI_nearest, ZI_linear)
print(f"  Panel A grid: {XI.shape[0]}x{XI.shape[1]}")
print(f"  Panel A log10(density) range: [{np.nanmin(ZI):.3f}, {np.nanmax(ZI):.3f}]")
print(f"  Panel A NaN fill ratio: {np.isnan(ZI_linear).mean():.3%} (linear -> nearest)")

surface = ax.plot_surface(
    XI,
    YI,
    ZI,
    cmap='viridis',
    linewidth=0,
    antialiased=True,
    shade=True,
    alpha=0.96,
)
# Add subtle base-plane shadow for depth perception
z_min = float(np.nanmin(ZI))
z_max = float(np.nanmax(ZI))
ax.contourf(
    XI,
    YI,
    ZI,
    zdir='z',
    offset=z_min - 0.06,
    levels=12,
    cmap='Greys',
    alpha=0.22,
)
ax.set_zlim(z_min - 0.06, z_max)
ax.view_init(elev=35, azim=-60)

# white background for 3D panel
ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
ax.xaxis.line.set_color((1.0, 1.0, 1.0, 0.0))
ax.yaxis.line.set_color((1.0, 1.0, 1.0, 0.0))
ax.zaxis.line.set_color((1.0, 1.0, 1.0, 0.0))

try:
    ax.set_box_aspect((1, 1, 1), zoom=1.15)
except TypeError:
    ax.set_box_aspect((1, 1, 1))

# UMAP coordinates are mainly for geometric context in this panel;
# hide axis ticks/labels to avoid visual clutter and overlap with colorbar.
ax.set_xlabel('')
ax.set_ylabel('')
ax.set_zlabel('')
ax.set_xticks([])
ax.set_yticks([])
ax.set_zticks([])
ax.tick_params(axis='x', pad=0)
ax.tick_params(axis='y', pad=0)
ax.tick_params(axis='z', pad=0)
ax.set_title('a', fontsize=12, fontweight='bold', pad=8)

cax_a = inset_axes(
    ax,
    width="3.5%",
    height="52%",
    loc='lower left',
    bbox_to_anchor=(0.96, 0.24, 1, 1),
    bbox_transform=ax.transAxes,
    borderpad=0,
)
cb = fig.colorbar(surface, cax=cax_a)
cb.ax.tick_params(labelsize=6, pad=1)

# ============ Panel B: Violin plot (7-class) ============
ax = fig.add_subplot(gs[0, 1])

violin_data = []
violin_labels = []
violin_colors = []
for cat in PLOT_ORDER:
    vals = metadata.loc[metadata['cat7'] == cat, 'density'].values
    violin_data.append(vals)
    n = len(vals)
    violin_labels.append(f'{cat}\n(n={n})')
    violin_colors.append(COLORS_7[cat])

positions = list(range(len(PLOT_ORDER)))
parts = ax.violinplot(violin_data, positions=positions,
                      showmeans=False, showmedians=False, showextrema=False)

for i, pc in enumerate(parts['bodies']):
    pc.set_facecolor(violin_colors[i])
    pc.set_alpha(0.7)

# Draw mean (black dashed) and median (red solid) manually
for i, vals in enumerate(violin_data):
    mean_val = vals.mean()
    median_val = np.median(vals)
    ax.hlines(median_val, i - 0.3, i + 0.3, colors='red', linewidths=1.2, zorder=4)
    ax.hlines(mean_val, i - 0.3, i + 0.3, colors='black', linewidths=1.2,
              linestyles='dashed', zorder=4)

# μ label above violin (at actual max of distribution)
for i, vals in enumerate(violin_data):
    mean_val = vals.mean()
    top = vals.max()
    ax.text(i, top + (top * 0.02), f'μ={mean_val:.4f}', ha='center', va='bottom',
            fontsize=5.5, fontweight='bold', color=violin_colors[i])

ax.set_xticks(positions)
ax.set_xticklabels(violin_labels, fontsize=6.3)
ax.set_ylabel('Manifold Density')
ax.set_title('b', fontsize=12, fontweight='bold', loc='center', pad=10)
ax.grid(True, alpha=0.2, axis='y', linewidth=0.5)
ax.set_box_aspect(1)

legend_handles_b = [
    Line2D([0], [0], color='black', linewidth=1.2, linestyle='dashed', label='Mean'),
    Line2D([0], [0], color='red', linewidth=1.2, linestyle='solid', label='Median'),
]
ax.legend(handles=legend_handles_b, loc='upper left', frameon=True, fontsize=7,
          framealpha=0.9, edgecolor='gray')

# ============ Panel C: Fold-change dot plot (Cleveland style) ============
ax = fig.add_subplot(gs[0, 2])

fc_vals = [fold_change.get(cat, float('nan')) for cat in PLOT_ORDER]
y_pos = list(range(len(PLOT_ORDER)))

ax.axvline(1.0, color='black', linewidth=0.8, linestyle='--', zorder=1)

for y0 in y_pos:
    ax.hlines(y0, min(fc_vals) - 0.05, max(fc_vals) + 0.25,
              colors='gray', linewidths=0.4, linestyles='dotted', zorder=0)

for i, cat in enumerate(PLOT_ORDER):
    ax.scatter(fc_vals[i], i, color=COLORS_7[cat], s=80, zorder=3, edgecolors='black',
               linewidths=0.5)

for i, cat in enumerate(PLOT_ORDER):
    val = fc_vals[i]
    if cat in mw_results:
        p = mw_results[cat][1]
        p_str = f'p={p:.2e}'
    else:
        p_str = 'ref'
    ax.text(val + 0.018, i, f'{val:.2f}x  {p_str}',
            va='center', ha='left', fontsize=6.5)

ax.set_yticks(y_pos)
ax.set_yticklabels(PLOT_ORDER, fontsize=8)
ax.set_xlabel('Fold-change vs Astral95 (Mann-Whitney U)')
ax.set_title('c', fontsize=12, fontweight='bold', loc='center', pad=10)
ax.set_xlim(min(fc_vals) - 0.08, max(fc_vals) + 0.35)
ax.grid(False)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.set_box_aspect(1)

# ============ Panel D: Density CDF by category ============
ax = fig.add_subplot(gs[0, 3])

for cat in PLOT_ORDER:
    vals = np.sort(metadata.loc[metadata['cat7'] == cat, 'density'].values)
    cdf = np.arange(1, len(vals) + 1) / len(vals)
    ax.plot(vals, cdf, color=COLORS_7[cat], label=cat, linewidth=1.5)

ax.set_xlabel('Manifold Density')
ax.set_ylabel('Cumulative Fraction')
ax.set_title('d', fontsize=12, fontweight='bold', loc='center', pad=10)
ax.legend(fontsize=7, framealpha=0.9, loc='lower right')
ax.grid(True, alpha=0.2, linewidth=0.5)
ax.text(0.02, 0.97,
        'Right shift = higher density\n(Mann-Whitney U vs Astral95)',
        transform=ax.transAxes, fontsize=6.5, va='top',
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.85,
                  edgecolor='gray', linewidth=0.7))
ax.set_box_aspect(1)

fig.subplots_adjust(left=0.02, right=0.98, top=0.91, bottom=0.18, wspace=0.48)

print("\n[3/4] Saving figure...")
Path('manuscript').mkdir(parents=True, exist_ok=True)
plt.savefig('manuscript/figure5.png', dpi=600, bbox_inches='tight', facecolor='white')
plt.close()
print("  Saved: manuscript/figure5.png")

# ============ Summary Statistics ============
print("\n[4/4] Figure statistics")
print("\n" + "=" * 70)
print("FIGURE 5 STATISTICS")
print("=" * 70)

print(f"\nDensity by category (7-class):")
for cat in PLOT_ORDER:
    s = stats.loc[cat]
    print(f"  {cat:<16s}: mean={s['mean']:.6f}, median={s['median']:.6f}, "
          f"std={s['std']:.6f}, n={int(s['count'])}")

print(f"\nFold-change vs Astral95 (ref mean={ref_mean:.6f}, Mann-Whitney U test):")
for cat in PLOT_ORDER:
    fc = fold_change.get(cat, float('nan'))
    if cat in mw_results:
        u, p = mw_results[cat]
        print(f"  {cat:<16s}: {fc:.4f}x  (U={u:.0f}, p={p:.2e})")
    else:
        print(f"  {cat:<16s}: {fc:.4f}x  (reference)")

print(f"\nPanel D CDF analysis (Kolmogorov-Smirnov test vs Astral95):")
for cat in PLOT_ORDER:
    if cat == ref_cat:
        med = np.median(ref_density)
        print(f"  {cat:<16s}: median={med:.6f}  (reference)")
        continue
    cat_density = metadata[metadata['cat7'] == cat]['density'].values
    ks_stat, ks_p = ks_2samp(cat_density, ref_density)
    med = np.median(cat_density)
    direction = 'right-shift' if med > np.median(ref_density) else 'left-shift'
    print(f"  {cat:<16s}: median={med:.6f}  KS={ks_stat:.4f}, p={ks_p:.2e}  [{direction}]")
