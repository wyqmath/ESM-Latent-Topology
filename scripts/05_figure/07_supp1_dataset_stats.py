# -- Project root setup --
import os as _os
from pathlib import Path as _Path
_os.chdir(_Path(__file__).resolve().parents[2])
"""
Supplementary Figure 1: Dataset Statistics
===========================================

Input:
- data/metadata_final_with_en.csv (11068 sequences)
- data/all_sequences_final.fasta (11068 sequences)

Output:
- manuscript/supp1.png (DPI=600)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde, spearmanr
from Bio.SeqUtils.ProtParam import ProteinAnalysis
from collections import Counter
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import os

# White background, no grid background
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = 'white'
plt.rcParams['savefig.facecolor'] = 'white'

os.makedirs('manuscript', exist_ok=True)

# Load metadata
print("Loading metadata...")
metadata = pd.read_csv('data/metadata_final_with_en.csv')
print(f"Total sequences: {len(metadata)}")

# 7-category assignment (consistent with Figure 2 / Figure 4)
def assign_category(row):
    subcat = row['subcategory']

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

metadata['category7'] = metadata.apply(assign_category, axis=1)

# Color scheme consistent with Figure 2
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

plot_order = ['astral95', 'anchor', 'idp', 'knotted', 'random', 'integrable', 'fold_switching']

# Load sequences
print("Loading sequences...")
sequences = {}
with open('data/all_sequences_final.fasta', 'r') as f:
    seq_id = None
    seq = []
    for line in f:
        line = line.strip()
        if line.startswith('>'):
            if seq_id is not None:
                sequences[seq_id] = ''.join(seq)
            seq_id = line[1:].split()[0]
            seq = []
        else:
            seq.append(line)
    if seq_id is not None:
        sequences[seq_id] = ''.join(seq)
print(f"Loaded {len(sequences)} sequences")

# Compute GRAVY and pI in metadata order (reuse Fig3 cleaning rules)
print("Computing GRAVY and pI...")
gravy_scores = []
pi_scores = []

for sid in metadata['seq_id'].values:
    seq = sequences.get(sid, '')
    clean_seq = seq.replace('X', '').replace('U', 'C').replace('B', 'N').replace('Z', 'Q')

    if not clean_seq:
        gravy_scores.append(np.nan)
        pi_scores.append(np.nan)
        continue

    try:
        pa = ProteinAnalysis(clean_seq)
        gravy_scores.append(pa.gravy())
        pi_scores.append(pa.isoelectric_point())
    except Exception:
        gravy_scores.append(np.nan)
        pi_scores.append(np.nan)

metadata['gravy'] = gravy_scores
metadata['pi'] = pi_scores
valid_physchem = metadata['gravy'].notna() & metadata['pi'].notna()
print(f"Valid physicochemical points: {valid_physchem.sum()}")

# Layout: 2x2 with square panels
fig = plt.figure(figsize=(16, 14))
gs = fig.add_gridspec(2, 2, wspace=0.28, hspace=0.32)

# ============ Panel A: Sequence length KDE (integrable on separate y-axis) ============
ax1 = fig.add_subplot(gs[0, 0])
ax1_int = ax1.twinx()
ax1.set_box_aspect(1)
ax1_int.set_box_aspect(1)

# Use percentile window for display to avoid ultra-long tail dominating the axis
x_max_plot = metadata['length'].quantile(0.99)
x_range = np.linspace(0, x_max_plot, 500)

for cat in plot_order:
    cat_data = metadata[metadata['category7'] == cat]['length'].values
    if len(cat_data) <= 5:
        continue

    kde = gaussian_kde(cat_data, bw_method='scott')
    density = kde(x_range)

    if cat == 'integrable':
        ax1_int.plot(
            x_range,
            density,
            color=COLORS[cat],
            linewidth=2.1,
            alpha=0.95,
            linestyle='--',
            label=f"{DISPLAY_NAMES[cat]} (n={len(cat_data)})"
        )
        ax1_int.fill_between(x_range, density, alpha=0.12, color=COLORS[cat])
    else:
        ax1.plot(
            x_range,
            density,
            color=COLORS[cat],
            linewidth=1.8,
            alpha=0.85,
            label=f"{DISPLAY_NAMES[cat]} (n={len(cat_data)})"
        )
        ax1.fill_between(x_range, density, alpha=0.15, color=COLORS[cat])

mean_length = metadata['length'].mean()
median_length = metadata['length'].median()
ax1.axvline(mean_length, color='red', linestyle='--', linewidth=1.2, alpha=0.6)
ax1.axvline(median_length, color='navy', linestyle=':', linewidth=1.2, alpha=0.6)
ylim = ax1.get_ylim()
ax1.text(mean_length + 3, ylim[1] * 0.93, f'Mean={mean_length:.0f}', fontsize=7.5, color='red')
ax1.text(median_length + 3, ylim[1] * 0.85, f'Median={median_length:.0f}', fontsize=7.5, color='navy')

ax1.set_xlabel('Sequence Length (residues)', fontsize=11)
ax1.set_ylabel('Density (others)', fontsize=10)
ax1_int.set_ylabel('Density (Integrable)', fontsize=10, color=COLORS['integrable'])
ax1_int.tick_params(axis='y', labelcolor=COLORS['integrable'])
ax1.set_title('a', fontsize=13, fontweight='bold', loc='center', pad=8)
ax1.grid(True, alpha=0.2)
ax1.set_xlim(0, x_max_plot)

# merge legends from left and right axes
handles1, labels1 = ax1.get_legend_handles_labels()
handles2, labels2 = ax1_int.get_legend_handles_labels()
ax1.legend(
    handles1 + handles2,
    labels1 + labels2,
    fontsize=6,
    loc='upper right',
    frameon=True,
    facecolor='white',
    edgecolor='gray',
    framealpha=1,
    handlelength=1.5,
    labelspacing=0.3,
    borderpad=0.4
)

# ============ Panel B: pI vs GRAVY 2D landscape (7-class colored + KDE contours) ============
ax2 = fig.add_subplot(gs[0, 1])
ax2.set_box_aspect(1)

physchem_data = metadata.loc[valid_physchem, ['category7', 'gravy', 'pi']]

if len(physchem_data) > 0:
    # Robust plotting window; enlarge Y window to avoid visual truncation
    x_lo, x_hi = np.quantile(physchem_data['gravy'].values, [0.01, 0.99])
    y_lo, y_hi = np.quantile(physchem_data['pi'].values, [0.005, 0.995])
    x_pad = max(0.04, 0.04 * (x_hi - x_lo))
    y_pad = max(0.04, 0.04 * (y_hi - y_lo))

    x_min_plot, x_max_plot_b = x_lo - x_pad, x_hi + x_pad
    y_min_plot, y_max_plot = y_lo - y_pad, y_hi + y_pad

    x_grid = np.linspace(x_min_plot, x_max_plot_b, 120)
    y_grid = np.linspace(y_min_plot, y_max_plot, 120)
    xx, yy = np.meshgrid(x_grid, y_grid)

    for cat in plot_order:
        cat_df = physchem_data[physchem_data['category7'] == cat]
        if len(cat_df) == 0:
            continue

        # Clip display points to plotting window (keep view bounded)
        cat_disp = cat_df[
            (cat_df['gravy'] >= x_min_plot) & (cat_df['gravy'] <= x_max_plot_b) &
            (cat_df['pi'] >= y_min_plot) & (cat_df['pi'] <= y_max_plot)
        ]

        if len(cat_disp) > 0:
            ax2.scatter(
                cat_disp['gravy'],
                cat_disp['pi'],
                s=5,
                alpha=0.18,
                color=COLORS[cat],
                edgecolors='none',
                rasterized=True,
                zorder=2
            )

        # Overlay one category-specific 2D KDE contour (cleaner)
        if len(cat_df) >= 20:
            try:
                values = np.vstack([cat_df['gravy'].values, cat_df['pi'].values])
                kde2d = gaussian_kde(values, bw_method='scott')
                zz = kde2d(np.vstack([xx.ravel(), yy.ravel()])).reshape(xx.shape)
                positive = zz[zz > 0]
                if positive.size > 0:
                    level = np.quantile(positive, 0.90)
                    cs = ax2.contour(
                        xx,
                        yy,
                        zz,
                        levels=[level],
                        colors=[COLORS[cat]],
                        linewidths=1.4,
                        alpha=1.0,
                        zorder=3
                    )
                    for coll in cs.collections:
                        coll.set_clip_on(True)
                        coll.set_clip_path(ax2.patch)
            except Exception:
                pass

    ax2.set_xlim(x_min_plot, x_max_plot_b)
    ax2.set_ylim(y_min_plot, y_max_plot)

# Manual legend so legend colors match contour colors exactly
legend_handles = [
    Line2D([0], [0], color=COLORS[cat], lw=1.8, label=f"{DISPLAY_NAMES[cat]}")
    for cat in plot_order
]
ax2.legend(
    handles=legend_handles,
    fontsize=6,
    loc='upper right',
    frameon=True,
    facecolor='white',
    edgecolor='gray',
    framealpha=1,
    handlelength=1.2,
    labelspacing=0.3,
    borderpad=0.4
)
ax2.set_xlabel('GRAVY', fontsize=11)
ax2.set_ylabel('Isoelectric Point (pI)', fontsize=11)
ax2.set_title('b', fontsize=13, fontweight='bold', loc='center', pad=8)
ax2.grid(True, alpha=0.2)

# ============ Panel C: Category composition ============
ax3 = fig.add_subplot(gs[1, 0])
ax3.set_box_aspect(1)

cat_counts = metadata['category7'].value_counts()
ordered_cats = [c for c in plot_order if c in cat_counts.index]
ordered_values = [cat_counts[c] for c in ordered_cats]
ordered_colors = [COLORS[c] for c in ordered_cats]
total = sum(ordered_values)

explode = [0.02 if v / total > 0.05 else 0.06 for v in ordered_values]

# Shrink pie to leave room for legend
ax3.pie(
    ordered_values,
    colors=ordered_colors,
    startangle=90,
    explode=explode,
    radius=1.00,
    center=(0.0, 0.0),
    wedgeprops=dict(linewidth=0.5, edgecolor='white')
)
ax3.set_title('c', fontsize=13, fontweight='bold', loc='center', pad=8)

# Keep symmetric square bounds and remove ticks
ax3.set_xlim(-1.2, 1.2)
ax3.set_ylim(-1.2, 1.2)
ax3.set_xticks([])
ax3.set_yticks([])

# Legend labels include N and percentage
legend_handles_c = []
for i, c in enumerate(ordered_cats):
    n_val = ordered_values[i]
    pct_val = (n_val / total) * 100
    label_str = f"{DISPLAY_NAMES[c]} (N={n_val}, {pct_val:.1f}%)"
    legend_handles_c.append(Patch(facecolor=COLORS[c], edgecolor='none', label=label_str))

# 2-row, 4-column compact legend
ax3.legend(
    handles=legend_handles_c,
    loc='upper center',
    bbox_to_anchor=(0.5, 0.02),
    ncol=4,
    fontsize=6,
    frameon=True,
    facecolor='white',
    edgecolor='gray',
    framealpha=1.0,
    columnspacing=0.5,
    handletextpad=0.3,
    handlelength=1.0,
    handleheight=1.0,
    borderpad=0.4,
    labelspacing=0.3
)

# ============ Panel D: Amino acid composition (grouped bars) ============
ax4 = fig.add_subplot(gs[1, 1])
ax4.set_box_aspect(1)

aa_composition = {}
for cat in plot_order:
    cat_seqs = metadata[metadata['category7'] == cat]['seq_id'].values
    all_aa = []
    for sid in cat_seqs:
        seq = sequences.get(sid)
        if seq:
            clean_seq = seq.replace('X', '').replace('U', 'C').replace('B', 'N').replace('Z', 'Q')
            all_aa.extend(list(clean_seq))

    aa_counts = Counter(all_aa)
    total_aa = sum(aa_counts.values())
    if total_aa > 0:
        aa_composition[cat] = {aa: count / total_aa * 100 for aa, count in aa_counts.items()}

standard_aa = 'ACDEFGHIKLMNPQRSTVWY'
x = np.arange(len(standard_aa))
active_cats = [cat for cat in plot_order if cat in aa_composition]
n_cats = len(active_cats)
width = 0.84 / max(n_cats, 1)
offset = -width * (n_cats - 1) / 2

for i, cat in enumerate(active_cats):
    freqs = [aa_composition[cat].get(aa, 0) for aa in standard_aa]
    ax4.bar(
        x + offset + i * width,
        freqs,
        width,
        color=COLORS[cat],
        alpha=0.88,
        label=DISPLAY_NAMES[cat]
    )

ax4.set_xlabel('Amino Acid', fontsize=11)
ax4.set_ylabel('Frequency (%)', fontsize=11)
ax4.set_title('d', fontsize=13, fontweight='bold', loc='center', pad=8)
ax4.set_xticks(x)
ax4.set_xticklabels(list(standard_aa), fontsize=8)
ax4.grid(True, alpha=0.2, axis='y')
ax4.set_xlim(-0.6, len(standard_aa) - 0.4)
ax4.legend(
    fontsize=6,
    loc='upper right',
    ncol=2,
    frameon=True,
    facecolor='white',
    edgecolor='gray',
    framealpha=1,
    handlelength=1.0,
    handletextpad=0.3,
    labelspacing=0.3,
    borderpad=0.4
)

# Save
print("\nSaving figure...")
plt.savefig('manuscript/supp1.png', dpi=600, bbox_inches='tight', facecolor='white')
print("✓ Saved to manuscript/supp1.png")

# Print summary stats for caption writing
length_mean = metadata['length'].mean()
length_std = metadata['length'].std()
length_median = metadata['length'].median()
length_min = metadata['length'].min()
length_max = metadata['length'].max()
length_p01 = metadata['length'].quantile(0.01)
length_p99 = metadata['length'].quantile(0.99)

gravy_valid = metadata.loc[valid_physchem, 'gravy']
pi_valid = metadata.loc[valid_physchem, 'pi']
rho_gp, p_gp = spearmanr(gravy_valid, pi_valid)

# Global amino-acid composition (after cleaning rules)
all_aa_global = []
for sid in metadata['seq_id'].values:
    seq = sequences.get(sid, '')
    clean_seq = seq.replace('X', '').replace('U', 'C').replace('B', 'N').replace('Z', 'Q')
    all_aa_global.extend(list(clean_seq))

global_aa_counts = Counter(all_aa_global)
global_total_aa = sum(global_aa_counts.values())
global_top5_aa = global_aa_counts.most_common(5)

print("\n" + "=" * 70)
print("SUPP1 — STATS FOR CAPTION")
print("=" * 70)
print("[Panel a] Sequence length KDE")
print(f"  n = {len(metadata)}")
print(f"  Length mean ± std = {length_mean:.1f} ± {length_std:.1f}")
print(f"  Length median = {length_median:.1f}")
print(f"  Length range = [{length_min}, {length_max}]")
print(f"  Display window (1st–99th pctl) = [{length_p01:.0f}, {length_p99:.0f}]")

print("\n[Panel b] pI vs GRAVY landscape")
print(f"  Valid physicochemical points = {valid_physchem.sum()} / {len(metadata)}")
print(f"  GRAVY range = [{gravy_valid.min():.3f}, {gravy_valid.max():.3f}], mean ± std = {gravy_valid.mean():.3f} ± {gravy_valid.std():.3f}")
print(f"  pI range = [{pi_valid.min():.3f}, {pi_valid.max():.3f}], mean ± std = {pi_valid.mean():.3f} ± {pi_valid.std():.3f}")
print(f"  Spearman rho(GRAVY, pI) = {rho_gp:.3f}, p = {p_gp:.2e}")

print("\n[Panel c] Category composition")
for cat in plot_order:
    count = (metadata['category7'] == cat).sum()
    if count > 0:
        print(f"  {DISPLAY_NAMES[cat]}: {count} ({count / len(metadata) * 100:.1f}%)")

print("\n[Panel d] Amino-acid composition")
print(f"  Total residues (cleaned) = {global_total_aa}")
print("  Global top-5 amino acids:")
for aa, cnt in global_top5_aa:
    print(f"    {aa}: {cnt / global_total_aa * 100:.2f}%")
print("=" * 70)

plt.close()
