# -- Project root setup --
import os as _os
from pathlib import Path as _Path
_os.chdir(_Path(__file__).resolve().parents[2])
"""
Figure 3: Decoding Latent Axes (4 panels)
Panel A: PCA colored by GRAVY (hydropathy)
Panel B: PCA colored by pI (isoelectric point)
Panel C: PCA colored by structural class (α/β from ASTRAL95)
Panel D: Correlation summary bar chart
Output: manuscript/figure3.png
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from Bio import SeqIO
from Bio.SeqUtils.ProtParam import ProteinAnalysis
from scipy.stats import spearmanr

print("=" * 70)
print("Figure 3: Decoding Latent Axes")
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
print("\nLoading data...")
pca_2d = np.load('data/pca/pca_embeddings_2d.npy')
metadata = pd.read_csv('data/metadata_final_with_en.csv')
explained_var = np.load('data/pca/explained_variance.npy')
print(f"  Sequences: {len(pca_2d)}")

# Compute physicochemical properties from FASTA
print("Computing physicochemical properties...")
gravy_scores = []
pi_scores = []
arom_scores = []
for record in SeqIO.parse('data/all_sequences_final.fasta', 'fasta'):
    seq = str(record.seq).replace('X', '').replace('U', 'C').replace('B', 'N').replace('Z', 'Q')
    try:
        pa = ProteinAnalysis(seq)
        gravy_scores.append(pa.gravy())
        pi_scores.append(pa.isoelectric_point())
        arom_scores.append(pa.aromaticity())
    except:
        gravy_scores.append(np.nan)
        pi_scores.append(np.nan)
        arom_scores.append(np.nan)

gravy_scores = np.array(gravy_scores)
pi_scores = np.array(pi_scores)
arom_scores = np.array(arom_scores)

valid_g = ~np.isnan(gravy_scores)
valid_p = ~np.isnan(pi_scores)
valid_a = ~np.isnan(arom_scores)

print(f"  Valid GRAVY: {valid_g.sum()}, pI: {valid_p.sum()}, Arom: {valid_a.sum()}")

# Spearman correlations
rho_pc1_gravy, p_pc1_gravy = spearmanr(pca_2d[valid_g, 0], gravy_scores[valid_g])
rho_pc2_gravy, p_pc2_gravy = spearmanr(pca_2d[valid_g, 1], gravy_scores[valid_g])
rho_pc1_pi, p_pc1_pi = spearmanr(pca_2d[valid_p, 0], pi_scores[valid_p])
rho_pc2_pi, p_pc2_pi = spearmanr(pca_2d[valid_p, 1], pi_scores[valid_p])
rho_pc1_arom, p_pc1_arom = spearmanr(pca_2d[valid_a, 0], arom_scores[valid_a])
rho_pc2_arom, p_pc2_arom = spearmanr(pca_2d[valid_a, 1], arom_scores[valid_a])

# Structural class from ASTRAL95
metadata['struct_class'] = metadata['seq_id'].apply(
    lambda x: x.split('|')[1] if 'astral95' in x and len(x.split('|')) > 1 else None
)
struct_classes = ['all-alpha', 'all-beta', 'alpha+beta', 'alpha/beta']
struct_colors = {'all-alpha': '#e41a1c', 'all-beta': '#377eb8',
                 'alpha+beta': '#4daf4a', 'alpha/beta': '#984ea3'}

# Length correlation for summary
lengths = metadata['length'].values
rho_pc1_len, p_pc1_len = spearmanr(pca_2d[:, 0], lengths)
rho_pc2_len, p_pc2_len = spearmanr(pca_2d[:, 1], lengths)

# ============ Figure: 1×4 layout ============
fig, axes = plt.subplots(1, 4, figsize=(24, 5.5))

# Panel A: GRAVY gradient
ax = axes[0]
gravy_lo, gravy_hi = np.nanpercentile(gravy_scores[valid_g], [2, 98])
sc = ax.scatter(pca_2d[valid_g, 0], pca_2d[valid_g, 1],
                c=gravy_scores[valid_g], cmap='coolwarm',
                vmin=gravy_lo, vmax=gravy_hi,
                s=3, alpha=0.7, edgecolors='none', rasterized=True)
cbar = plt.colorbar(sc, ax=ax, shrink=0.8)
cbar.set_label('GRAVY Score', fontsize=8)
cbar.ax.tick_params(labelsize=7)
ax.set_xlabel(f'PC1 ({explained_var[0]*100:.1f}% var)')
ax.set_ylabel(f'PC2 ({explained_var[1]*100:.1f}% var)')
ax.set_title('a', fontsize=12, fontweight='bold', loc='center', pad=10)
ax.grid(True, alpha=0.2, linewidth=0.5)
props = dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='gray', linewidth=0.8)
ax.text(0.98, 0.02,
        f'ρ(PC1, GRAVY)={rho_pc1_gravy:.3f}\nρ(PC2, GRAVY)={rho_pc2_gravy:.3f}\n'
        f'Color: 2nd–98th pctl [{gravy_lo:.2f}, {gravy_hi:.2f}]',
        transform=ax.transAxes, fontsize=6.5, va='bottom', ha='right', bbox=props)

# Panel B: pI gradient
ax = axes[1]
pi_lo, pi_hi = np.nanpercentile(pi_scores[valid_p], [2, 98])
sc = ax.scatter(pca_2d[valid_p, 0], pca_2d[valid_p, 1],
                c=pi_scores[valid_p], cmap='coolwarm',
                vmin=pi_lo, vmax=pi_hi,
                s=3, alpha=0.7, edgecolors='none', rasterized=True)
cbar = plt.colorbar(sc, ax=ax, shrink=0.8)
cbar.set_label('Isoelectric Point (pI)', fontsize=8)
cbar.ax.tick_params(labelsize=7)
ax.set_xlabel(f'PC1 ({explained_var[0]*100:.1f}% var)')
ax.set_ylabel(f'PC2 ({explained_var[1]*100:.1f}% var)')
ax.set_title('b', fontsize=12, fontweight='bold', loc='center', pad=10)
ax.grid(True, alpha=0.2, linewidth=0.5)
ax.text(0.98, 0.02,
        f'ρ(PC1, pI)={rho_pc1_pi:.3f}\nρ(PC2, pI)={rho_pc2_pi:.3f}\n'
        f'Color: 2nd–98th pctl [{pi_lo:.2f}, {pi_hi:.2f}]',
        transform=ax.transAxes, fontsize=6.5, va='bottom', ha='right', bbox=props)

# Panel C: Structural class
ax = axes[2]
non_astral = metadata['struct_class'].isna().values
ax.scatter(pca_2d[non_astral, 0], pca_2d[non_astral, 1],
           c='lightgray', s=1, alpha=0.2, edgecolors='none', rasterized=True)
for sc_name in struct_classes:
    mask = (metadata['struct_class'] == sc_name).values
    n = mask.sum()
    if n > 0:
        ax.scatter(pca_2d[mask, 0], pca_2d[mask, 1],
                   c=struct_colors[sc_name],
                   s=5, alpha=0.6, edgecolors='none', rasterized=True)
ax.set_xlabel(f'PC1 ({explained_var[0]*100:.1f}% var)')
ax.set_ylabel(f'PC2 ({explained_var[1]*100:.1f}% var)')
ax.set_title('c', fontsize=12, fontweight='bold', loc='center', pad=10)
ax.grid(True, alpha=0.2, linewidth=0.5)

# Uniform-size legend handles for Panel C
from matplotlib.lines import Line2D
legend_handles_c = [Line2D([0], [0], marker='o', color='w',
                           markerfacecolor='lightgray', markersize=6,
                           label='Other', linestyle='None')]
for sc_name in struct_classes:
    n = (metadata['struct_class'] == sc_name).sum()
    if n > 0:
        legend_handles_c.append(Line2D([0], [0], marker='o', color='w',
                                       markerfacecolor=struct_colors[sc_name], markersize=6,
                                       label=f'{sc_name} (n={n})', linestyle='None'))
ax.legend(handles=legend_handles_c, loc='upper left', frameon=True, fontsize=6,
          framealpha=0.9, edgecolor='gray')

# Panel D: Correlation summary bar chart
ax = axes[3]
labels = ['Length', 'GRAVY', 'pI', 'Arom']
pc1_vals = [rho_pc1_len, rho_pc1_gravy, rho_pc1_pi, rho_pc1_arom]
pc2_vals = [rho_pc2_len, rho_pc2_gravy, rho_pc2_pi, rho_pc2_arom]

x = np.arange(len(labels))
width = 0.35
bars1 = ax.bar(x - width/2, pc1_vals, width, label='ρ(PC1, ·)', color='#3498db', alpha=0.8)
bars2 = ax.bar(x + width/2, pc2_vals, width, label='ρ(PC2, ·)', color='#e74c3c', alpha=0.8)

ax.axhline(0, color='black', linewidth=0.8)
ax.set_ylabel('Spearman ρ')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.set_title('d', fontsize=12, fontweight='bold', loc='center', pad=10)
ax.legend(loc='upper right', frameon=True, fontsize=7, framealpha=0.9)
ax.grid(True, alpha=0.2, axis='y', linewidth=0.5)
ax.set_ylim(-0.5, 0.5)

# Add value labels
for bar in bars1:
    h = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., h + 0.01 * np.sign(h),
            f'{h:.3f}', ha='center', va='bottom' if h >= 0 else 'top', fontsize=6)
for bar in bars2:
    h = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., h + 0.01 * np.sign(h),
            f'{h:.3f}', ha='center', va='bottom' if h >= 0 else 'top', fontsize=6)

plt.tight_layout()
plt.savefig('manuscript/figure3.png', dpi=600, bbox_inches='tight')
plt.close()

# ============ Summary Statistics (for figure caption) ============
print("\n" + "=" * 70)
print("FIGURE 3 — STATISTICS FOR CAPTION")
print("=" * 70)

print(f"\n{'─'*70}")
print(f"[Panel A] PCA colored by GRAVY (hydropathy)")
print(f"{'─'*70}")
print(f"  n = {valid_g.sum()} (valid GRAVY scores)")
print(f"  GRAVY range: [{np.nanmin(gravy_scores[valid_g]):.3f}, {np.nanmax(gravy_scores[valid_g]):.3f}]")
print(f"  GRAVY mean ± std: {np.nanmean(gravy_scores[valid_g]):.3f} ± {np.nanstd(gravy_scores[valid_g]):.3f}")
print(f"  Color clip: 2nd–98th pctl [{gravy_lo:.3f}, {gravy_hi:.3f}]")
print(f"  ρ(PC1, GRAVY) = {rho_pc1_gravy:.4f}, p = {p_pc1_gravy:.2e}")
print(f"  ρ(PC2, GRAVY) = {rho_pc2_gravy:.4f}, p = {p_pc2_gravy:.2e}")
print(f"  → PC2 shows moderate negative correlation with hydropathy")

print(f"\n{'─'*70}")
print(f"[Panel B] PCA colored by pI (isoelectric point)")
print(f"{'─'*70}")
print(f"  n = {valid_p.sum()} (valid pI scores)")
print(f"  pI range: [{np.nanmin(pi_scores[valid_p]):.2f}, {np.nanmax(pi_scores[valid_p]):.2f}]")
print(f"  pI mean ± std: {np.nanmean(pi_scores[valid_p]):.2f} ± {np.nanstd(pi_scores[valid_p]):.2f}")
print(f"  Color clip: 2nd–98th pctl [{pi_lo:.2f}, {pi_hi:.2f}]")
print(f"  ρ(PC1, pI) = {rho_pc1_pi:.4f}, p = {p_pc1_pi:.2e}")
print(f"  ρ(PC2, pI) = {rho_pc2_pi:.4f}, p = {p_pc2_pi:.2e}")
print(f"  → PC2 shows weak positive correlation with pI")

print(f"\n{'─'*70}")
print(f"[Panel C] PCA colored by structural class (ASTRAL95 SCOP)")
print(f"{'─'*70}")
for sc_name in struct_classes:
    n = (metadata['struct_class'] == sc_name).sum()
    print(f"  {sc_name:<15s}: n = {n}")
print(f"  {'Other (gray)':<15s}: n = {metadata['struct_class'].isna().sum()}")
print(f"  → Horseshoe manifold is organized by secondary structure type")

print(f"\n{'─'*70}")
print(f"[Panel D] Spearman correlation summary (PC1 & PC2 vs properties)")
print(f"{'─'*70}")
print(f"  {'Property':<15s} {'ρ(PC1)':<12s} {'p(PC1)':<12s} {'ρ(PC2)':<12s} {'p(PC2)':<12s}")
print(f"  {'─'*55}")
print(f"  {'Length':<15s} {rho_pc1_len:<12.4f} {p_pc1_len:<12.2e} {rho_pc2_len:<12.4f} {p_pc2_len:<12.2e}")
print(f"  {'GRAVY':<15s} {rho_pc1_gravy:<12.4f} {p_pc1_gravy:<12.2e} {rho_pc2_gravy:<12.4f} {p_pc2_gravy:<12.2e}")
print(f"  {'pI':<15s} {rho_pc1_pi:<12.4f} {p_pc1_pi:<12.2e} {rho_pc2_pi:<12.4f} {p_pc2_pi:<12.2e}")
print(f"  {'Aromaticity':<15s} {rho_pc1_arom:<12.4f} {p_pc1_arom:<12.2e} {rho_pc2_arom:<12.4f} {p_pc2_arom:<12.2e}")
print(f"  Variance explained: PC1={explained_var[0]*100:.2f}%, PC2={explained_var[1]*100:.2f}%")

print(f"\n{'─'*70}")
print(f"KEY MESSAGES:")
print(f"  1. PC1 shows no strong correlation with any single property (all |ρ| < 0.17)")
print(f"     → PC1 encodes a complex, non-trivial sequence grammar")
print(f"  2. PC2 correlates with Aromaticity (ρ={rho_pc2_arom:.3f}) and GRAVY (ρ={rho_pc2_gravy:.3f})")
print(f"     → PC2 captures biochemical composition gradients")
print(f"  3. Structural classes (α/β) organize along the horseshoe")
print(f"     → The manifold reflects secondary structure preferences")
print(f"{'─'*70}")

print(f"\nSaved: manuscript/figure3.png")
print("=" * 70)
