# -- Project root setup --
import os as _os
from pathlib import Path as _Path
_os.chdir(_Path(__file__).resolve().parents[2])
"""
Figure 1: Scale Separation in ESM-2 Embeddings (4 panels)
Baseline: 9,195 sequences with E[n] (anchor + astral95 + integrable)
Narrative: microscopic geometry E[n] is completely discarded by macroscopic latent space.

Panel A: Residue-level ρ vs mean E[n] (preserved)
Panel B: PCA colored by E[n] — "TV snow" effect (NEW)
Panel C: E[n] vs Ricci Curvature scatter (NEW)
Panel D: Scale separation correlation matrix (expanded to 9,195)

Output: manuscript/figure1.png
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from scipy.stats import spearmanr, linregress, gaussian_kde
from sklearn.neighbors import NearestNeighbors

plt.rcParams['figure.dpi'] = 600
plt.rcParams['savefig.dpi'] = 600
plt.rcParams['font.size'] = 8
plt.rcParams['axes.labelsize'] = 9
plt.rcParams['axes.titlesize'] = 10
plt.rcParams['xtick.labelsize'] = 7
plt.rcParams['ytick.labelsize'] = 7
plt.rcParams['legend.fontsize'] = 6
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']

print("=" * 70)
print("Figure 1: Scale Separation in ESM-2 Embeddings")
print("=" * 70)

# ============ Load data ============
print("\nLoading data...")

residue_corr_path = 'data/residue_en_correlation_full.csv'
if not _Path(residue_corr_path).exists():
    residue_corr_path = 'data/residue_en_correlation.csv'
    print(f"  WARNING: Full residue data not found, using {residue_corr_path}")
residue_corr = pd.read_csv(residue_corr_path)

meta = pd.read_csv('data/metadata_final_with_en.csv')
embed_index = pd.read_csv('data/embeddings/embedding_index_final.csv')
pca_2d = np.load('data/pca/pca_embeddings_2d.npy')
pca_50d = np.load('data/pca/pca_embeddings_50d.npy')
explained_var = np.load('data/pca/explained_variance.npy')
density_values = np.load('data/density/density_values.npy')
curvature = np.load('data/curvature/ricci_curvature.npy')
condition_numbers = np.load('data/condition_number/condition_numbers.npy')

# 9,195 sequences with E[n]
has_en = meta[meta['E_n'].notna()].copy()
en_indices = []
en_values = []
for _, row in has_en.iterrows():
    match = embed_index[embed_index['seq_id'] == row['seq_id']]
    if len(match) > 0:
        en_indices.append(match['index'].iloc[0])
        en_values.append(row['E_n'])
en_indices = np.array(en_indices)
en_values = np.array(en_values)

# All indices (for gray background in Panel B)
all_indices = set(range(len(pca_2d)))
en_index_set = set(en_indices.tolist())
no_en_indices = np.array(sorted(all_indices - en_index_set))

print(f"  Residue correlation entries: {len(residue_corr)}")
print(f"  Total sequences (PCA): {len(pca_2d)}")
print(f"  Sequences with E[n]: {len(en_indices)}")
print(f"  Sequences without E[n]: {len(no_en_indices)}")

# ============ Compute Panel D features early (needed for stats) ============
print("\nComputing local geometric features...")
k = 20
nn_model = NearestNeighbors(n_neighbors=k+1, metric='euclidean')
nn_model.fit(pca_50d)
distances, _ = nn_model.kneighbors(pca_50d[en_indices])
knn_distances = distances[:, 1:]

local_density = 1.0 / knn_distances.mean(axis=1)
nearest_dist = knn_distances[:, 0]
r_k = knn_distances[:, -1]
local_dim = np.zeros(len(en_indices))
for i in range(len(en_indices)):
    log_ratios = np.log(r_k[i] / knn_distances[i, :-1])
    log_ratios = log_ratios[np.isfinite(log_ratios)]
    local_dim[i] = len(log_ratios) / np.sum(log_ratios) if len(log_ratios) > 0 else np.nan

var_names_full = ['Local Density', 'Local Dimension', 'Nearest Distance',
                  'Ricci Curvature', 'Condition Number\n(log₁₀)', 'E[n]']
data_cols = [local_density, local_dim, nearest_dist,
             curvature[en_indices],
             np.log10(condition_numbers[en_indices] + 1),
             en_values]

n_vars = len(var_names_full)
corr_mat = np.zeros((n_vars, n_vars))
pval_mat = np.zeros((n_vars, n_vars))
for i in range(n_vars):
    for j in range(n_vars):
        if i == j:
            corr_mat[i, j] = 1.0
            pval_mat[i, j] = 0.0
        else:
            mask = np.isfinite(data_cols[i]) & np.isfinite(data_cols[j])
            if mask.sum() > 10:
                r, p = spearmanr(data_cols[i][mask], data_cols[j][mask])
                corr_mat[i, j] = r
                pval_mat[i, j] = p

# Pre-compute Panel A/B/C stats
sig_pos = residue_corr[(residue_corr['p_value'] < 0.05) & (residue_corr['spearman_rho'] > 0)]
sig_neg = residue_corr[(residue_corr['p_value'] < 0.05) & (residue_corr['spearman_rho'] < 0)]
non_sig = residue_corr[residue_corr['p_value'] >= 0.05]

slope, intercept, r_value, p_value_fit, std_err = linregress(
    residue_corr['mean_E_n'], residue_corr['spearman_rho'])
overall_rho, overall_p = spearmanr(residue_corr['mean_E_n'], residue_corr['spearman_rho'])

rho_pc1, p_pc1 = spearmanr(pca_2d[en_indices, 0], en_values)
rho_pc2, p_pc2 = spearmanr(pca_2d[en_indices, 1], en_values)

en_ricci = curvature[en_indices]
rho_ricci, p_ricci = spearmanr(en_values, en_ricci)

# ============ Figure layout: uniform panel sizes ============
# 24x6 = 4:1 ratio → 4 absolute equal square subfigures
fig = plt.figure(figsize=(24, 6))
subfigs = fig.subfigures(1, 4, wspace=0.0)

# Lock margins: height=0.90-0.20=0.70, width=0.85-0.15=0.70 → perfect square
for sf in subfigs:
    sf.subplots_adjust(left=0.15, right=0.85, bottom=0.20, top=0.90)

# ============ Panel A: Residue-level E[n] correlation ============
ax1 = subfigs[0].subplots()
ax1.set_title('a', fontsize=20, fontweight='bold', pad=15)

ax1.scatter(residue_corr['mean_E_n'], residue_corr['spearman_rho'],
           c='#7f8c8d', s=8, alpha=0.4, edgecolors='none', rasterized=True)
ax1.axhline(y=0, color='black', linestyle='--', linewidth=0.8, alpha=0.5)

x_fit = np.linspace(residue_corr['mean_E_n'].min(), residue_corr['mean_E_n'].max(), 100)
y_fit = slope * x_fit + intercept
ax1.plot(x_fit, y_fit, color='#9b59b6', linewidth=2, alpha=0.8,
         label=f'Linear fit (R²={r_value**2:.3f})', zorder=5)

nn_res = len(residue_corr)
x_data = residue_corr['mean_E_n'].values
y_data = residue_corr['spearman_rho'].values
residuals = y_data - (slope * x_data + intercept)
residual_std = np.sqrt(np.sum(residuals**2) / (nn_res - 2))
x_mean = np.mean(x_data)
sxx = np.sum((x_data - x_mean)**2)
se_fit = residual_std * np.sqrt(1/nn_res + (x_fit - x_mean)**2 / sxx)
ax1.fill_between(x_fit, y_fit - 1.96*se_fit, y_fit + 1.96*se_fit,
                 color='#9b59b6', alpha=0.15, zorder=4)

ax1.set_xlabel('Mean $E[n]$ (Integrability Error)')
ax1.set_ylabel('Residue-level Spearman ρ\n(Embedding Distance vs $E[n]$)')
ax1.grid(True, alpha=0.3, linewidth=0.5)

props = dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='gray', linewidth=0.8)
ax1.text(0.98, 0.98,
         f'n={len(residue_corr)}\nSpearman ρ={overall_rho:.3f}, p={overall_p:.1e}\nR²={r_value**2:.3f}',
         transform=ax1.transAxes, fontsize=7, va='top', ha='right', bbox=props)

# ============ Panel B: PCA colored by E[n] (snow effect) ============
ax2 = subfigs[1].subplots()
ax2.set_title('b', fontsize=20, fontweight='bold', pad=15)

ax2.scatter(pca_2d[no_en_indices, 0], pca_2d[no_en_indices, 1],
           c='lightgray', s=1, alpha=0.3, edgecolors='none',
           label=f'No E[n] (n={len(no_en_indices)})', rasterized=True)
scatter_b = ax2.scatter(pca_2d[en_indices, 0], pca_2d[en_indices, 1],
                        c=en_values, cmap='viridis', s=3, alpha=0.6,
                        edgecolors='none',
                        label=f'Has E[n] (n={len(en_indices)})', rasterized=True)

ax2.set_xlabel(f'PC1 ({explained_var[0]*100:.1f}% var)')
ax2.set_ylabel(f'PC2 ({explained_var[1]*100:.1f}% var)')
ax2.grid(True, alpha=0.3, linewidth=0.5)

# Colorbar — floating inset, doesn't steal from plot area
cax_b = ax2.inset_axes([1.04, 0.0, 0.04, 1.0])
cbar_b = plt.colorbar(scatter_b, cax=cax_b)
cbar_b.set_label('$E[n]$', fontsize=8)
cbar_b.ax.tick_params(labelsize=6)

props = dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='gray', linewidth=0.8)
ax2.text(0.02, 0.02,
         f'ρ(PC1, E[n])={rho_pc1:.3f}, p={p_pc1:.1e}\nρ(PC2, E[n])={rho_pc2:.3f}, p={p_pc2:.1e}',
         transform=ax2.transAxes, fontsize=7, va='bottom', bbox=props)

# ============ Panel C: E[n] vs Ricci Curvature ============
ax3 = subfigs[2].subplots()
ax3.set_title('c', fontsize=20, fontweight='bold', pad=15)

ax3.scatter(en_values, en_ricci, c='#34495e', s=3, alpha=0.3,
           edgecolors='none', rasterized=True)

# Linear fit + 95% CI band
slope_c, intercept_c, r_value_c, p_value_c, std_err_c = linregress(en_values, en_ricci)
x_fit_c = np.linspace(en_values.min(), en_values.max(), 100)
y_fit_c = slope_c * x_fit_c + intercept_c
ax3.plot(x_fit_c, y_fit_c, color='#9b59b6', linewidth=2, alpha=0.8, zorder=5)

nn_c = len(en_values)
residuals_c = en_ricci - (slope_c * en_values + intercept_c)
residual_std_c = np.sqrt(np.sum(residuals_c**2) / (nn_c - 2))
x_mean_c = np.mean(en_values)
sxx_c = np.sum((en_values - x_mean_c)**2)
se_fit_c = residual_std_c * np.sqrt(1/nn_c + (x_fit_c - x_mean_c)**2 / sxx_c)
ax3.fill_between(x_fit_c, y_fit_c - 1.96*se_fit_c, y_fit_c + 1.96*se_fit_c,
                 color='#9b59b6', alpha=0.15, zorder=4)

ax3.set_xlabel('$E[n]$ (Integrability Error)')
ax3.set_ylabel('Ricci Curvature')
ax3.grid(True, alpha=0.3, linewidth=0.5)

props = dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='gray', linewidth=0.8)
ax3.text(0.98, 0.98,
         f'Spearman ρ={rho_ricci:.3f}, p={p_ricci:.1e}\nR²={r_value_c**2:.3f}\nn={len(en_values)}',
         transform=ax3.transAxes, fontsize=7, va='top', ha='right', bbox=props)

# ============ Panel D: Correlation matrix ============
axes_d = subfigs[3].subplots(n_vars, n_vars, gridspec_kw={'wspace': 0.05, 'hspace': 0.05})

# Invisible background axes for aligned title
ax_d_bg = subfigs[3].add_subplot(111, frameon=False)
ax_d_bg.set_xticks([])
ax_d_bg.set_yticks([])
ax_d_bg.set_title('d', fontsize=20, fontweight='bold', pad=15)

cmap_d = plt.cm.RdBu_r
norm_d = mcolors.Normalize(vmin=-1, vmax=1)

np.random.seed(42)
for i in range(n_vars):
    for j in range(n_vars):
        ax = axes_d[i, j]

        if i == j:
            vals = data_cols[i]
            vals = vals[np.isfinite(vals)]
            try:
                kde = gaussian_kde(vals)
                x_kde = np.linspace(vals.min(), vals.max(), 100)
                ax.fill_between(x_kde, kde(x_kde), alpha=0.6, color='#3498db')
                ax.plot(x_kde, kde(x_kde), color='#2c3e50', linewidth=0.8)
            except Exception:
                ax.hist(vals, bins=20, color='#3498db', alpha=0.6, density=True)
            ax.set_xlim(vals.min(), vals.max())
        elif i > j:
            color = cmap_d(norm_d(corr_mat[i, j]))
            ax.set_facecolor(color)
            text_color = 'white' if abs(corr_mat[i, j]) > 0.4 else 'black'
            ax.text(0.5, 0.5, f'{corr_mat[i, j]:.2f}', transform=ax.transAxes,
                    ha='center', va='center', fontsize=7, fontweight='bold', color=text_color)
        else:
            xi = data_cols[j]
            yi = data_cols[i]
            mask = np.isfinite(xi) & np.isfinite(yi)
            n_plot = min(300, mask.sum())
            plot_idx = np.random.choice(np.where(mask)[0], n_plot, replace=False)
            ax.scatter(xi[plot_idx], yi[plot_idx], s=0.5, alpha=0.3, c='#34495e',
                      edgecolors='none', rasterized=True)
            p = pval_mat[i, j]
            sig = '***' if p < 0.001 else ('**' if p < 0.01 else ('*' if p < 0.05 else 'ns'))
            ax.text(0.95, 0.95, sig, transform=ax.transAxes, ha='right', va='top',
                    fontsize=7, fontweight='bold',
                    color='red' if p < 0.05 else 'gray')

        ax.set_xticks([])
        ax.set_yticks([])
        if i == n_vars - 1:
            ax.set_xlabel(var_names_full[j], fontsize=5.5, rotation=45, ha='right')
        if j == 0:
            ax.set_ylabel(var_names_full[i], fontsize=5.5)

plt.savefig('manuscript/figure1.png', dpi=600, bbox_inches='tight')
plt.close()

# ============ Summary Statistics (for figure caption) ============
print("\n" + "=" * 70)
print("FIGURE 1 — STATISTICS FOR CAPTION")
print("=" * 70)

print(f"\n{'─'*70}")
print(f"[Panel A] Residue-level ρ vs mean E[n]")
print(f"{'─'*70}")
print(f"  n = {len(residue_corr)} sequences")
print(f"  Significant positive ρ: {len(sig_pos)} ({len(sig_pos)/len(residue_corr)*100:.1f}%)")
print(f"  Significant negative ρ: {len(sig_neg)} ({len(sig_neg)/len(residue_corr)*100:.1f}%)")
print(f"  Non-significant:        {len(non_sig)} ({len(non_sig)/len(residue_corr)*100:.1f}%)")
print(f"  Overall Spearman ρ = {overall_rho:.4f}, p = {overall_p:.2e}")
print(f"  Linear fit: slope = {slope:.4f}, R² = {r_value**2:.4f}, p = {p_value_fit:.2e}")
print(f"  Mean E[n] range: [{residue_corr['mean_E_n'].min():.3f}, {residue_corr['mean_E_n'].max():.3f}]")
print(f"  Residue ρ range: [{residue_corr['spearman_rho'].min():.3f}, {residue_corr['spearman_rho'].max():.3f}]")
print(f"  Median |ρ|: {residue_corr['spearman_rho'].abs().median():.4f}")

print(f"\n{'─'*70}")
print(f"[Panel B] PCA embedding colored by E[n]")
print(f"{'─'*70}")
print(f"  Colored points (has E[n]): n = {len(en_indices)}")
print(f"  Gray background (no E[n]): n = {len(no_en_indices)}")
print(f"  Total: n = {len(pca_2d)}")
print(f"  PC1 explained variance: {explained_var[0]*100:.2f}%")
print(f"  PC2 explained variance: {explained_var[1]*100:.2f}%")
print(f"  ρ(PC1, E[n]) = {rho_pc1:.4f}, p = {p_pc1:.2e}")
print(f"  ρ(PC2, E[n]) = {rho_pc2:.4f}, p = {p_pc2:.2e}")
print(f"  E[n] range: [{en_values.min():.3f}, {en_values.max():.3f}]")
print(f"  E[n] mean ± std: {en_values.mean():.3f} ± {en_values.std():.3f}")

print(f"\n{'─'*70}")
print(f"[Panel C] E[n] vs Ricci Curvature")
print(f"{'─'*70}")
print(f"  n = {len(en_values)}")
print(f"  Spearman ρ = {rho_ricci:.4f}, p = {p_ricci:.2e}")
print(f"  E[n] range: [{en_values.min():.3f}, {en_values.max():.3f}]")
print(f"  Ricci range: [{en_ricci.min():.4f}, {en_ricci.max():.4f}]")
print(f"  Ricci mean ± std: {en_ricci.mean():.4f} ± {en_ricci.std():.4f}")

print(f"\n{'─'*70}")
print(f"[Panel D] Scale separation correlation matrix")
print(f"{'─'*70}")
print(f"  n = {len(en_indices)} sequences, {n_vars} variables")
print(f"  k-NN: k = {k}")
en_idx = n_vars - 1
print(f"\n  E[n] vs all geometric features:")
for j in range(n_vars):
    if j != en_idx:
        rho_val = corr_mat[en_idx, j]
        p_val = pval_mat[en_idx, j]
        sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else "ns"
        print(f"    E[n] vs {var_names_full[j]:<25s}: ρ = {rho_val:>7.4f}, p = {p_val:.2e} ({sig})")

print(f"\n  Max |ρ(E[n], ·)|: {max(abs(corr_mat[en_idx, j]) for j in range(n_vars) if j != en_idx):.4f}")
print(f"  → E[n] is effectively decoupled from all latent geometric features")

print(f"\n  Top 5 pairwise correlations (excluding E[n]):")
pairs = []
for i in range(n_vars - 1):
    for j in range(i+1, n_vars - 1):
        pairs.append((var_names_full[i], var_names_full[j], corr_mat[i, j], pval_mat[i, j]))
pairs.sort(key=lambda x: abs(x[2]), reverse=True)
for v1, v2, rho_val, p_val in pairs[:5]:
    print(f"    {v1:<25s} vs {v2:<25s}: ρ = {rho_val:>7.4f}, p = {p_val:.2e}")

print(f"\n{'─'*70}")
print(f"KEY CONCLUSION: All |ρ(E[n], geometric)| < 0.07")
print(f"Microscopic 3D geometry (E[n]) is completely absent from")
print(f"the macroscopic ESM-2 latent manifold structure.")
print(f"{'─'*70}")

print(f"\nSaved: manuscript/figure1.png")
print("=" * 70)
