#!/usr/bin/env python3
# -- Project root setup --
import os as _os
from pathlib import Path as _Path
_os.chdir(_Path(__file__).resolve().parents[2])
"""
Supplementary Figure S4: 50D vs 2D Density Comparison
Demonstrates the curse of dimensionality in high-dimensional density estimation.

Output: manuscript/supp6.png
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from scipy.spatial.distance import pdist, squareform

# Set style
plt.style.use('seaborn-v0_8-paper')
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 10

print("=" * 70)
print("Supplementary Figure S4: 50D vs 2D Density Comparison")
print("=" * 70)

print("\n[1/4] Simulating high-dimensional data...")

np.random.seed(42)
n_sample = 1000

# 50D simulation
d_50 = 50
embeddings_50d = np.random.randn(n_sample, d_50)
distances_50d = squareform(pdist(embeddings_50d, metric='euclidean'))
mean_dist_50d = np.mean(distances_50d[np.triu_indices_from(distances_50d, k=1)])
std_dist_50d = np.std(distances_50d[np.triu_indices_from(distances_50d, k=1)])
min_dist_50d = np.min(distances_50d[np.triu_indices_from(distances_50d, k=1)])
max_dist_50d = np.max(distances_50d[np.triu_indices_from(distances_50d, k=1)])

# Scott's rule bandwidth for 50D
bandwidth_50d = n_sample ** (-1.0 / (d_50 + 4))
sigma_50d = bandwidth_50d * np.std(embeddings_50d)

# Compute 50D density
density_50d = np.zeros(n_sample)
for i in range(n_sample):
    dists = distances_50d[i]
    density_50d[i] = np.sum(np.exp(-dists**2 / (2 * sigma_50d**2))) / n_sample

unique_50d_count = len(np.unique(np.round(density_50d, 10)))

print(f"  50D samples: {n_sample}, dimensions: {d_50}")
print(f"  50D mean pairwise distance: {mean_dist_50d:.3f} ± {std_dist_50d:.3f}")
print(f"  50D distance range: [{min_dist_50d:.3f}, {max_dist_50d:.3f}]")
print(f"  50D Scott bandwidth: {bandwidth_50d:.6f}, sigma: {sigma_50d:.6f}")
print(f"  50D density range: [{density_50d.min():.2e}, {density_50d.max():.2e}]")
print(f"  50D unique density values (10 d.p.): {unique_50d_count}")

# 2D simulation
print("\n[2/4] Simulating low-dimensional data...")
d_2 = 2
embeddings_2d = np.random.randn(n_sample, d_2)
kde_2d = gaussian_kde(embeddings_2d.T)
density_2d = kde_2d(embeddings_2d.T)
unique_2d_count = len(np.unique(np.round(density_2d, 10)))

print(f"  2D samples: {n_sample}, dimensions: {d_2}")
print(f"  2D density range: [{density_2d.min():.2e}, {density_2d.max():.2e}]")
print(f"  2D unique density values (10 d.p.): {unique_2d_count}")
print(f"  Unique value ratio (2D/50D): {unique_2d_count / max(unique_50d_count, 1):.0f}×")

# Theoretical expected distance: sqrt(2d) for standard normal in d dimensions
theoretical_dist = np.sqrt(2 * d_50)
print(f"\n  Theoretical mean distance (√(2d)): {theoretical_dist:.2f}")
print(f"  Observed / Theoretical: {mean_dist_50d / theoretical_dist:.3f}")

# Create figure
print("\n[3/4] Generating figure...")
fig = plt.figure(figsize=(16, 10))
gs = fig.add_gridspec(3, 3, hspace=0.4, wspace=0.3)

# Panel A: 50D density histogram
ax1 = fig.add_subplot(gs[0, 0])
ax1.hist(density_50d, bins=50, color='#FF6B6B', alpha=0.7, edgecolor='black')
ax1.set_xlabel('Density', fontsize=12)
ax1.set_ylabel('Count', fontsize=12)
ax1.set_title('a', fontsize=13, fontweight='bold', loc='center')
ax1.text(0.95, 0.95, f'Unique: {unique_50d_count}',
         transform=ax1.transAxes, ha='right', va='top',
         bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
ax1.set_yscale('log')

# Panel B: 2D density histogram
ax2 = fig.add_subplot(gs[0, 1])
ax2.hist(density_2d, bins=50, color='#4ECDC4', alpha=0.7, edgecolor='black')
ax2.set_xlabel('Density', fontsize=12)
ax2.set_ylabel('Count', fontsize=12)
ax2.set_title('b', fontsize=13, fontweight='bold', loc='center')
ax2.text(0.95, 0.95, f'Unique: {unique_2d_count}',
         transform=ax2.transAxes, ha='right', va='top',
         bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

# Panel C: Distance distribution
ax3 = fig.add_subplot(gs[0, 2])
distances_flat = distances_50d[np.triu_indices_from(distances_50d, k=1)]
ax3.hist(distances_flat, bins=50, color='#95E1D3', alpha=0.7, edgecolor='black')
ax3.axvline(mean_dist_50d, color='red', linestyle='--', linewidth=2,
            label=f'Mean: {mean_dist_50d:.2f}')
ax3.axvline(np.sqrt(50), color='blue', linestyle='--', linewidth=2,
            label=r'$\sqrt{50}$' + f': {np.sqrt(50):.2f}')
ax3.set_xlabel('Pairwise Distance', fontsize=12)
ax3.set_ylabel('Count', fontsize=12)
ax3.set_title('c', fontsize=13, fontweight='bold', loc='center')
ax3.legend()

# Panel D: Kernel decay
ax4 = fig.add_subplot(gs[1, :])
distances_range = np.linspace(0, 15, 1000)
kernel_50d = np.exp(-distances_range**2 / (2 * sigma_50d**2))
kernel_2d = np.exp(-distances_range**2 / (2 * 0.5**2))

ax4.plot(distances_range, kernel_50d, linewidth=3, label='50D kernel', color='#FF6B6B')
ax4.plot(distances_range, kernel_2d, linewidth=3, label='2D kernel', color='#4ECDC4')
ax4.axvline(mean_dist_50d, color='red', linestyle='--', alpha=0.5)
ax4.set_xlabel('Distance', fontsize=12)
ax4.set_ylabel('Kernel Weight', fontsize=12)
ax4.set_title('d', fontsize=13, fontweight='bold', loc='center')
ax4.legend(fontsize=11)
ax4.set_yscale('log')
ax4.set_ylim([1e-10, 1.1])
ax4.grid(True, alpha=0.3)
ax4.text(0.5, 0.5, 'Curse of Dimensionality:\nIn 50D, kernel weights → 0',
         transform=ax4.transAxes, ha='center', va='center',
         fontsize=11, bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))

# Panel E: 2D scatter with density
ax5 = fig.add_subplot(gs[2, :2])
scatter = ax5.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1],
                     c=density_2d, s=10, alpha=0.6, cmap='viridis')
ax5.set_xlabel('Dimension 1', fontsize=12)
ax5.set_ylabel('Dimension 2', fontsize=12)
ax5.set_title('e', fontsize=13, fontweight='bold', loc='center')
cbar = plt.colorbar(scatter, ax=ax5)
cbar.set_label('Density', fontsize=11)

# Panel F: Quantization comparison
ax6 = fig.add_subplot(gs[2, 2])
precisions = [1, 2, 3, 4, 5, 10, 15, 20]
unique_50d_list = [len(np.unique(np.round(density_50d, p))) for p in precisions]
unique_2d_list = [len(np.unique(np.round(density_2d, p))) for p in precisions]

ax6.plot(precisions, unique_50d_list, 'o-', linewidth=2, markersize=8, label='50D', color='#FF6B6B')
ax6.plot(precisions, unique_2d_list, 's-', linewidth=2, markersize=8, label='2D', color='#4ECDC4')
ax6.set_xlabel('Decimal Precision', fontsize=12)
ax6.set_ylabel('Unique Density Values', fontsize=12)
ax6.set_title('f', fontsize=13, fontweight='bold', loc='center')
ax6.legend()
ax6.grid(True, alpha=0.3)
ax6.set_yscale('log')

output_path = 'manuscript/supp6.png'
plt.savefig(output_path, dpi=600, bbox_inches='tight', facecolor='white')
plt.close()

print(f"  Saved: {output_path}")

# Summary for figure caption
print("\n[4/4] Figure caption statistics:")
print("=" * 70)
print(f"  Panel a: 50D KDE collapses to {unique_50d_count} unique value(s)")
print(f"           density = {density_50d.mean():.2e} (effectively constant)")
print(f"  Panel b: 2D KDE yields {unique_2d_count} unique values")
print(f"           density range: [{density_2d.min():.4f}, {density_2d.max():.4f}]")
print(f"  Panel c: 50D mean pairwise distance = {mean_dist_50d:.2f}")
print(f"           theoretical √(2d) = {theoretical_dist:.2f}")
print(f"           distance concentration ratio (std/mean) = {std_dist_50d/mean_dist_50d:.4f}")
print(f"  Panel d: 50D Scott bandwidth σ = {sigma_50d:.6f}")
print(f"           kernel weight at mean distance = {np.exp(-mean_dist_50d**2/(2*sigma_50d**2)):.2e}")
print(f"  Panel e: 2D Gaussian KDE with n={n_sample} samples")
print(f"  Panel f: At 10 d.p., 50D has {unique_50d_list[5]} unique vs 2D has {unique_2d_list[5]} unique")
print(f"           Resolution ratio: {unique_2d_list[5]/max(unique_50d_list[5],1):.0f}×")
print("=" * 70)
print("Done.")
