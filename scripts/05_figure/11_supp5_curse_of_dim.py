#!/usr/bin/env python3
# -- Project root setup --
import os as _os
from pathlib import Path as _Path
_os.chdir(_Path(__file__).resolve().parents[2])
"""
Supplementary Figure S3: Mathematical Derivation of Curse of Dimensionality
Visualizes the theoretical basis for why high-dimensional KDE fails.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.special import gamma

plt.rcParams.update({'font.size': 10, 'axes.titlesize': 13, 'axes.labelsize': 12})

fig, axes = plt.subplots(2, 3, figsize=(16, 9), constrained_layout=True)

dimensions = np.arange(1, 51)
n = 10000

# ================= Panel A: Volume of unit sphere =================
ax1 = axes[0, 0]
volumes = np.pi**(dimensions/2) / gamma(dimensions/2 + 1)
ax1.plot(dimensions, volumes, 'o-', linewidth=2, markersize=5, color='#FF6B6B')
ax1.set_xlabel('Dimension d')
ax1.set_ylabel('Volume of Unit Sphere')
ax1.set_title('a', fontweight='bold', loc='center')
ax1.grid(True, alpha=0.3)
ax1.axvline(50, color='red', linestyle='--', alpha=0.5, label='d=50')
ax1.legend(loc='lower left')
ax1.text(0.5, 0.95, r'$V_d = \frac{\pi^{d/2}}{\Gamma(d/2 + 1)}$',
         transform=ax1.transAxes, ha='center', va='top',
         fontsize=13, bbox=dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='gray'))

# ================= Panel B: Expected distance =================
ax2 = axes[0, 1]
expected_dist = np.sqrt(dimensions / 6)
ax2.plot(dimensions, expected_dist, 'o-', linewidth=2, markersize=5, color='#4ECDC4')
ax2.set_xlabel('Dimension d')
ax2.set_ylabel('Expected Distance')
ax2.set_title('b', fontweight='bold', loc='center')
ax2.grid(True, alpha=0.3)
ax2.axvline(50, color='red', linestyle='--', alpha=0.5)
ax2.axhline(np.sqrt(50/6), color='red', linestyle='--', alpha=0.5,
            label=f'd=50: {np.sqrt(50/6):.2f}')
ax2.legend(loc='lower right')
ax2.text(15, 2.5, r'$E[\|x-y\|] \approx \sqrt{d/6}$',
         fontsize=13, bbox=dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='gray'))

# ================= Panel C: Kernel weight decay =================
ax3 = axes[0, 2]
bandwidths = n ** (-1.0 / (dimensions + 4))
kernel_weights = np.exp(-expected_dist**2 / (2 * bandwidths**2))
ax3.semilogy(dimensions, kernel_weights, 'o-', linewidth=2, markersize=5, color='#95E1D3')
ax3.set_xlabel('Dimension d')
ax3.set_ylabel('Kernel Weight (log scale)')
ax3.set_title('c', fontweight='bold', loc='center')
ax3.grid(True, alpha=0.3)
ax3.axvline(50, color='red', linestyle='--', alpha=0.5)
ax3.axhline(kernel_weights[49], color='red', linestyle='--', alpha=0.5,
            label=f'd=50: {kernel_weights[49]:.2e}')
ax3.legend(loc='upper right')

# ================= Panel D: Expected neighbors vs radius =================
ax4 = axes[1, 0]
r_values = np.linspace(0.1, 3, 100)
dims_to_plot = [2, 5, 10, 20, 50]
colors = plt.cm.viridis(np.linspace(0, 0.9, len(dims_to_plot)))

for d, color in zip(dims_to_plot, colors):
    volumes_r = (np.pi**(d/2) / gamma(d/2 + 1)) * r_values**d
    n_neighbors = n * volumes_r
    ax4.plot(r_values, n_neighbors, linewidth=2.5, label=f'd={d}', color=color)

ax4.set_xlabel('Radius r')
ax4.set_ylabel('Expected Neighbors')
ax4.set_title('d', fontweight='bold', loc='center')
ax4.legend(fontsize=9, loc='upper left')
ax4.grid(True, alpha=0.3)
ax4.set_yscale('log')
ax4.set_ylim([0.1, n*10])
ax4.axhline(100, color='gray', linestyle=':', alpha=0.5)

# ================= Panel E: Concentration of measure =================
ax5 = axes[1, 1]
dims_to_plot2 = [2, 10, 50, 100]
r_range = np.linspace(0, 1, 1000)
colors2 = plt.cm.plasma(np.linspace(0, 0.9, len(dims_to_plot2)))

for d, color in zip(dims_to_plot2, colors2):
    pdf = d * r_range**(d-1)
    ax5.plot(r_range, pdf, linewidth=2.5, label=f'd={d}', color=color)

ax5.set_xlabel('Distance from Origin r')
ax5.set_ylabel('Probability Density')
ax5.set_title('e', fontweight='bold', loc='center')
ax5.legend(fontsize=9, loc='upper left')
ax5.grid(True, alpha=0.3)

# ================= Panel F: Required sample size =================
ax6 = axes[1, 2]
k = 5
n_required = np.array([5.0**d for d in dimensions])
ax6.semilogy(dimensions, n_required, 'o-', linewidth=2, markersize=5, color='#FF6B6B')
ax6.axhline(1e6, color='green', linestyle='--', linewidth=2, label='1M samples')
ax6.axhline(1e4, color='blue', linestyle='--', linewidth=2, label='10k (our dataset)')
ax6.set_xlabel('Dimension d')
ax6.set_ylabel('Required Sample Size (log scale)')
ax6.set_title('f', fontweight='bold', loc='center')
ax6.legend(fontsize=9, loc='upper left')
ax6.grid(True, alpha=0.3)
ax6.axvline(50, color='red', linestyle='--', alpha=0.5)

# Save
output_path = 'manuscript/supp5.png'
plt.savefig(output_path, dpi=600, bbox_inches='tight', facecolor='white')
print(f"Saved: {output_path}")

# Terminal output
print("\n" + "="*60)
print("CURSE OF DIMENSIONALITY - MATHEMATICAL SUMMARY")
print("="*60)
print()
print("Key formulas:")
print(f"  1. Volume of d-ball: V_d(r) = π^(d/2) / Γ(d/2+1) · r^d")
print(f"  2. Expected distance (uniform): E[||x-y||] ≈ √(d/6)")
print(f"  3. Expected distance (Gaussian): E[||x-y||] ≈ √d")
print(f"  4. Scott's bandwidth: h = n^(-1/(d+4)) · σ")
print(f"  5. Gaussian kernel: K(x) = exp(-||x||² / 2h²)")
print()
print(f"For d=50, n={n}:")
print(f"  Unit sphere volume V_50 = {volumes[49]:.2e}")
print(f"  Expected distance = {expected_dist[49]:.4f}")
print(f"  Scott bandwidth h = {bandwidths[49]:.4f}")
print(f"  Kernel weight at mean distance = {kernel_weights[49]:.2e}")
print(f"  Required samples (k={k}) = {5.0**50:.2e}")
print(f"  Sample deficit = {5.0**50/n:.2e}×")
print()
print("Conclusion: All kernel weights ≈ 0")
print("  → Density becomes discrete/quantized")
print("  → Correlation analysis impossible in 50D")
print("="*60)

plt.close()
