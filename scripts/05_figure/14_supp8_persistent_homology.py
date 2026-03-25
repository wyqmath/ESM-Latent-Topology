# -- Project root setup --
import os as _os
from pathlib import Path as _Path
_os.chdir(_Path(__file__).resolve().parents[2])
import numpy as np
import matplotlib.pyplot as plt
import pickle
import json
from pathlib import Path

# 设置matplotlib参数
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 10
plt.rcParams['axes.linewidth'] = 1.5
plt.rcParams['xtick.major.width'] = 1.5
plt.rcParams['ytick.major.width'] = 1.5

# 色盲友好配色
COLORS = {
    'H0': '#0173B2',  # 蓝色 (保留用于Betti曲线)
    'H1': '#DE8F05',  # 橙色
    'H2': '#029E73',  # 绿色
}

def load_persistence_data():
    """加载持续同调数据"""
    with open('data/persistent_homology/persistence_diagrams.pkl', 'rb') as f:
        data = pickle.load(f)

    diagrams = data['diagrams']

    with open('data/persistent_homology/betti_numbers.json', 'r') as f:
        betti_data = json.load(f)

    return diagrams, betti_data

def plot_barcode(ax, diagrams):
    """绘制条形码图 (已移除 H0，仅显示 H1, H2)"""
    # 仅提取 H1 和 H2
    h1_features = diagrams[1].copy()
    h2_features = diagrams[2].copy()

    # 处理可能存在的 inf 值 (尽管 H1/H2 通常不会有 inf，以防万一)
    all_deaths = np.concatenate([h1_features[:, 1], h2_features[:, 1]])
    if len(all_deaths) > 0 and np.any(np.isfinite(all_deaths)):
        finite_max = np.max(all_deaths[np.isfinite(all_deaths)])
        cap_val = finite_max * 1.1
        h1_features[:, 1] = np.where(np.isinf(h1_features[:, 1]), cap_val, h1_features[:, 1])
        h2_features[:, 1] = np.where(np.isinf(h2_features[:, 1]), cap_val, h2_features[:, 1])

    # 计算持续性并排序
    def sort_by_persistence(features):
        if len(features) == 0:
            return features
        persistence = features[:, 1] - features[:, 0]
        sorted_idx = np.argsort(persistence)[::-1]
        return features[sorted_idx]

    h1_sorted = sort_by_persistence(h1_features)
    h2_sorted = sort_by_persistence(h2_features)

    # 只显示前50个最持久的特征
    max_features = 50
    y_pos = 0

    # 1. 绘制H2条形码 (在最下方)
    for i, feature in enumerate(h2_sorted[:max_features]):
        ax.plot([feature[0], feature[1]], [y_pos, y_pos],
               color=COLORS['H2'], linewidth=2.5, solid_capstyle='butt')
        y_pos += 1

    h2_end = y_pos
    if len(h2_sorted) > 0:
        y_pos += 8  # 增加 H1 和 H2 之间的垂直间距，让图面更舒展

    # 2. 绘制H1条形码 (在上方)
    for i, feature in enumerate(h1_sorted[:max_features]):
        ax.plot([feature[0], feature[1]], [y_pos, y_pos],
               color=COLORS['H1'], linewidth=2.5, solid_capstyle='butt')
        y_pos += 1

    h1_end = y_pos

    # 3. 添加维度标签 (使用 axes transform for x, data for y)
    import matplotlib.transforms as mtransforms
    trans = mtransforms.blended_transform_factory(ax.transAxes, ax.transData)
    if len(h2_sorted) > 0:
        ax.text(0.02, (0 + h2_end) / 2, 'H₂', fontsize=12, fontweight='bold',
               ha='left', va='center', color=COLORS['H2'], transform=trans)
    if len(h1_sorted) > 0:
        ax.text(0.02, (h2_end + 8 + h1_end) / 2, 'H₁', fontsize=12, fontweight='bold',
               ha='left', va='center', color=COLORS['H1'], transform=trans)

    ax.set_xlabel('Filtration Value', fontsize=12, fontweight='bold')
    ax.set_ylabel('Topological Features', fontsize=12, fontweight='bold')
    ax.set_ylim(-2, y_pos + 2)
    ax.set_yticks([])
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5, axis='x')

def plot_persistence_diagram(ax, diagrams):
    """绘制持续图 (Persistence Diagram) — 仅显示 H1, H2"""
    h1_features = diagrams[1]
    h2_features = diagrams[2]

    finite_vals = []
    for features in (h1_features, h2_features):
        if len(features) > 0:
            finite_vals.extend(features[np.isfinite(features)])

    if len(finite_vals) > 0:
        min_val = float(np.min(finite_vals))
        max_val = float(np.max(finite_vals))
    else:
        min_val, max_val = 0.0, 1.0

    diag_min = min(0.0, min_val)
    diag_max = max_val * 1.05 if max_val > 0 else 1.0

    def scatter_features(features, color, label):
        if len(features) == 0:
            return
        births = features[:, 0]
        deaths = features[:, 1]
        finite_mask = np.isfinite(births) & np.isfinite(deaths)
        if np.any(finite_mask):
            ax.scatter(
                births[finite_mask],
                deaths[finite_mask],
                s=20,
                alpha=0.75,
                color=color,
                edgecolors='none',
                label=label,
            )

    scatter_features(h1_features, COLORS['H1'], 'H₁')
    scatter_features(h2_features, COLORS['H2'], 'H₂')

    ax.plot([diag_min, diag_max], [diag_min, diag_max], linestyle='--', color='gray', linewidth=1.5, label='y = x')
    ax.set_xlim(diag_min, diag_max)
    ax.set_ylim(diag_min, diag_max)
    ax.set_xlabel('Birth', fontsize=12, fontweight='bold')
    ax.set_ylabel('Death', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    ax.legend(frameon=False, fontsize=10, loc='lower right')


def summarize_statistics(diagrams):
    """打印图注所需的关键统计信息"""
    print("\n[Stats] Persistence diagram summary")

    for dim, label in [(1, 'H1'), (2, 'H2')]:
        features = diagrams[dim]
        births = features[:, 0] if len(features) > 0 else np.array([])
        deaths = features[:, 1] if len(features) > 0 else np.array([])

        finite_mask = np.isfinite(births) & np.isfinite(deaths)
        finite_births = births[finite_mask]
        finite_deaths = deaths[finite_mask]

        print(f"  - {label} points: {int(np.sum(finite_mask))}")
        if len(finite_births) > 0:
            persistence = finite_deaths - finite_births
            p50 = float(np.percentile(persistence, 50))
            p90 = float(np.percentile(persistence, 90))
            print(
                f"    birth range: [{finite_births.min():.4f}, {finite_births.max():.4f}], "
                f"death range: [{finite_deaths.min():.4f}, {finite_deaths.max():.4f}]"
            )
            print(
                f"    persistence median/p90: {p50:.4f} / {p90:.4f}, "
                f"max: {persistence.max():.4f}"
            )
        else:
            print("    birth/death range: N/A")

    scales, betti_0, betti_1, betti_2 = compute_betti_curves(diagrams)
    betti_series = [("β0", betti_0), ("β1", betti_1), ("β2", betti_2)]
    print("\n[Stats] Betti peak summary")
    for name, curve in betti_series:
        peak_idx = int(np.argmax(curve))
        print(f"  - {name} peak: {curve[peak_idx]:.0f} at scale {scales[peak_idx]:.4f}")

    print("\n[Stats] Betti area-under-curve (0-50)")
    for name, curve in betti_series:
        auc = float(np.trapezoid(curve, scales))
        print(f"  - {name} AUC: {auc:.2f}")


def compute_betti_curves(diagrams, max_scale=50, n_points=200):
    """计算Betti曲线"""
    scales = np.linspace(0, max_scale, n_points)
    betti_0 = np.zeros(n_points)
    betti_1 = np.zeros(n_points)
    betti_2 = np.zeros(n_points)

    h0_features = diagrams[0]
    h1_features = diagrams[1]
    h2_features = diagrams[2]

    for i, scale in enumerate(scales):
        betti_0[i] = np.sum((h0_features[:, 0] <= scale) & (h0_features[:, 1] > scale))
        betti_1[i] = np.sum((h1_features[:, 0] <= scale) & (h1_features[:, 1] > scale))
        betti_2[i] = np.sum((h2_features[:, 0] <= scale) & (h2_features[:, 1] > scale))

    return scales, betti_0, betti_1, betti_2

def plot_betti_curves(ax, diagrams):
    """绘制Betti曲线 — 双Y轴：左轴β₀，右轴β₁/β₂"""
    scales, betti_0, betti_1, betti_2 = compute_betti_curves(diagrams)

    # 左Y轴: β₀
    ln0, = ax.plot(scales, betti_0, color=COLORS['H0'], linewidth=2.5, label='β₀ (Connected Components)')
    ax.set_xlabel('Filtration Scale', fontsize=12, fontweight='bold')
    ax.set_ylabel('β₀ (Connected Components)', fontsize=12, fontweight='bold', color=COLORS['H0'])
    ax.tick_params(axis='y', labelcolor=COLORS['H0'])
    ax.set_xlim(0, scales.max())
    ax.set_ylim(0, betti_0.max() * 1.1)
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)

    # 右Y轴: β₁, β₂
    ax2 = ax.twinx()
    ln1, = ax2.plot(scales, betti_1, color=COLORS['H1'], linewidth=2.5, label='β₁ (Loops)')
    ln2, = ax2.plot(scales, betti_2, color=COLORS['H2'], linewidth=2.5, label='β₂ (Voids)')
    ax2.set_ylabel('β₁, β₂ (Loops & Voids)', fontsize=12, fontweight='bold', color=COLORS['H1'])
    ax2.tick_params(axis='y', labelcolor=COLORS['H1'])
    ax2.set_ylim(0, max(betti_1.max(), betti_2.max()) * 1.1)

    # 合并图例
    lns = [ln0, ln1, ln2]
    labs = [l.get_label() for l in lns]
    ax.legend(lns, labs, frameon=False, fontsize=8, loc='upper right')

def main():
    print("=" * 80)
    print("Supplementary Figure 8: Persistent Homology Detailed Results")
    print("=" * 80)

    output_dir = Path('manuscript')
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n[1/3] Loading persistence data...")
    diagrams, betti_data = load_persistence_data()

    print("\n[2/3] Generating figure...")
    fig = plt.figure(figsize=(18, 5))

    # Panel A: Barcode Plot (H1, H2)
    ax1 = plt.subplot(1, 3, 1)
    plot_barcode(ax1, diagrams)
    ax1.set_title('a', fontsize=14, fontweight='bold', loc='center', pad=10)

    # Panel B: Persistence Diagram (H1, H2)
    ax2 = plt.subplot(1, 3, 2)
    plot_persistence_diagram(ax2, diagrams)
    ax2.set_title('b', fontsize=14, fontweight='bold', loc='center', pad=10)

    # Panel C: Betti Curves
    ax3 = plt.subplot(1, 3, 3)
    plot_betti_curves(ax3, diagrams)
    ax3.set_title('c', fontsize=14, fontweight='bold', loc='center', pad=10)

    plt.tight_layout()

    print("\n[3/3] Saving figure...")
    output_file = output_dir / 'supp8.png'
    fig.savefig(output_file, dpi=600, bbox_inches='tight', facecolor='white')
    print(f"  - Saved: {output_file}")
    plt.close()

    summarize_statistics(diagrams)
    print("\n✓ Supplementary Figure 8 generation complete!")

if __name__ == '__main__':
    main()
