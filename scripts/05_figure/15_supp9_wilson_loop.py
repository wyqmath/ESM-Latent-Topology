#!/usr/bin/env python3
# -- Project root setup --
import os as _os
from pathlib import Path as _Path
_os.chdir(_Path(__file__).resolve().parents[2])

import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


plt.rcParams['figure.dpi'] = 600
plt.rcParams['savefig.dpi'] = 600
plt.rcParams['font.size'] = 9
plt.rcParams['font.family'] = 'DejaVu Sans'


def check_inputs(paths: dict) -> bool:
    print("[1/6] Checking input files...")
    ok = True
    for name, p in paths.items():
        exists = p.exists()
        print(f"  - {name:<18s}: {'OK' if exists else 'MISSING'} ({p})")
        ok = ok and exists
    return ok


def parse_en_means(report_path: Path) -> dict:
    """Parse E[n] mean per path from wilson_loop_report.txt."""
    text = report_path.read_text(encoding='utf-8', errors='ignore')
    lines = text.splitlines()

    en_map = {}
    current_path = None

    for line in lines:
        s = line.strip()

        m_path = re.match(r'^路径\s*[:：]\s*(.+)$', s)
        if m_path:
            current_path = m_path.group(1).strip()
            continue

        m_path_alt = re.match(r'^路径\s+(.+?)\s*[:：]?$', s)
        if m_path_alt and 'E[' not in s:
            current_path = m_path_alt.group(1).strip()
            continue

        m_en = re.search(r'E\[n\]均值\s*[:：]\s*([-+]?\d*\.?\d+)', s)
        if m_en and current_path:
            en_map[current_path] = float(m_en.group(1))

    return en_map


def panel_a(ax, umap_2d, paths_data, wilson_df):
    ax.scatter(umap_2d[:, 0], umap_2d[:, 1], s=2, c='#bfbfbf', alpha=0.35, linewidths=0)

    wilson_map = dict(zip(wilson_df['path_name'], wilson_df['wilson_loop']))
    colors = plt.cm.tab10(np.linspace(0, 1, len(paths_data)))

    # 手动微调，使用“圈上的点”作为定位基准
    label_offsets = {
        'anchor_region': (0, 2),        # 往下移动一点（相对之前更贴近圈）
        'phase_boundary': (0, -10),     # 放在圈下方
        'center_large': (24, 2),        # 再往右 6px
        'center_small': (24, 2),        # 再往右 6px
        'random_region': (-8, 8),       # 再往左上角
    }

    for i, p in enumerate(paths_data):
        name = p['name']
        path_2d = p['path_2d']
        # 显式闭合路径，避免视觉缺口
        closed_path = np.vstack([path_2d, path_2d[0]])
        ax.plot(closed_path[:, 0], closed_path[:, 1], lw=1.5, color=colors[i], label=name)

        # 使用“圈上的点”作为锚点，而不是圆心
        if name == 'anchor_region':
            anchor_idx = int(np.argmax(path_2d[:, 1]))   # 圈顶部点
        elif name == 'phase_boundary':
            anchor_idx = int(np.argmin(path_2d[:, 1]))   # 圈底部点
        elif name == 'random_region':
            # 圈左上角：最大化 (-x + y)
            anchor_idx = int(np.argmax(-path_2d[:, 0] + path_2d[:, 1]))
        else:
            anchor_idx = int(np.argmax(path_2d[:, 0]))   # 默认圈右侧点

        ax_pt = path_2d[anchor_idx]
        x_anchor, y_anchor = float(ax_pt[0]), float(ax_pt[1])

        wv = wilson_map.get(name, np.nan)
        label = f"{name}\nW={wv:.2f}" if np.isfinite(wv) else name

        dx, dy = label_offsets.get(name, (4, 4))
        va = 'bottom' if dy >= 0 else 'top'
        ax.annotate(label, (x_anchor, y_anchor), fontsize=7, color=colors[i],
                    xytext=(dx, dy), textcoords='offset points',
                    ha='center', va=va)

    ax.set_title('a', fontsize=14, fontweight='bold', pad=8)
    ax.set_xlabel('UMAP 1')
    ax.set_ylabel('UMAP 2')
    ax.grid(True, alpha=0.25, linestyle='--')
    ax.legend(frameon=False, fontsize=7, loc='best')


def panel_b(ax, wilson_df):
    x = np.arange(len(wilson_df))
    wvals = wilson_df['wilson_loop'].values
    stdvals = wilson_df['std_phase'].values
    maxvals = wilson_df['max_phase'].values

    ax.plot(x, wvals, color='#2F5D8A', marker='o', lw=2.2, ms=5, label='Wilson loop')
    ax.fill_between(x, 0, wvals, color='#2F5D8A', alpha=0.10)
    ax.set_ylabel('Wilson loop', color='#2F5D8A')
    ax.tick_params(axis='y', labelcolor='#2F5D8A')

    ax2 = ax.twinx()
    ax2.plot(x, stdvals, color='#C44E52', marker='s', lw=1.8, ls='--', ms=4, label='Phase std')
    ax2.plot(x, maxvals, color='#55A868', marker='^', lw=1.8, ls=':', ms=4, label='Phase max')
    ax2.set_ylabel('Phase fluctuation', color='#C44E52')
    ax2.tick_params(axis='y', labelcolor='#C44E52')

    ax.set_xticks(x)
    ax.set_xticklabels(wilson_df['path_name'].tolist(), rotation=30, ha='right')
    ax.set_title('b', fontsize=14, fontweight='bold', pad=8)
    ax.grid(True, axis='y', alpha=0.25, linestyle='--')

    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, frameon=False, fontsize=8, loc='upper left')


def panel_c(ax, wilson_df, paths_data):
    phase_map = {p['name']: np.asarray(p.get('phases', []), dtype=float) for p in paths_data}
    wilson_map = dict(zip(wilson_df['path_name'], wilson_df['wilson_loop']))

    names = wilson_df['path_name'].tolist()
    all_curves = {}
    for name in names:
        phases = phase_map.get(name, np.array([]))
        if phases.size == 0:
            continue
        all_curves[name] = np.cumsum(phases)

    if not all_curves:
        ax.text(0.5, 0.5, 'No phase sequence found', ha='center', va='center', transform=ax.transAxes)
        ax.set_title('c', fontsize=14, fontweight='bold', pad=8)
        return

    # 选三条代表路径：Wilson 最高、最低、以及中位附近
    ordered = wilson_df.sort_values('wilson_loop').reset_index(drop=True)
    rep_names = [
        ordered.iloc[-1]['path_name'],
        ordered.iloc[0]['path_name'],
        ordered.iloc[len(ordered) // 2]['path_name'],
    ]
    seen = set()
    rep_names = [n for n in rep_names if not (n in seen or seen.add(n))]

    # 非代表路径作为背景，逐条命名并区分线型
    non_rep_names = [n for n in names if n not in rep_names and n in all_curves]
    bg_styles = ['-', '--', ':']
    for i, name in enumerate(non_rep_names):
        curve = all_curves[name]
        steps = np.arange(1, len(curve) + 1)
        ax.plot(
            steps,
            curve,
            color='#9e9e9e',
            lw=1.2,
            alpha=0.75,
            ls=bg_styles[i % len(bg_styles)],
            zorder=1,
            label=f"{name} (background)",
        )

    rep_colors = ['#2F5D8A', '#C44E52', '#55A868']
    for i, name in enumerate(rep_names):
        if name not in all_curves:
            continue
        curve = all_curves[name]
        steps = np.arange(1, len(curve) + 1)
        wv = wilson_map.get(name, np.nan)
        label = f"{name} (W={wv:.2f})" if np.isfinite(wv) else name
        ax.plot(steps, curve, color=rep_colors[i % len(rep_colors)], lw=2.2, alpha=0.95, label=label, zorder=3)

    ax.set_title('c', fontsize=14, fontweight='bold', pad=8)
    ax.set_xlabel('Path step')
    ax.set_ylabel('Cumulative phase')
    ax.grid(True, alpha=0.25, linestyle='--')
    ax.legend(frameon=False, fontsize=7, loc='best')


def print_summary(wilson_df, en_map, paths_data):
    w = wilson_df['wilson_loop'].values
    print("\n[6/6] Summary stats")
    print(f"  - Wilson loop mean: {np.mean(w):.4f}")
    print(f"  - Wilson loop std : {np.std(w):.4f}")
    print(f"  - Wilson range    : [{np.min(w):.4f}, {np.max(w):.4f}]")

    std_vals = wilson_df['std_phase'].values
    max_vals = wilson_df['max_phase'].values
    print(f"  - Phase std mean  : {np.mean(std_vals):.4f}")
    print(f"  - Phase max mean  : {np.mean(max_vals):.4f}")

    # 相关性（仅描述性）
    if len(w) > 1:
        corr_w_std = float(np.corrcoef(w, std_vals)[0, 1])
        corr_w_max = float(np.corrcoef(w, max_vals)[0, 1])
        print(f"  - Corr(W, std_phase): {corr_w_std:+.4f}")
        print(f"  - Corr(W, max_phase): {corr_w_max:+.4f}")

    print("\n  Wilson ranking (high -> low):")
    ranked = wilson_df.sort_values('wilson_loop', ascending=False)
    for i, (_, row) in enumerate(ranked.iterrows(), start=1):
        print(f"  {i}. {row['path_name']:15s} W={row['wilson_loop']:.4f}, std={row['std_phase']:.4f}, max={row['max_phase']:.4f}")

    phase_map = {p['name']: np.asarray(p.get('phases', []), dtype=float) for p in paths_data}
    ordered = wilson_df.sort_values('wilson_loop').reset_index(drop=True)
    rep_names = [
        ordered.iloc[-1]['path_name'],
        ordered.iloc[0]['path_name'],
        ordered.iloc[len(ordered) // 2]['path_name'],
    ]
    seen = set()
    rep_names = [n for n in rep_names if not (n in seen or seen.add(n))]

    print("\n  Representative cumulative-phase endpoints:")
    for name in rep_names:
        phases = phase_map.get(name, np.array([]))
        if phases.size == 0:
            print(f"  - {name:15s} cumulative_end=N/A (no phases)")
            continue
        cumulative_end = float(np.cumsum(phases)[-1])
        wv = float(wilson_df.loc[wilson_df['path_name'] == name, 'wilson_loop'].iloc[0])
        print(f"  - {name:15s} cumulative_end={cumulative_end:>8.4f}, Wilson={wv:>7.4f}")

    print("\n  Non-representative background paths in panel c:")
    non_rep_names = [n for n in wilson_df['path_name'].tolist() if n not in rep_names]
    if non_rep_names:
        for name in non_rep_names:
            wv = float(wilson_df.loc[wilson_df['path_name'] == name, 'wilson_loop'].iloc[0])
            print(f"  - {name:15s} Wilson={wv:>7.4f}")
    else:
        print("  - None")

    print("\n  Per-path values:")
    for _, row in wilson_df.iterrows():
        name = row['path_name']
        r = row['radius']
        wl = row['wilson_loop']
        ap = row['avg_phase']
        sp = row['std_phase']
        mp = row['max_phase']
        if name in en_map:
            print(
                f"  - {name:15s} radius={r:>4.1f}, wilson={wl:>7.4f}, avg_phase={ap:>7.4f}, "
                f"std_phase={sp:>7.4f}, max_phase={mp:>7.4f}, E[n]_mean={en_map[name]:.4f}"
            )
        else:
            print(
                f"  - {name:15s} radius={r:>4.1f}, wilson={wl:>7.4f}, avg_phase={ap:>7.4f}, "
                f"std_phase={sp:>7.4f}, max_phase={mp:>7.4f}, E[n]_mean=N/A"
            )


def main():
    print("=" * 80)
    print("Supplementary Figure 9: Wilson Loop (minimal runnable)")
    print("=" * 80)

    paths = {
        'wilson_csv': Path('data/wilson_loops/wilson_loop_values.csv'),
        'wilson_paths': Path('data/wilson_loops/wilson_loop_paths.npy'),
        'wilson_report': Path('data/wilson_loops/wilson_loop_report.txt'),
        'umap_2d': Path('data/umap/umap_embeddings_2d.npy'),
        'metadata': Path('data/metadata_final_with_en.csv'),
    }

    if not check_inputs(paths):
        raise FileNotFoundError('Missing required input files, stop here.')

    print("\n[2/6] Loading data...")
    wilson_df = pd.read_csv(paths['wilson_csv'])
    umap_2d = np.load(paths['umap_2d'])
    paths_data = np.load(paths['wilson_paths'], allow_pickle=True)
    if hasattr(paths_data, 'tolist'):
        paths_data = paths_data.tolist()

    en_map = {}
    try:
        en_map = parse_en_means(paths['wilson_report'])
        print(f"  - Parsed E[n] mean for {len(en_map)} paths from report")
    except Exception as e:
        print(f"  - Failed to parse E[n] means: {e}")

    print("\n[3/6] Building 1x3 canvas...")
    fig = plt.figure(figsize=(20, 6))

    ax1 = plt.subplot(1, 3, 1)
    panel_a(ax1, umap_2d, paths_data[:5], wilson_df)

    ax2 = plt.subplot(1, 3, 2)
    panel_b(ax2, wilson_df)

    ax3 = plt.subplot(1, 3, 3)
    panel_c(ax3, wilson_df, paths_data)

    plt.tight_layout()

    print("\n[4/6] Saving figure...")
    out_dir = Path('manuscript')
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / 'supp9.png'
    fig.savefig(out_file, dpi=600, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"  - Saved: {out_file}")

    print("\n[5/6] Basic output check...")
    print(f"  - Exists: {out_file.exists()}")
    if out_file.exists():
        print(f"  - Size  : {out_file.stat().st_size / 1024:.1f} KB")

    print_summary(wilson_df, en_map, paths_data)
    print("\n✓ Supplementary Figure 9 generation complete")


if __name__ == '__main__':
    main()
