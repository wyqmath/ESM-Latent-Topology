#!/usr/bin/env python3
"""
Figure 7: SaProt structure-aware control experiment — single composite figure.
  Panel (a): PCA scatter — ESM-2 (left) vs SaProt (right)
  Panel (b): Silhouette distributions — split violin for Knotted & FS
  Panel (c): Fold-switching conf1 vs conf2 in SaProt latent space

Output: manuscript/fig7_saprot_control.png
"""
import os, csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D
from pathlib import Path
from sklearn.metrics import silhouette_samples, silhouette_score
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT)

# ── Style ──
plt.rcParams.update({
    'figure.dpi': 600, 'savefig.dpi': 600,
    'font.size': 8, 'axes.labelsize': 9,
    'xtick.labelsize': 8, 'ytick.labelsize': 8,
    'legend.fontsize': 7,
    'font.family': 'sans-serif',
    'font.sans-serif': ['DejaVu Sans'],
})

COLORS = {
    'anchor': '#bdbdbd',
    'knotted': '#ff7f0e',
    'fold_switching': '#d62728',
    'esm2': '#1f77b4',
    'saprot': '#2ca02c',
    'conf1': '#e377c2',
    'conf2': '#17becf',
}
CAT_DISPLAY = {
    'knotted': 'Knotted', 'fold_switching': 'Fold-switching', 'anchor': 'Anchor',
}


def load_data():
    """Load all data for both models."""
    # ESM-2
    esm_pca_2d = np.load('data/pca/pca_embeddings_2d.npy')
    esm_pca_50d = np.load('data/pca/pca_embeddings_50d.npy')
    esm_meta = pd.read_csv('data/metadata_final_with_en.csv')
    esm_var = np.load('data/pca/explained_variance.npy')

    esm_mask = esm_meta['subcategory'].isin(['anchor', 'knotted', 'fold_switching']).values
    esm_labels = esm_meta.loc[esm_mask, 'subcategory'].values.astype(str)
    esm_sil = silhouette_samples(esm_pca_50d[esm_mask], esm_labels)

    # SaProt
    sap_pca_2d = np.load('saprot_control/data/pca_2d.npy')
    sap_pca_50d = np.load('saprot_control/data/pca_50d.npy')
    sap_var = np.load('saprot_control/data/explained_variance.npy')
    sap_index = list(csv.DictReader(open('saprot_control/data/saprot_index.csv')))
    sap_labels = np.array([r['analysis_label'] for r in sap_index])
    sap_detail = np.array([r['label'] for r in sap_index])
    sap_sil = silhouette_samples(sap_pca_50d, sap_labels)

    return dict(
        esm_pca_2d=esm_pca_2d, esm_pca_50d=esm_pca_50d, esm_meta=esm_meta,
        esm_var=esm_var, esm_mask=esm_mask, esm_labels=esm_labels, esm_sil=esm_sil,
        sap_pca_2d=sap_pca_2d, sap_pca_50d=sap_pca_50d, sap_var=sap_var,
        sap_labels=sap_labels, sap_detail=sap_detail, sap_sil=sap_sil,
    )


# ══════════════════════════════════════════════════════════════════════
# Panel (a): PCA scatter — ESM-2 (left) vs SaProt (right)
# ══════════════════════════════════════════════════════════════════════
def draw_panel_a(ax1, ax2, D):
    for ax, pca_2d, meta_or_labels, var, side in [
        (ax1, D['esm_pca_2d'], D['esm_meta'], D['esm_var'], 'esm'),
        (ax2, D['sap_pca_2d'], D['sap_labels'], D['sap_var'], 'sap'),
    ]:
        if side == 'esm':
            sub = meta_or_labels['subcategory'].values
            anchor_m = sub == 'anchor'
            knotted_m = sub == 'knotted'
            fs_m = sub == 'fold_switching'
        else:
            anchor_m = meta_or_labels == 'anchor'
            knotted_m = meta_or_labels == 'knotted'
            fs_m = meta_or_labels == 'fold_switching'

        # Anchor background
        ax.scatter(pca_2d[anchor_m, 0], pca_2d[anchor_m, 1],
                   c='lightgray', s=1, alpha=0.3, edgecolors='none', rasterized=True)
        # Knotted
        ax.scatter(pca_2d[knotted_m, 0], pca_2d[knotted_m, 1],
                   c=COLORS['knotted'], s=12, alpha=0.8, edgecolors='white',
                   linewidths=0.3, zorder=5, rasterized=True)
        # Fold-switching
        ax.scatter(pca_2d[fs_m, 0], pca_2d[fs_m, 1],
                   c=COLORS['fold_switching'], s=12, alpha=0.8, edgecolors='white',
                   linewidths=0.3, zorder=5, rasterized=True)

        ax.set_xlabel(f'PC1 ({var[0]*100:.1f}%)')
        ax.set_ylabel(f'PC2 ({var[1]*100:.1f}%)')
        ax.grid(True, alpha=0.2, linewidth=0.5)

        # Legend
        handles = [
            Line2D([0], [0], marker='o', color='w', markerfacecolor='lightgray',
                   markersize=5, label=f'Anchor (n={anchor_m.sum()})', linestyle='None'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor=COLORS['knotted'],
                   markersize=5, label=f'Knotted (n={knotted_m.sum()})', linestyle='None'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor=COLORS['fold_switching'],
                   markersize=5, label=f'Fold-switching (n={fs_m.sum()})', linestyle='None'),
        ]
        ax.legend(handles=handles, loc='upper left', frameon=True, framealpha=0.9, fontsize=6)

        # Silhouette annotation box
        sil = D['esm_sil'] if side == 'esm' else D['sap_sil']
        lab = D['esm_labels'] if side == 'esm' else D['sap_labels']
        lines = []
        for cat in ['knotted', 'fold_switching']:
            m = lab == cat
            lines.append(f'{CAT_DISPLAY[cat]}: {sil[m].mean():+.3f}')
        props = dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='gray')
        ax.text(0.02, 0.02, '\n'.join(lines), transform=ax.transAxes,
                fontsize=6, va='bottom', ha='left', bbox=props, family='monospace')

    # Sub-panel labels
    ax1.set_title('ESM-2 (sequence only)', fontsize=9, fontweight='bold', pad=4)
    ax2.set_title('SaProt (sequence + structure)', fontsize=9, fontweight='bold', pad=4)


# ══════════════════════════════════════════════════════════════════════
# Panel (b): Silhouette distributions — split violin
# ══════════════════════════════════════════════════════════════════════
def draw_panel_b(ax, D):
    categories = ['knotted', 'fold_switching']
    positions = [0, 1]
    width = 0.35

    for i, cat in enumerate(categories):
        esm_s = D['esm_sil'][D['esm_labels'] == cat]
        sap_s = D['sap_sil'][D['sap_labels'] == cat]

        # Left half-violin: ESM-2
        vp1 = ax.violinplot([esm_s], positions=[positions[i] - width/2],
                            showmeans=False, showextrema=False, widths=width)
        for body in vp1['bodies']:
            m = np.mean(body.get_paths()[0].vertices[:, 0])
            body.get_paths()[0].vertices[:, 0] = np.clip(
                body.get_paths()[0].vertices[:, 0], -np.inf, m)
            body.set_facecolor(COLORS['esm2'])
            body.set_edgecolor('black')
            body.set_linewidth(0.5)
            body.set_alpha(0.7)

        # Right half-violin: SaProt
        vp2 = ax.violinplot([sap_s], positions=[positions[i] + width/2],
                            showmeans=False, showextrema=False, widths=width)
        for body in vp2['bodies']:
            m = np.mean(body.get_paths()[0].vertices[:, 0])
            body.get_paths()[0].vertices[:, 0] = np.clip(
                body.get_paths()[0].vertices[:, 0], m, np.inf)
            body.set_facecolor(COLORS['saprot'])
            body.set_edgecolor('black')
            body.set_linewidth(0.5)
            body.set_alpha(0.7)

        # Individual points (strip/jitter)
        rng = np.random.RandomState(42)
        jitter_e = rng.uniform(-0.08, 0.02, size=len(esm_s))
        jitter_s = rng.uniform(-0.02, 0.08, size=len(sap_s))
        ax.scatter(positions[i] - width/2 + jitter_e, esm_s,
                   c=COLORS['esm2'], s=2, alpha=0.25, edgecolors='none', zorder=3)
        ax.scatter(positions[i] + width/2 + jitter_s, sap_s,
                   c=COLORS['saprot'], s=2, alpha=0.25, edgecolors='none', zorder=3)

        # Mean markers
        ax.scatter([positions[i] - width/2], [esm_s.mean()],
                   c=COLORS['esm2'], s=50, marker='D', edgecolors='black',
                   linewidths=0.8, zorder=10)
        ax.scatter([positions[i] + width/2], [sap_s.mean()],
                   c=COLORS['saprot'], s=50, marker='D', edgecolors='black',
                   linewidths=0.8, zorder=10)

        # Significance bracket
        u_stat, p_val = stats.mannwhitneyu(sap_s, esm_s, alternative='greater')
        pooled_std = np.sqrt(((len(sap_s)-1)*sap_s.std()**2 +
                              (len(esm_s)-1)*esm_s.std()**2) / (len(sap_s)+len(esm_s)-2))
        d = (sap_s.mean() - esm_s.mean()) / pooled_std

        ymax = max(esm_s.max(), sap_s.max())
        bracket_y = ymax + 0.012
        ax.plot([positions[i] - width/2, positions[i] - width/2,
                 positions[i] + width/2, positions[i] + width/2],
                [bracket_y - 0.005, bracket_y, bracket_y, bracket_y - 0.005],
                color='black', linewidth=0.8)

        sig_text = f'Mann-Whitney U, p = {p_val:.2e}, d = {d:.2f}'
        ax.text(positions[i], bracket_y + 0.004, sig_text,
                ha='center', va='bottom', fontsize=5.5)

    # Zero line
    ax.axhline(y=0, color='black', linestyle='--', linewidth=0.8, alpha=0.5)

    ax.set_xticks(positions)
    ax.set_xticklabels([CAT_DISPLAY[c] for c in categories], fontsize=8)
    ax.set_ylabel('Silhouette Score (50D PCA)', fontsize=8)
    ax.grid(True, axis='y', alpha=0.2, linewidth=0.5)

    # Legend
    handles = [
        mpatches.Patch(facecolor=COLORS['esm2'], edgecolor='black',
                       linewidth=0.5, alpha=0.7, label='ESM-2 (seq only)'),
        mpatches.Patch(facecolor=COLORS['saprot'], edgecolor='black',
                       linewidth=0.5, alpha=0.7, label='SaProt (seq + struct)'),
        Line2D([0], [0], marker='D', color='w', markerfacecolor='gray',
               markeredgecolor='black', markersize=6, label='Mean', linestyle='None'),
    ]
    ax.legend(handles=handles, loc='lower right', frameon=True, framealpha=0.9, fontsize=6)


# ══════════════════════════════════════════════════════════════════════
# Panel (c): Fold-switching conf1 vs conf2 in SaProt
# ══════════════════════════════════════════════════════════════════════
def draw_panel_c(ax, D):
    fs_mask = np.array([l.startswith('fold_switching') for l in D['sap_detail']])
    fs_pca = D['sap_pca_2d'][fs_mask]
    fs_labels = D['sap_detail'][fs_mask]
    fs_pca_50d = D['sap_pca_50d'][fs_mask]

    conf1_m = fs_labels == 'fold_switching_conf1'
    conf2_m = fs_labels == 'fold_switching_conf2'

    # Background: all SaProt anchor as context
    anchor_m = D['sap_labels'] == 'anchor'
    ax.scatter(D['sap_pca_2d'][anchor_m, 0], D['sap_pca_2d'][anchor_m, 1],
               c='#eeeeee', s=1, alpha=0.2, edgecolors='none', rasterized=True,
               label='_nolegend_')

    # Conf1
    ax.scatter(fs_pca[conf1_m, 0], fs_pca[conf1_m, 1],
               c=COLORS['conf1'], s=20, alpha=0.85, edgecolors='black',
               linewidths=0.3, zorder=5, marker='o')
    # Conf2
    ax.scatter(fs_pca[conf2_m, 0], fs_pca[conf2_m, 1],
               c=COLORS['conf2'], s=20, alpha=0.85, edgecolors='black',
               linewidths=0.3, zorder=5, marker='^')

    ax.set_xlabel(f'PC1 ({D["sap_var"][0]*100:.1f}%)')
    ax.set_ylabel(f'PC2 ({D["sap_var"][1]*100:.1f}%)')
    ax.grid(True, alpha=0.2, linewidth=0.5)

    # Silhouette annotation
    fs_sil = silhouette_score(fs_pca_50d, fs_labels)
    fs_samples = silhouette_samples(fs_pca_50d, fs_labels)
    c1_sil = fs_samples[conf1_m].mean()
    c2_sil = fs_samples[conf2_m].mean()

    props = dict(boxstyle='round', facecolor='white', alpha=0.6, edgecolor='gray')
    annot = (f'Conf1 vs Conf2 Sil: {fs_sil:.3f}\n'
             f'Conf1: {c1_sil:+.3f} (n={conf1_m.sum()})\n'
             f'Conf2: {c2_sil:+.3f} (n={conf2_m.sum()})')
    ax.text(0.98, 0.98, annot, transform=ax.transAxes,
            fontsize=7, va='top', ha='right', bbox=props, family='monospace')

    # Legend
    handles = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor=COLORS['conf1'],
               markeredgecolor='black', markersize=6, label=f'Conf 1 (n={conf1_m.sum()})',
               linestyle='None'),
        Line2D([0], [0], marker='^', color='w', markerfacecolor=COLORS['conf2'],
               markeredgecolor='black', markersize=6, label=f'Conf 2 (n={conf2_m.sum()})',
               linestyle='None'),
    ]
    ax.legend(handles=handles, loc='upper left', frameon=True, framealpha=0.6, fontsize=7)

    return fs_sil, c1_sil, c2_sil, conf1_m.sum(), conf2_m.sum()


# ══════════════════════════════════════════════════════════════════════
# Main: composite figure
# ══════════════════════════════════════════════════════════════════════
def main():
    print("=" * 70)
    print("Figure 7: SaProt Structure-Aware Control")
    print("=" * 70)

    D = load_data()

    # ── Layout: top row = panel (a) with 2 subplots; bottom row = (b) + (c) ──
    fig = plt.figure(figsize=(12, 10))
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.30, wspace=0.28,
                           height_ratios=[1, 1])

    # Panel (a): top row, two subplots
    ax_a1 = fig.add_subplot(gs[0, 0])
    ax_a2 = fig.add_subplot(gs[0, 1])
    draw_panel_a(ax_a1, ax_a2, D)

    # Panel (b): bottom-left
    ax_b = fig.add_subplot(gs[1, 0])
    draw_panel_b(ax_b, D)

    # Panel (c): bottom-right
    ax_c = fig.add_subplot(gs[1, 1])
    fs_sil, c1_sil, c2_sil, n1, n2 = draw_panel_c(ax_c, D)

    # Sub-figure labels a, b, c (no parentheses)
    for ax, label in [(ax_a1, 'a'), (ax_b, 'b'), (ax_c, 'c')]:
        ax.text(-0.12, 1.05, label, transform=ax.transAxes,
                fontsize=12, fontweight='bold', va='bottom', ha='left')

    out = 'manuscript/fig7_saprot_control.png'
    plt.savefig(out, dpi=600, bbox_inches='tight')
    plt.close()
    print(f"\n  Figure saved: {out}")

    # ══════════════════════════════════════════════════════════════
    # Terminal output for figure caption
    # ══════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("FIGURE 7 CAPTION DATA")
    print("=" * 70)

    # Panel (a) data
    print("\n── Panel (a): PCA Scatter ──")
    print(f"  ESM-2:  PC1={D['esm_var'][0]*100:.1f}%, PC2={D['esm_var'][1]*100:.1f}%")
    print(f"  SaProt: PC1={D['sap_var'][0]*100:.1f}%, PC2={D['sap_var'][1]*100:.1f}%")
    for name, lab, sil in [('ESM-2', D['esm_labels'], D['esm_sil']),
                           ('SaProt', D['sap_labels'], D['sap_sil'])]:
        print(f"  {name}:")
        for cat in ['anchor', 'knotted', 'fold_switching']:
            m = lab == cat
            print(f"    {CAT_DISPLAY[cat]:20s} n={m.sum():>4d}  Sil={sil[m].mean():+.3f}")

    # Panel (b) data
    print("\n── Panel (b): Silhouette Distributions ──")
    print(f"  {'Category':<20s} {'ESM-2':>10s} {'SaProt':>10s} {'Delta':>8s}  {'p':>12s}  {'d':>6s}")
    print(f"  {'-'*70}")
    for cat in ['knotted', 'fold_switching']:
        e = D['esm_sil'][D['esm_labels'] == cat]
        s = D['sap_sil'][D['sap_labels'] == cat]
        u, p = stats.mannwhitneyu(s, e, alternative='greater')
        pooled = np.sqrt(((len(s)-1)*s.std()**2 + (len(e)-1)*e.std()**2) / (len(s)+len(e)-2))
        d = (s.mean() - e.mean()) / pooled
        print(f"  {CAT_DISPLAY[cat]:<20s} {e.mean():>+10.3f} {s.mean():>+10.3f} "
              f"{s.mean()-e.mean():>+8.3f}  p={p:.2e}  d={d:.2f}")

    print(f"\n  Fraction positive Silhouette:")
    for cat in ['knotted', 'fold_switching']:
        e = D['esm_sil'][D['esm_labels'] == cat]
        s = D['sap_sil'][D['sap_labels'] == cat]
        print(f"    {CAT_DISPLAY[cat]}: ESM-2 {(e>0).mean()*100:.0f}% -> SaProt {(s>0).mean()*100:.0f}%")

    # Panel (c) data
    print(f"\n── Panel (c): Fold-switching Conf1 vs Conf2 ──")
    print(f"  Conf1: n={n1}, mean Sil={c1_sil:+.4f}")
    print(f"  Conf2: n={n2}, mean Sil={c2_sil:+.4f}")
    print(f"  Overall Silhouette (conf1 vs conf2): {fs_sil:.4f}")
    print(f"  Verdict: {'Indistinguishable' if abs(fs_sil) < 0.05 else 'Weakly separated'}")

    # Paired distance analysis
    fs_mask = np.array([l.startswith('fold_switching') for l in D['sap_detail']])
    fs_50d = D['sap_pca_50d'][fs_mask]
    fs_lab = D['sap_detail'][fs_mask]
    c1_pts = fs_50d[fs_lab == 'fold_switching_conf1']
    c2_pts = fs_50d[fs_lab == 'fold_switching_conf2']
    n_pairs = min(len(c1_pts), len(c2_pts))
    paired_dists = np.linalg.norm(c1_pts[:n_pairs] - c2_pts[:n_pairs], axis=1)
    rng = np.random.RandomState(42)
    all_dists = []
    for _ in range(1000):
        i, j = rng.choice(len(fs_50d), 2, replace=False)
        all_dists.append(np.linalg.norm(fs_50d[i] - fs_50d[j]))
    all_dists = np.array(all_dists)
    print(f"\n  Paired conf1<->conf2 distance: {paired_dists.mean():.2f} +/- {paired_dists.std():.2f}")
    print(f"  Random FS<->FS distance:       {all_dists.mean():.2f} +/- {all_dists.std():.2f}")
    t_d, p_d = stats.ttest_ind(paired_dists, all_dists)
    print(f"  t-test (paired vs random): t={t_d:.2f}, p={p_d:.3f}")
    if p_d > 0.05:
        print(f"  -> Paired conformations are NOT closer than random FS pairs")
    else:
        print(f"  -> Paired conformations {'closer' if t_d < 0 else 'farther'} than random")

    # Distributional detail
    print(f"\n── Distributional Detail ──")
    for name, lab, sil in [('ESM-2', D['esm_labels'], D['esm_sil']),
                           ('SaProt', D['sap_labels'], D['sap_sil'])]:
        print(f"  {name}:")
        print(f"    {'Category':<20s} {'Mean':>7s} {'Median':>7s} {'IQR':>18s} {'%pos':>5s} {'%neg':>5s}")
        for cat in ['anchor', 'knotted', 'fold_switching']:
            m = lab == cat
            vals = sil[m]
            q25, q75 = np.percentile(vals, [25, 75])
            pct_pos = (vals > 0).mean() * 100
            pct_neg = (vals < 0).mean() * 100
            print(f"    {CAT_DISPLAY[cat]:<20s} {vals.mean():>+7.3f} {np.median(vals):>+7.3f} "
                  f"[{q25:+.3f}, {q75:+.3f}] {pct_pos:>4.0f}% {pct_neg:>4.0f}%")

    # 5-NN balanced accuracy
    print(f"\n── 5-NN Balanced Accuracy ──")
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.model_selection import StratifiedKFold
    from sklearn.metrics import balanced_accuracy_score
    for name, X, y in [('ESM-2', D['esm_pca_50d'][D['esm_mask']], D['esm_labels']),
                        ('SaProt', D['sap_pca_50d'], D['sap_labels'])]:
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        accs = []
        for train_idx, test_idx in skf.split(X, y):
            clf = KNeighborsClassifier(n_neighbors=5)
            clf.fit(X[train_idx], y[train_idx])
            pred = clf.predict(X[test_idx])
            accs.append(balanced_accuracy_score(y[test_idx], pred))
        print(f"  {name}: {np.mean(accs):.3f} +/- {np.std(accs):.3f}  (chance=0.333)")

    # Dataset summary
    print(f"\n── Dataset Summary ──")
    for name, lab in [('ESM-2', D['esm_labels']), ('SaProt', D['sap_labels'])]:
        print(f"  {name}: total={len(lab)}")
        for cat in ['anchor', 'knotted', 'fold_switching']:
            print(f"    {cat}: {(lab==cat).sum()}")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
