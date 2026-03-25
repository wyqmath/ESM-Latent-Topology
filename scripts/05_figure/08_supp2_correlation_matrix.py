# -- Project root setup --
import os as _os
from pathlib import Path as _Path

_os.chdir(_Path(__file__).resolve().parents[2])
"""
Supplementary Figure S2: Category-decoupled correlation analysis

Output:
- manuscript/supp2.png (DPI=600)

Panels (2x3):
(a) Global (n=9,195 with E[n])
(b) Anchor (n=856)
(c) Astral95 (n=8,289)
(d) Integrable (n=50)
(e) IDP (E[n] masked)
(f) Knotted (E[n] masked)
"""

from pathlib import Path

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from matplotlib.patches import Rectangle
from scipy.stats import gaussian_kde, spearmanr
from sklearn.neighbors import NearestNeighbors

plt.rcParams["font.family"] = "DejaVu Sans"
plt.rcParams["font.size"] = 10
plt.rcParams["figure.dpi"] = 600
plt.rcParams["savefig.dpi"] = 600

VARS = [
    "Local Density",
    "Local Dimension",
    "Nearest Distance",
    "Ricci Curvature",
    "Condition Number (log10)",
    "E[n]",
]
DISPLAY_VARS = [
    "Density",
    "Dimension",
    "Nearest dist",
    "Ricci",
    "Cond# (log10)",
    "E[n]",
]
EN_IDX = VARS.index("E[n]")


def assign_category(seq_id: str) -> str:
    prefix = seq_id.split("|")[0]
    if prefix == "anchor":
        return "anchor"
    if prefix == "astral95":
        return "astral95"
    if prefix == "integrable":
        return "integrable"
    if prefix == "control":
        return "random"
    if prefix == "idp":
        return "idp"
    if prefix == "knotted":
        return "knotted"
    if prefix == "fold_switching":
        return "fold_switching"
    return "unknown"


def compute_local_features(pca_50d: np.ndarray, k: int = 20) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    nn = NearestNeighbors(n_neighbors=k + 1, metric="euclidean")
    nn.fit(pca_50d)
    distances, _ = nn.kneighbors(pca_50d)
    knn_distances = distances[:, 1:]  # exclude self

    local_density = 1.0 / knn_distances.mean(axis=1)
    nearest_dist = knn_distances[:, 0]

    r_k = knn_distances[:, -1]
    local_dim = np.full(pca_50d.shape[0], np.nan)
    eps = 1e-12
    for i in range(pca_50d.shape[0]):
        denom = np.maximum(knn_distances[i, :-1], eps)
        vals = np.log(np.maximum(r_k[i], eps) / denom)
        vals = vals[np.isfinite(vals)]
        if len(vals) > 0 and np.sum(vals) != 0:
            local_dim[i] = len(vals) / np.sum(vals)

    return local_density, local_dim, nearest_dist


def spearman_matrix(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    n_vars = len(df.columns)
    corr = np.full((n_vars, n_vars), np.nan)
    pval = np.full((n_vars, n_vars), np.nan)

    for i, c1 in enumerate(df.columns):
        for j, c2 in enumerate(df.columns):
            if i == j:
                corr[i, j] = 1.0
                pval[i, j] = 0.0
                continue
            mask = ~(df[c1].isna() | df[c2].isna() | np.isinf(df[c1]) | np.isinf(df[c2]))
            if mask.sum() > 10:
                rho, p = spearmanr(df.loc[mask, c1], df.loc[mask, c2])
                corr[i, j] = rho
                pval[i, j] = p

    return corr, pval


def sig_label(p: float) -> str:
    if np.isnan(p):
        return ""
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return "ns"


def draw_mask_cell(ax, show_na: bool = False) -> None:
    ax.add_patch(
        Rectangle(
            (0, 0),
            1,
            1,
            transform=ax.transAxes,
            facecolor="#d9d9d9",
            edgecolor="#9a9a9a",
            hatch="///",
            linewidth=0.6,
            zorder=3,
        )
    )
    if show_na:
        ax.text(0.5, 0.5, "N/A", transform=ax.transAxes, ha="center", va="center", fontsize=7.5, fontweight="bold")


def draw_pairgrid_panel(
    fig,
    outer_spec,
    corr: np.ndarray,
    pval: np.ndarray,
    data_cols: list[np.ndarray],
    label: str,
    panel_title: str,
    n_panel: int,
    masked_en: bool,
) -> None:
    # background axis for centered panel letter only
    ax_bg = fig.add_subplot(outer_spec, frameon=False)
    ax_bg.set_xticks([])
    ax_bg.set_yticks([])
    ax_bg.set_title(f"{label} ({panel_title}, n={n_panel:,})", fontsize=13, fontweight="bold", pad=8)

    inner = GridSpecFromSubplotSpec(6, 6, subplot_spec=outer_spec, wspace=0.03, hspace=0.03)
    cmap = plt.cm.RdBu_r
    norm = mcolors.Normalize(vmin=-1, vmax=1)

    rng = np.random.default_rng(42)

    for i in range(6):
        for j in range(6):
            ax = fig.add_subplot(inner[i, j])
            ax.set_xticks([])
            ax.set_yticks([])

            if masked_en and (i == EN_IDX or j == EN_IDX):
                draw_mask_cell(ax, show_na=(i == EN_IDX and j == EN_IDX))
            elif i == j:
                vals = data_cols[i]
                vals = vals[np.isfinite(vals)]
                if len(vals) > 1:
                    try:
                        kde = gaussian_kde(vals)
                        x_kde = np.linspace(vals.min(), vals.max(), 100)
                        y_kde = kde(x_kde)
                        ax.fill_between(x_kde, y_kde, alpha=0.6, color="#3498db")
                        ax.plot(x_kde, y_kde, color="#2c3e50", linewidth=0.6)
                        ax.set_xlim(vals.min(), vals.max())
                    except Exception:
                        ax.hist(vals, bins=16, color="#3498db", alpha=0.6, density=True)
            elif i > j:
                v = corr[i, j]
                if np.isfinite(v):
                    ax.set_facecolor(cmap(norm(v)))
                    text_color = "white" if abs(v) > 0.4 else "black"
                    ax.text(
                        0.5,
                        0.5,
                        f"{v:.2f}",
                        transform=ax.transAxes,
                        ha="center",
                        va="center",
                        fontsize=7,
                        fontweight="bold",
                        color=text_color,
                    )
            else:
                x = data_cols[j]
                y = data_cols[i]
                mask = np.isfinite(x) & np.isfinite(y)
                idx = np.where(mask)[0]
                if len(idx) > 0:
                    n_plot = min(280, len(idx))
                    sel = rng.choice(idx, n_plot, replace=False)
                    ax.scatter(x[sel], y[sel], s=0.6, alpha=0.3, c="#34495e", edgecolors="none", rasterized=True)
                    p = pval[i, j]
                    sig = sig_label(p)
                    ax.text(
                        0.95,
                        0.95,
                        sig,
                        transform=ax.transAxes,
                        ha="right",
                        va="top",
                        fontsize=6.5,
                        fontweight="bold",
                        color="red" if p < 0.05 else "gray",
                    )

            if i == 5:
                ax.set_xlabel(DISPLAY_VARS[j], fontsize=5.2, rotation=45, ha="right")
            if j == 0:
                ax.set_ylabel(DISPLAY_VARS[i], fontsize=5.2)


def strongest_pair(corr: np.ndarray, var_names: list[str], exclude_en: bool) -> tuple[str, str, float]:
    best = ("", "", -1.0)
    for i in range(len(var_names)):
        for j in range(i + 1, len(var_names)):
            if exclude_en and (i == EN_IDX or j == EN_IDX):
                continue
            v = corr[i, j]
            if np.isnan(v):
                continue
            if abs(v) > abs(best[2]):
                best = (var_names[i], var_names[j], v)
    return best


def pair_stats(
    corr: np.ndarray,
    pval: np.ndarray,
    var_names: list[str],
    exclude_en: bool,
) -> list[tuple[str, str, float, float]]:
    rows: list[tuple[str, str, float, float]] = []
    for i in range(len(var_names)):
        for j in range(i + 1, len(var_names)):
            if exclude_en and (i == EN_IDX or j == EN_IDX):
                continue
            r = corr[i, j]
            p = pval[i, j]
            if np.isfinite(r) and np.isfinite(p):
                rows.append((var_names[i], var_names[j], float(r), float(p)))
    return rows


def pair_rho_p(
    corr: np.ndarray,
    pval: np.ndarray,
    var_names: list[str],
    v1: str,
    v2: str,
) -> tuple[float, float]:
    i = var_names.index(v1)
    j = var_names.index(v2)
    return float(corr[i, j]), float(pval[i, j])


def print_pair_rows(title: str, rows: list[tuple[str, str, float, float]], limit: int = 5) -> None:
    print(f"    {title}")
    if not rows:
        print("      (none)")
        return
    for a, b, r, p in rows[:limit]:
        print(f"      {a:<22s} vs {b:<22s}: rho={r:+.4f}, p={p:.2e}")


if __name__ == "__main__":
    print("=" * 78)
    print("Supplementary Figure S2: Category-decoupled correlation analysis")
    print("=" * 78)

    print("\n[1/5] Loading data...")
    pca_50d = np.load("data/pca/pca_embeddings_50d.npy")
    density = np.load("data/density/density_values.npy")
    curvature = np.load("data/curvature/ricci_curvature.npy")
    condition_numbers = np.load("data/condition_number/condition_numbers.npy")
    meta = pd.read_csv("data/metadata_final_with_en.csv", usecols=["seq_id", "E_n"])
    embed_index = pd.read_csv("data/embeddings/embedding_index_final.csv", usecols=["seq_id", "index"])

    print(f"  Embeddings: {pca_50d.shape}")
    print(f"  Metadata rows: {len(meta):,}")
    print(f"  Embedding index rows: {len(embed_index):,}")

    print("\n[2/5] Computing local geometric features (k=20)...")
    local_density, local_dim, nearest_dist = compute_local_features(pca_50d, k=20)

    print("\n[3/5] Building unified table...")
    features = pd.DataFrame(
        {
            "index": np.arange(len(pca_50d)),
            "Local Density": local_density,
            "Local Dimension": local_dim,
            "Nearest Distance": nearest_dist,
            "Ricci Curvature": curvature,
            "Condition Number (log10)": np.log10(condition_numbers + 1),
            "Density (KDE)": density,
        }
    )

    df = embed_index.merge(features, on="index", how="left")
    df = df.merge(meta, on="seq_id", how="left")
    df["category"] = df["seq_id"].map(assign_category)
    df = df.rename(columns={"E_n": "E[n]"})

    print(f"  Unified table rows: {len(df):,}")
    print(f"  E[n] available: {df['E[n]'].notna().sum():,}")

    panel_specs = [
        ("Global", df["E[n]"].notna(), False),
        ("Anchor", (df["category"] == "anchor") & df["E[n]"].notna(), False),
        ("Astral95", (df["category"] == "astral95") & df["E[n]"].notna(), False),
        ("Integrable", (df["category"] == "integrable") & df["E[n]"].notna(), False),
        ("IDP", df["category"] == "idp", True),
        ("Knotted", df["category"] == "knotted", True),
    ]
    panel_letters = ["a", "b", "c", "d", "e", "f"]

    print("\n[4/5] Computing panel-wise correlations...")
    results = []
    caption_rows = []
    strongest_pairs_by_panel: dict[str, tuple[str, str, float]] = {}
    recurring_counter: dict[tuple[str, str], int] = {}

    for title, mask, masked_en in panel_specs:
        sub = df.loc[mask, VARS].copy()
        corr, pval = spearman_matrix(sub)
        data_cols = [sub[v].to_numpy() for v in VARS]
        n_panel = int(mask.sum())
        results.append((title, n_panel, masked_en, corr, pval, data_cols))

        print(f"  {title:<10s}: n={n_panel:,}, E[n] masked={masked_en}")
        if not masked_en:
            en_ricci_rho = corr[EN_IDX, VARS.index("Ricci Curvature")]
            en_ricci_p = pval[EN_IDX, VARS.index("Ricci Curvature")]
            en_pairs = [(VARS[j], corr[EN_IDX, j], pval[EN_IDX, j]) for j in range(len(VARS)) if j != EN_IDX]
            best_en_var, best_en_rho, best_en_p = max(en_pairs, key=lambda t: abs(t[1]))
            print(f"    E[n] vs Ricci: rho={en_ricci_rho:.4f}, p={en_ricci_p:.2e}")
            print(f"    max |rho(E[n], ·)|: {best_en_var} -> rho={best_en_rho:.4f}, p={best_en_p:.2e}")
            caption_rows.append((title, n_panel, best_en_var, best_en_rho, best_en_p))
        else:
            caption_rows.append((title, n_panel, "E[n] masked", np.nan, np.nan))

        rows_all = pair_stats(corr, pval, VARS, exclude_en=masked_en)
        rows_abs = sorted(rows_all, key=lambda t: abs(t[2]), reverse=True)
        rows_sig = sorted([r for r in rows_all if r[3] < 0.05], key=lambda t: t[3])
        rows_nonsig = sorted([r for r in rows_all if r[3] >= 0.05], key=lambda t: abs(t[2]), reverse=True)

        if rows_abs:
            a, b, r, _ = rows_abs[0]
            strongest_pairs_by_panel[title] = (a, b, r)
            print(f"    strongest |rho|: {a} vs {b} -> rho={r:.4f}")

            top3_pairs = rows_abs[:3]
            for pa, pb, _, _ in top3_pairs:
                key = tuple(sorted((pa, pb)))
                recurring_counter[key] = recurring_counter.get(key, 0) + 1

        print_pair_rows("top 5 by |rho|:", rows_abs, limit=5)
        print_pair_rows("top 5 most significant (p<0.05):", rows_sig, limit=5)
        print_pair_rows("top 5 non-significant but largest |rho| (p>=0.05):", rows_nonsig, limit=5)

        nd_ld_rho, nd_ld_p = pair_rho_p(corr, pval, VARS, "Nearest Distance", "Local Dimension")
        nd_den_rho, nd_den_p = pair_rho_p(corr, pval, VARS, "Nearest Distance", "Local Density")
        ricci_ld_rho, ricci_ld_p = pair_rho_p(corr, pval, VARS, "Ricci Curvature", "Local Dimension")
        print(
            f"    key pair check: ND~Dim rho={nd_ld_rho:+.4f} (p={nd_ld_p:.2e}); "
            f"ND~Density rho={nd_den_rho:+.4f} (p={nd_den_p:.2e}); "
            f"Ricci~Dim rho={ricci_ld_rho:+.4f} (p={ricci_ld_p:.2e})"
        )

    print("\n[Caption stats] S2 panel summary for 图注.md")
    print("  (Global/Anchor/Astral95/Integrable report max |rho(E[n],·)|; IDP/Knotted are masked)")
    for title, n_panel, best_var, best_rho, best_p in caption_rows:
        if np.isfinite(best_rho):
            print(
                f"  - {title:<10s} n={n_panel:>5,}: max |rho(E[n],·)| = {best_rho:+.4f} "
                f"({best_var}, p={best_p:.2e})"
            )
        else:
            print(f"  - {title:<10s} n={n_panel:>5,}: E[n] row/column masked (N/A)")

    print("\n[Cross-panel pattern summary]")
    if strongest_pairs_by_panel:
        print("  strongest |rho| pair per panel:")
        for panel_name, (a, b, r) in strongest_pairs_by_panel.items():
            print(f"  - {panel_name:<10s}: {a} vs {b} (rho={r:+.4f})")

    recurring_sorted = sorted(recurring_counter.items(), key=lambda t: (-t[1], t[0]))
    print("  most recurrent pairs among panel-wise top-3 |rho|:")
    for (a, b), c in recurring_sorted[:5]:
        print(f"  - {a} vs {b}: appears in {c} panel(s)")

    print("\n[5/5] Rendering and saving figure...")
    fig = plt.figure(figsize=(22, 14))
    outer = GridSpec(2, 3, figure=fig, wspace=0.12, hspace=0.17)

    for spec, (title, n_panel, masked_en, corr, pval, data_cols), letter in zip(outer, results, panel_letters):
        draw_pairgrid_panel(
            fig=fig,
            outer_spec=spec,
            corr=corr,
            pval=pval,
            data_cols=data_cols,
            label=letter,
            panel_title=title,
            n_panel=n_panel,
            masked_en=masked_en,
        )

    sm = plt.cm.ScalarMappable(cmap=plt.cm.RdBu_r, norm=mcolors.Normalize(vmin=-1, vmax=1))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=fig.axes, fraction=0.02, pad=0.01)
    cbar.set_label("Spearman ρ", fontsize=12)

    out_path = Path("manuscript/supp2.png")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=600, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    print(f"Saved: {out_path}")
    print("=" * 78)
