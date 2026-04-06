#!/usr/bin/env python3
"""
Step 5: PCA + Silhouette analysis.
- StandardScaler → PCA(50D) → silhouette_samples (3-class: anchor/knotted/fold_switching)
- Also compute for fold-switching conf1 vs conf2 separately
Output: saprot_control/data/pca_50d.npy, pca_2d.npy, silhouette_scores.csv
"""
import os, csv
import torch
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_samples, silhouette_score

ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT)


def main():
    print("=" * 70)
    print("Step 5: PCA + Silhouette Analysis")
    print("=" * 70)

    # Load embeddings + index
    embeddings = torch.load('saprot_control/data/saprot_embeddings.pt',
                            map_location='cpu', weights_only=True).float().numpy()
    index = list(csv.DictReader(open('saprot_control/data/saprot_index.csv')))

    print(f"  Embeddings: {embeddings.shape}")
    print(f"  Index: {len(index)} entries")

    labels = [r['analysis_label'] for r in index]
    label_set = sorted(set(labels))
    print(f"  Labels: {label_set}")
    for l in label_set:
        print(f"    {l}: {labels.count(l)}")

    # StandardScaler + PCA
    print("\n  StandardScaler + PCA...")
    scaler = StandardScaler()
    embeddings_scaled = scaler.fit_transform(embeddings)

    pca_50 = PCA(n_components=50, random_state=42)
    pca_50d = pca_50.fit_transform(embeddings_scaled)
    print(f"  PCA 50D explained variance: {pca_50.explained_variance_ratio_.sum():.3f}")

    pca_2 = PCA(n_components=2, random_state=42)
    pca_2d = pca_2.fit_transform(embeddings_scaled)
    print(f"  PCA 2D explained variance: {pca_2.explained_variance_ratio_.sum():.3f}")

    # Save PCA
    np.save('saprot_control/data/pca_50d.npy', pca_50d)
    np.save('saprot_control/data/pca_2d.npy', pca_2d)
    np.save('saprot_control/data/explained_variance.npy', pca_2.explained_variance_ratio_)

    # ── Silhouette (3-class: anchor/knotted/fold_switching) ──
    print("\n  Computing Silhouette (3-class)...")
    labels_arr = np.array(labels)
    sil_samples = silhouette_samples(pca_50d, labels_arr)
    sil_overall = silhouette_score(pca_50d, labels_arr)

    print(f"  Overall Silhouette: {sil_overall:.4f}")
    print()
    for l in label_set:
        mask = labels_arr == l
        mean_sil = sil_samples[mask].mean()
        std_sil = sil_samples[mask].std()
        n = mask.sum()
        print(f"  {l:<25s}: mean={mean_sil:+.4f} ± {std_sil:.4f}  (n={n})")

    # ── Silhouette for fold-switching conf1 vs conf2 ──
    print("\n  Computing Silhouette for fold-switching conf1 vs conf2...")
    detail_labels = [r['label'] for r in index]
    detail_arr = np.array(detail_labels)

    fs_mask = np.array([l.startswith('fold_switching') for l in detail_labels])
    if fs_mask.sum() > 1:
        fs_pca = pca_50d[fs_mask]
        fs_labels = detail_arr[fs_mask]
        unique_fs = np.unique(fs_labels)
        if len(unique_fs) > 1:
            fs_sil = silhouette_score(fs_pca, fs_labels)
            print(f"  FS conf1 vs conf2 Silhouette: {fs_sil:.4f}")
            for l in unique_fs:
                m = fs_labels == l
                print(f"    {l}: mean={silhouette_samples(fs_pca, fs_labels)[m].mean():.4f} (n={m.sum()})")
        else:
            print(f"  Only one FS label found: {unique_fs}")

    # Save silhouette scores
    sil_path = Path('saprot_control/data/silhouette_scores.csv')
    with open(sil_path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['seq_id', 'label', 'analysis_label', 'silhouette'])
        for i, row in enumerate(index):
            w.writerow([row['seq_id'], row['label'], row['analysis_label'], f"{sil_samples[i]:.6f}"])
    print(f"\n  Saved: {sil_path}")

    # ── Summary stats for manuscript ──
    print("\n" + "=" * 70)
    print("SUMMARY FOR MANUSCRIPT")
    print("=" * 70)
    print(f"\nSaProt 3-class Silhouette (50D PCA):")
    print(f"  Overall: {sil_overall:.4f}")
    for l in ['anchor', 'knotted', 'fold_switching']:
        mask = labels_arr == l
        if mask.any():
            print(f"  {l}: {sil_samples[mask].mean():+.4f} (n={mask.sum()})")

    print(f"\nESM-2 reference (from main analysis):")
    print(f"  Knotted: -0.151")
    print(f"  Fold-switching: -0.108")
    print(f"  IDP: -0.057")

    print("\nDone!")


if __name__ == "__main__":
    main()
