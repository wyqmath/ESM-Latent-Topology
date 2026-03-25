# ESM-Latent-Topology

## Pipeline

```
01_data_extraction/          (7)   Extract & merge sequences from 7 datasets
02_embeddings/               (6)   Metadata, ESM-2 inference, region embeddings
03_dimensionality_reduction/ (2)   UMAP + PCA
04_analysis/                 (7)   PH, Ricci curvature, density, Wilson loops, etc.
05_figure/                   (15)  6 main figures + 9 supplementary figures
```

37 scripts total. Each auto-sets working directory to project root.

## External Data (not included)

Before running the pipeline, download and place the following:

1. **ASTRAL SCOPe 2.08 FASTA** → `data/raw/astral-scopedom-seqres-gd-sel-gs-bib-95-2.08.fa`
   - https://scop.berkeley.edu/downloads/scopeseq-2.08/
2. **PDB structure files** → `data/pdb_files/*.pdb`
   - ~8,300 PDB files (7,670 from ASTRAL95 + 852 from Anchor, 195 overlap)
   - Download from RCSB using IDs in `data/astral95_pdb_ids.txt` and `data/anchor_pdb_ids.txt`
3. **ESM-2 3B model weights** — auto-downloaded by `esm` package on first run

The integrability error E[n] and related Hasimoto-frame geometry are computed using code from [discrete_hasimoto_protein](https://github.com/wyqmath/discrete_hasimoto_protein).