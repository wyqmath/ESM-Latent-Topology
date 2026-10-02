#!/usr/bin/env bash
# P5.07 corrected representation cache: pinned ESM2, fp32 forward, versioned outputs.
#SBATCH -J p507_fp32
#SBATCH -p NV_4090D
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 04:00:00
#SBATCH -o /lenovofs1/home/jyma/PLM_benchmark/logs/p507_fp32_%j.out

set -euo pipefail
cd /lenovofs1/home/jyma/PLM_benchmark/p303_resplit/proj
PY=/lenovofs1/home/jyma/PLM_benchmark/p303_resplit/venv/bin/python
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

EMB_NEW=data/interim/p507_emb_20261002
EMB_OLD=data/interim/p507_knots_emb_20261002
mkdir -p "$EMB_NEW" "$EMB_OLD"

"$PY" scripts/extract_representations.py \
  --fasta data/interim/p507_probe.fa \
  --out-dir "$EMB_NEW" --mean-layers 5,11,17,23,29,33 \
  --device cuda --batch-max-tokens 16384 --append-manifest
"$PY" scripts/extract_representations.py \
  --fasta data/interim/p507_embed_extra.fa \
  --out-dir "$EMB_NEW" --mean-layers 5,11,17,23,29,33 \
  --device cuda --batch-max-tokens 16384 --append-manifest
"$PY" scripts/extract_representations.py \
  --fasta data/interim/p303/knots_dev.fa \
  --out-dir "$EMB_OLD" --resid-layers 23,29,33 \
  --device cuda --batch-max-tokens 16384 --append-manifest

echo "[p507-fp32] DONE $(date -u +%Y-%m-%dT%H:%M:%SZ)"
