#!/usr/bin/env bash
# P5.07 新链嵌入抽取（ESM-2 650M，global set 6 层均值，fp16）。
#SBATCH -J p507_embed
#SBATCH -p NV_4090D
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 02:00:00
#SBATCH -o /lenovofs1/home/jyma/PLM_benchmark/logs/p507_embed_%j.out

set -eo pipefail
cd /lenovofs1/home/jyma/PLM_benchmark/p303_resplit/proj
PY=/lenovofs1/home/jyma/PLM_benchmark/p303_resplit/venv/bin/python
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_ENDPOINT=https://hf-mirror.com
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

mkdir -p data/interim/p507_emb logs

$PY scripts/extract_representations.py \
  --fasta data/interim/p507_probe.fa \
  --out-dir data/interim/p507_emb \
  --mean-layers 5,11,17,23,29,33 \
  --device cuda \
  --batch-max-tokens 49152

echo "[p507embed] DONE $(date -u +%Y-%m-%dT%H:%M:%SZ)"
