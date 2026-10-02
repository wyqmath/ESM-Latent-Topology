#!/usr/bin/env bash
# P4.05/P4.06-08 前置：条件系统（PYP+RNase A）池化嵌入抽取（hidden 33）。
#SBATCH -J p405_cond_emb
#SBATCH -p NV_4090D
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=32G
#SBATCH -t 00:30:00
#SBATCH -o /lenovofs1/home/jyma/PLM_benchmark/logs/p405_cond_emb_%j.out

set -eo pipefail
cd /lenovofs1/home/jyma/PLM_benchmark/p303_resplit/proj
PY=/lenovofs1/home/jyma/PLM_benchmark/p303_resplit/venv/bin/python
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_ENDPOINT=https://hf-mirror.com
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

$PY scripts/extract_representations.py \
  --fasta data/interim/p405/conditional.fa \
  --out-dir data/interim/p405/emb_conditional \
  --mean-layers 33 \
  --device cuda --batch-max-tokens 8192

echo "[p405-embed] DONE $(date -u +%Y-%m-%dT%H:%M:%SZ)"
