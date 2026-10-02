#!/usr/bin/env bash
# P5.07 修复：43 条冻结链补抽嵌入（增量写入 p507_emb 同库）。
#SBATCH -J p507_emb2
#SBATCH -p NV_4090D
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 01:00:00
#SBATCH -o /lenovofs1/home/jyma/PLM_benchmark/logs/p507_emb2_%j.out
set -eo pipefail
cd /lenovofs1/home/jyma/PLM_benchmark/p303_resplit/proj
PY=/lenovofs1/home/jyma/PLM_benchmark/p303_resplit/venv/bin/python
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_ENDPOINT=https://hf-mirror.com
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
$PY scripts/extract_representations.py --fasta data/interim/p507_embed_extra.fa \
  --out-dir data/interim/p507_emb --mean-layers 5,11,17,23,29,33 --device cuda \
  --batch-max-tokens 49152
echo "[p507emb2] DONE $(date -u +%Y-%m-%dT%H:%M:%SZ)"
