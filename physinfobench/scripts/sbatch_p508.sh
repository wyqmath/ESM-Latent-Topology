#!/usr/bin/env bash
#SBATCH -J p508_eval
#SBATCH -p NV_4090D
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 04:00:00
#SBATCH -o /lenovofs1/home/jyma/PLM_benchmark/logs/p508_eval_%j.out
set -eo pipefail
cd /lenovofs1/home/jyma/PLM_benchmark/p303_resplit/proj
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_ENDPOINT=https://hf-mirror.com
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=/lenovofs1/home/jyma/PLM_benchmark/p303_resplit/venv/bin/python
mkdir -p data/interim/p508_emb results/p508_disorder_contrast logs
$PY scripts/p508_cluster_eval.py
echo "[p508] DONE $(date -u +%Y-%m-%dT%H:%M:%SZ)"
