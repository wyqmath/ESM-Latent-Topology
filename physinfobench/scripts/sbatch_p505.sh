#!/usr/bin/env bash
# P5.05 保留集泛化评估（集群侧，final_lock 冻结执行）。
# 前置门禁：final_lock.gate.user_authorization_recorded 非空（用户明确授权）；
#          运行器内部强校验，缺失即 die。
#SBATCH -J p505_holdout
#SBATCH -p NV_4090D
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=96G
#SBATCH -t 04:00:00
#SBATCH -o /lenovofs1/home/jyma/PLM_benchmark/logs/p505_holdout_%j.out

set -eo pipefail
cd /lenovofs1/home/jyma/PLM_benchmark/p303_resplit/proj
PY=/lenovofs1/home/jyma/PLM_benchmark/p303_resplit/venv/bin/python
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_ENDPOINT=https://hf-mirror.com
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

mkdir -p data/interim/p504 data/interim/p505 results/holdout logs

$PY scripts/run_p505_holdout.py --step all

echo "[p505] DONE $(date -u +%Y-%m-%dT%H:%M:%SZ)"
