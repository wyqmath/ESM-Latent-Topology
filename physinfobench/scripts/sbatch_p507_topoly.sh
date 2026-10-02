#!/usr/bin/env bash
# P5.07 扩库候选 Topoly Alexander 复核（主 c2t200 + 敏感性 c1t200）。
# SLURM 纪律：--gres=gpu:1（纯 CPU 计算也占卡，QOSMinGRES）；conda plm_bench（含 topoly）。
#SBATCH -J p507_topoly
#SBATCH -p NV_4090D
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=32G
#SBATCH -t 08:00:00
#SBATCH -o /lenovofs1/home/jyma/PLM_benchmark/logs/p507_topoly_%j.out

set -eo pipefail
export PATH=/lenovofs1/home/jyma/.conda/envs/plm_bench/bin:$PATH
PY=/lenovofs1/home/jyma/.conda/envs/plm_bench/bin/python
cd /lenovofs1/home/jyma/PLM_benchmark/p303_resplit/proj
mkdir -p data/interim/p507_topoly_results logs

$PY scripts/p507_topoly_verify.py
echo "[p507] DONE $(date -u +%Y-%m-%dT%H:%M:%SZ)"
