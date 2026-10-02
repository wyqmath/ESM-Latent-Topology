#!/usr/bin/env bash
# P5.07 隔离重算：mmseqs 序列簇 + foldseek 预筛 + US-align 结构边确认。
# SLURM 纪律：--gres=gpu:1（CPU 计算占卡）。
#SBATCH -J p507_iso
#SBATCH -p NV_4090D
#SBATCH --gres=gpu:1
#SBATCH -c 16
#SBATCH --mem=64G
#SBATCH -t 12:00:00
#SBATCH -o /lenovofs1/home/jyma/PLM_benchmark/logs/p507_iso_%j.out

set -eo pipefail
export PATH=/lenovofs1/home/jyma/.conda/envs/plm_bench/bin:$PATH
PY=/lenovofs1/home/jyma/.conda/envs/plm_bench/bin/python
cd /lenovofs1/home/jyma/PLM_benchmark/p303_resplit/proj
mkdir -p data/interim/p507_isolate logs

$PY scripts/p507_isolate_cluster.py
echo "[p507iso] DONE $(date -u +%Y-%m-%dT%H:%M:%SZ)"
