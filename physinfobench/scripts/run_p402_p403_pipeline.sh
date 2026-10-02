#!/usr/bin/env bash
# P4.02（固定聚合比较）+ P4.03（可学习聚合）串行作业。前置：81312x 抽取完成。
# 纪律：申请 GPU（--gres=gpu:1）；计算走 SLURM；项目 venv。
#SBATCH -J p402_p403_agg
#SBATCH -p NV_4090D
#SBATCH --gres=gpu:1
#SBATCH -c 16
#SBATCH --mem=96G
#SBATCH -t 08:00:00
#SBATCH -o /lenovofs1/home/jyma/PLM_benchmark/logs/p402_p403_agg_%j.out

set -eo pipefail
cd /lenovofs1/home/jyma/PLM_benchmark/p303_resplit/proj
PY=/lenovofs1/home/jyma/PLM_benchmark/p303_resplit/venv/bin/python

echo "== P4.02 fixed aggregation =="
$PY scripts/run_p402_aggregation.py

echo "== P4.03 learned aggregation =="
$PY scripts/run_p403_learned.py

echo "[p402+p403] DONE $(date -u +%Y-%m-%dT%H:%M:%SZ)"
