#!/usr/bin/env bash
# P4.02-M1 整改：MIL 三臂按冻结协议重算（每 C 跨折 OOF 池化 AUROC 全局选 C）。
# 集群提交：sbatch scripts/run_p402_mil_recompute.sh（= 作业 813180）。
#SBATCH -J p402_mil
#SBATCH -p NV_4090D
#SBATCH --gres=gpu:1
#SBATCH -c 16
#SBATCH --mem=96G
#SBATCH -t 03:00:00
#SBATCH -o /lenovofs1/home/jyma/PLM_benchmark/logs/p402_mil_%j.out
set -eo pipefail
cd /lenovofs1/home/jyma/PLM_benchmark/p303_resplit/proj
PY=/lenovofs1/home/jyma/PLM_benchmark/p303_resplit/venv/bin/python
$PY scripts/run_p402_aggregation.py arms --task knots --arms MIL_mean,MIL_max,MIL_mean_h29
echo "[mil-recompute] DONE"
