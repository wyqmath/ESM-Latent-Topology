#!/usr/bin/env bash
# P5.02 确认集复核（集群侧）：lock 校验 → 确认集抽取 → H-KNOT/H-DISORDER 一次性评估。
# 集群提交：sbatch scripts/sbatch_p502.sh
# 纪律：申请 GPU（--gres=gpu:1，QOSMinGRES）；p303_resplit venv；完成后检查无遗留进程。
#SBATCH -J p502_confirm
#SBATCH -p NV_4090D
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=96G
#SBATCH -t 04:00:00
#SBATCH -o /lenovofs1/home/jyma/PLM_benchmark/logs/p502_confirm_%j.out

set -eo pipefail
cd /lenovofs1/home/jyma/PLM_benchmark/p303_resplit/proj
PY=/lenovofs1/home/jyma/PLM_benchmark/p303_resplit/venv/bin/python
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_ENDPOINT=https://hf-mirror.com
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

mkdir -p data/interim/p501 data/interim/p502 results/confirmation logs

$PY scripts/run_p502_confirmation_cluster.py --step all
$PY scripts/run_p502_confirmation_fsl2.py

echo "[p502] DONE $(date -u +%Y-%m-%dT%H:%M:%SZ)"
