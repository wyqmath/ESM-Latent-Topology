#!/usr/bin/env bash
# P4.02 前置：knots dev 750 条全链逐残基表示抽取（ESM-2 650M，hidden 23/29/33，fp16）。
# 集群提交：sbatch scripts/run_p402_extract_knots.sh
# 纪律：申请 GPU（--gres=gpu:1，QOSMinGRES）；plm_bench conda 环境；完成后检查无遗留进程。
#SBATCH -J p402_knots_extract
#SBATCH -p NV_4090D
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 02:00:00
#SBATCH -o /lenovofs1/home/jyma/PLM_benchmark/logs/p402_knots_extract_%j.out

set -eo pipefail
cd /lenovofs1/home/jyma/PLM_benchmark/p303_resplit/proj
PY=/lenovofs1/home/jyma/PLM_benchmark/p303_resplit/venv/bin/python
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_ENDPOINT=https://hf-mirror.com
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

mkdir -p data/interim/p402 logs

$PY scripts/extract_representations.py \
  --fasta data/interim/p303/knots_dev.fa \
  --out-dir data/interim/p402/emb_knots_resid \
  --resid-layers 23,29,33 \
  --device cuda \
  --batch-max-tokens 49152

echo "[p402-extract] DONE $(date -u +%Y-%m-%dT%H:%M:%SZ)"
