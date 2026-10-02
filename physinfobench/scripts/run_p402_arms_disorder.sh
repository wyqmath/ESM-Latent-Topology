#!/usr/bin/env bash
#SBATCH -J p402_dis
#SBATCH -p NV_4090D
#SBATCH --gres=gpu:1
#SBATCH -c 16
#SBATCH --mem=96G
#SBATCH -t 08:00:00
#SBATCH -o /lenovofs1/home/jyma/PLM_benchmark/logs/p402_dis_%j.out
set -eo pipefail
cd /lenovofs1/home/jyma/PLM_benchmark/p303_resplit/proj
PY=/lenovofs1/home/jyma/PLM_benchmark/p303_resplit/venv/bin/python
$PY scripts/run_p402_aggregation.py arms --task disorder \
  --arms ID,WIN4,WIN16,WIN64,MEAN,LAST,ID_h23,MEAN_h23
echo "[p402-dis] DONE $(date -u +%Y-%m-%dT%H:%M:%SZ)"
