#!/usr/bin/env bash
#SBATCH -J p402_cmp
#SBATCH -p NV_4090D
#SBATCH --gres=gpu:1
#SBATCH -c 16
#SBATCH --mem=96G
#SBATCH -t 04:00:00
#SBATCH -o /lenovofs1/home/jyma/PLM_benchmark/logs/p402_cmp_%j.out
set -eo pipefail
cd /lenovofs1/home/jyma/PLM_benchmark/p303_resplit/proj
PY=/lenovofs1/home/jyma/PLM_benchmark/p303_resplit/venv/bin/python
echo "== P4.02 compare =="
$PY scripts/run_p402_aggregation.py compare
echo "== P4.03 learned =="
$PY scripts/run_p403_learned.py
echo "[compare+learned] DONE $(date -u +%Y-%m-%dT%H:%M:%SZ)"
