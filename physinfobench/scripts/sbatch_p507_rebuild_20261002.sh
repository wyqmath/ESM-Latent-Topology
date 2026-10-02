#!/usr/bin/env bash
# Rebuild the P5.07 supplement only after dual-score tables and fp32 caches are complete.
#SBATCH -J p507_rebuild_v2
#SBATCH -p NV_4090D
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=32G
#SBATCH -t 02:00:00
#SBATCH -o /lenovofs1/home/jyma/PLM_benchmark/logs/p507_rebuild_v2_%j.out

set -euo pipefail
PROJECT_ROOT=/lenovofs1/home/jyma/PLM_benchmark/p303_resplit/proj
PY=/lenovofs1/home/jyma/PLM_benchmark/p303_resplit/venv/bin/python
export PLM_PROJECT_ROOT="$PROJECT_ROOT"
cd "$PROJECT_ROOT"

"$PY" scripts/p507_build_final_universe.py
"$PY" scripts/p507_type_probe_20261002.py \
  --output results/repairs/20261002/p507_type_probe
echo "[p507-rebuild-v2] DONE $(date -u +%Y-%m-%dT%H:%M:%SZ)"
