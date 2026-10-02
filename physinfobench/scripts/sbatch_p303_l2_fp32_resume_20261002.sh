#!/usr/bin/env bash
# Resume the failed stage-B scoring after validating and extracting the exact current split delta.
# The previous job completed the pinned-fp32 base, stage-A and endpoint extraction and selection.
#SBATCH -J p303_l2_resume
#SBATCH -p NV_4090D
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 04:00:00
#SBATCH -o /lenovofs1/home/jyma/PLM_benchmark/logs/p303_l2_resume_%j.out

set -euo pipefail
PROJECT_ROOT=/lenovofs1/home/jyma/PLM_benchmark/p303_resplit/proj
PY=/lenovofs1/home/jyma/PLM_benchmark/p303_resplit/venv/bin/python
export PLM_PROJECT_ROOT="$PROJECT_ROOT"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
cd "$PROJECT_ROOT"

EMB=data/interim/p303/emb_20261002
OUT=results/repairs/20261002/probes
test -s "$OUT/T-FS-L2-PU-RANK.stageA.selected.json"
"$PY" scripts/check_l2_stage_b_input_union.py \
  --manifest data/splits/fs_l2_background_manifest.tsv.gz \
  --base-fasta /lenovofs1/home/jyma/PLM_benchmark/p303_stage_b/l2_stage_b.fa \
  --delta-fasta data/interim/p303/l2_stage_b_delta_20261002.fa \
  --out "$OUT/l2_stage_b_input_union.json"
"$PY" scripts/extract_representations.py \
  --fasta data/interim/p303/l2_stage_b_delta_20261002.fa \
  --out-dir "$EMB/l2_stage_b_delta" --mean-layers 5,11,17,23,29,33 \
  --device cuda --batch-max-tokens 16384 \
  --pack-matrix "$EMB/l2_stage_b_delta/packed.npz"
"$PY" scripts/run_l2_stage_b_20261002.py
echo "[p303-l2-fp32-resume] DONE $(date -u +%Y-%m-%dT%H:%M:%SZ)"
