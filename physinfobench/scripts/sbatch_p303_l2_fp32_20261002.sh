#!/usr/bin/env bash
# Recompute the registered FS-L2 stage-A and stage-B diagnostics with pinned fp32 ESM2.
#SBATCH -J p303_l2_fp32
#SBATCH -p NV_4090D
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 12:00:00
#SBATCH -o /lenovofs1/home/jyma/PLM_benchmark/logs/p303_l2_fp32_%j.out

set -euo pipefail
PROJECT_ROOT=/lenovofs1/home/jyma/PLM_benchmark/p303_resplit/proj
PY=/lenovofs1/home/jyma/PLM_benchmark/p303_resplit/venv/bin/python
export PLM_PROJECT_ROOT="$PROJECT_ROOT"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
cd "$PROJECT_ROOT"

EMB=data/interim/p303/emb_20261002
OUT=results/repairs/20261002/probes
mkdir -p "$OUT"

"$PY" scripts/build_fs_endpoint_fasta.py \
  --summary /lenovofs1/home/jyma/PLM_benchmark/manifests/fold_pair_endpoint_sequence_structure_summary.tsv \
  --labels data/curated/fold_switch_global.tsv \
  --out data/interim/p303/fs_endpoints_20261002.fa
"$PY" scripts/check_l2_stage_b_input_union.py \
  --manifest data/splits/fs_l2_background_manifest.tsv.gz \
  --base-fasta /lenovofs1/home/jyma/PLM_benchmark/p303_stage_b/l2_stage_b.fa \
  --delta-fasta data/interim/p303/l2_stage_b_delta_20261002.fa \
  --out results/repairs/20261002/probes/l2_stage_b_input_union.json
"$PY" scripts/extract_representations.py \
  --fasta /lenovofs1/home/jyma/PLM_benchmark/p303_resplit/l2_stage_a.fa \
  --out-dir "$EMB/l2_stage_a" --mean-layers 5,11,17,23,29,33 \
  --device cuda --batch-max-tokens 16384
"$PY" scripts/extract_representations.py \
  --fasta /lenovofs1/home/jyma/PLM_benchmark/p303_stage_b/l2_stage_b.fa \
  --out-dir "$EMB/l2_stage_b" --mean-layers 5,11,17,23,29,33 \
  --device cuda --batch-max-tokens 16384 \
  --pack-matrix "$EMB/l2_stage_b/packed.npz"
"$PY" scripts/extract_representations.py \
  --fasta data/interim/p303/l2_stage_b_delta_20261002.fa \
  --out-dir "$EMB/l2_stage_b_delta" --mean-layers 5,11,17,23,29,33 \
  --device cuda --batch-max-tokens 16384 \
  --pack-matrix "$EMB/l2_stage_b_delta/packed.npz"
"$PY" scripts/extract_representations.py \
  --fasta data/interim/p303/fs_endpoints_20261002.fa \
  --out-dir "$EMB/fs_endpoints" --mean-layers 5,11,17,23,29,33 \
  --device cuda --batch-max-tokens 16384
P303_ONLY=FS_L2A "$PY" scripts/run_probes_p303_20261002.py
"$PY" scripts/run_l2_stage_b_20261002.py
echo "[p303-l2-fp32] DONE $(date -u +%Y-%m-%dT%H:%M:%SZ)"
