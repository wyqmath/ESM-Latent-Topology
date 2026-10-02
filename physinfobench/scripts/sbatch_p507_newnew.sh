#!/usr/bin/env bash
# P5.07 修复：新链-新链结构近邻边（foldseek 自搜 + US-align 确认）。
# pdb_new/（281 CA-PDB）已在前轮隔离作业产出。
#SBATCH -J p507_nn
#SBATCH -p NV_4090D
#SBATCH --gres=gpu:1
#SBATCH -c 16
#SBATCH --mem=64G
#SBATCH -t 06:00:00
#SBATCH -o /lenovofs1/home/jyma/PLM_benchmark/logs/p507_nn_%j.out

set -euo pipefail
OUT=/lenovofs1/home/jyma/PLM_benchmark/p303_resplit/proj/data/interim/p507_isolate
FS=/lenovofs1/home/jyma/PLM_benchmark/tools_linux/foldseek/bin/foldseek
US=/lenovofs1/home/jyma/PLM_benchmark/p303_knot_foldseek/tools/USalign/USalign
PY=/lenovofs1/home/jyma/.conda/envs/plm_bench/bin/python

if [ ! -s "$OUT/nn_hits.tsv" ]; then
  rm -rf $OUT/db_nn $OUT/aln_nn
  mkdir -p $OUT/db_nn
  $FS createdb $OUT/pdb_new $OUT/db_nn/nn
  $FS search $OUT/db_nn/nn $OUT/db_nn/nn $OUT/aln_nn $OUT/db_nn/tmp -e 0.001 -a
  $FS convertalis $OUT/db_nn/nn $OUT/db_nn/nn $OUT/aln_nn $OUT/nn_hits.tsv \
    --format-output query,target,qtmscore,ttmscore,rmsd
fi
"$PY" /lenovofs1/home/jyma/PLM_benchmark/p303_resplit/proj/scripts/p507_confirm_newnew.py \
  --out "$OUT" --usalign "$US"
echo "[p507nn] DONE $(date -u +%Y-%m-%dT%H:%M:%SZ)"
