#!/bin/bash
# P3.03 提取驱动（剩余部分）：disorder_dev + l2_stage_a
set -e
cd /Users/yuan/Documents/ChatGPT/PLM
export HF_ENDPOINT=https://hf-mirror.com
PY=.venv_local/bin/python
ML=5,11,17,23,29,33
RL=11,23,33
D=data/interim/p303
echo "=== [4/5] disorder_dev $(date '+%H:%M:%S') ==="
$PY scripts/extract_representations.py --fasta $D/disorder_dev.fa --out-dir $D/emb/disorder_dev --mean-layers $ML --resid-layers $RL --resid-index-file $D/disorder_resid_idx.tsv
echo "=== [5/5] l2_stage_a $(date '+%H:%M:%S') ==="
$PY scripts/extract_representations.py --fasta $D/l2_stage_a.fa --out-dir $D/emb/l2_stage_a --mean-layers $ML
echo "ALL_REMAINING_EXTRACTIONS_DONE $(date '+%H:%M:%S')"
