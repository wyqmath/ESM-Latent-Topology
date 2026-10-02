#!/bin/bash
# P3.03 提取驱动：按 probes.yaml 执行序 A/B 顺序提取（后台长跑；断点续跑=重跑即缓存命中）
set -e
cd /Users/yuan/Documents/ChatGPT/PLM
export HF_ENDPOINT=https://hf-mirror.com
PY=.venv_local/bin/python
ML=5,11,17,23,29,33
RL=11,23,33
D=data/interim/p303
E=$D/emb
echo "=== [1/5] fs_region $(date '+%H:%M:%S') ==="
$PY scripts/extract_representations.py --fasta $D/fs_region.fa --out-dir $E/fs_region --mean-layers $ML --resid-layers $RL --resid-index-file $D/fs_region_resid_idx.tsv
echo "=== [2/5] fs_endpoints $(date '+%H:%M:%S') ==="
$PY scripts/extract_representations.py --fasta $D/fs_endpoints.fa --out-dir $E/fs_endpoints --mean-layers $ML
echo "=== [3/5] knots_dev $(date '+%H:%M:%S') ==="
$PY scripts/extract_representations.py --fasta $D/knots_dev.fa --out-dir $E/knots_dev --mean-layers $ML
echo "=== [4/5] disorder_dev $(date '+%H:%M:%S') ==="
$PY scripts/extract_representations.py --fasta $D/disorder_dev.fa --out-dir $E/disorder_dev --mean-layers $ML --resid-layers $RL --resid-index-file $D/disorder_resid_idx.tsv
echo "=== [5/5] l2_stage_a $(date '+%H:%M:%S') ==="
$PY scripts/extract_representations.py --fasta $D/l2_stage_a.fa --out-dir $E/l2_stage_a --mean-layers $ML
echo "ALL_EXTRACTIONS_DONE $(date '+%H:%M:%S')"
