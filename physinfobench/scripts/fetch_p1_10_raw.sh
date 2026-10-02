#!/bin/bash
# P1.10 原始文件获取脚本（最终形态，2026-09-23）
# 用途：可复现地下载冻结背景宇宙所需的 7 个原始文件并输出 SHA256。
# 纪律：data/raw/** 不入 git；本脚本与 SHA256 记录入 data/manifests/fold_switch_background_sources.tsv。
# 网络事实（2026-09-23 07:49–07:51 预探测）：全部源直连可达，无需代理；
#   ftp.uniprot.org 22–160 KB/s 波动、支持 Accept-Ranges → FASTA 用 6 路分块并行；
#   ftp.ebi.ac.uk 70–290 KB/s；files.wwpdb.org ~0.9 MB/s。
# 镜像（FASTA 字节数一致 93801562）：
#   主 https://ftp.uniprot.org/...  备 https://ftp.ebi.ac.uk/pub/databases/uniprot/...  备 https://ftp.expasy.org/databases/uniprot/...
# 教训：release-notes.txt 在 complete/ 子目录下为 404，正确路径是 current_release/relnotes.txt。
set -u
ROOT="/Users/yuan/Documents/ChatGPT/PLM"
SP_DIR="$ROOT/data/raw/swissprot/2026-09-23"
SIFTS_DIR="$ROOT/data/raw/sifts/2026-09-23"
WWPDB_DIR="$ROOT/data/raw/wwpdb/2026-09-23"
TMP=/tmp/p1_10_parts
mkdir -p "$SP_DIR" "$SIFTS_DIR" "$WWPDB_DIR" "$TMP"
TZ=Asia/Shanghai date '+START %Y-%m-%d %H:%M:%S'

fetch () { # url out
  [ -s "$2" ] && { echo "SKIP $(basename "$2")"; return 0; }
  curl -s --retry 5 --retry-delay 3 --max-time 900 -o "$2" "$1" \
    && echo "OK $(basename "$2") $(stat -f%z "$2") bytes" \
    || echo "FAILED $(basename "$2")"
}

fetch "https://ftp.ebi.ac.uk/pub/databases/msd/sifts/flatfiles/csv/pdb_chain_uniprot.csv.gz" "$SIFTS_DIR/pdb_chain_uniprot.csv.gz"
fetch "https://ftp.ebi.ac.uk/pub/databases/msd/sifts/flatfiles/csv/uniprot_pdb.csv.gz" "$SIFTS_DIR/uniprot_pdb.csv.gz"
fetch "https://ftp.ebi.ac.uk/pub/databases/msd/sifts/flatfiles/csv/uniprot_segments_observed.csv.gz" "$SIFTS_DIR/uniprot_segments_observed.csv.gz"
fetch "https://files.wwpdb.org/pub/pdb/derived_data/pdb_entry_type.txt" "$WWPDB_DIR/pdb_entry_type.txt"
fetch "https://files.wwpdb.org/pub/pdb/derived_data/index/entries.idx" "$WWPDB_DIR/entries.idx"
fetch "https://ftp.uniprot.org/pub/databases/uniprot/current_release/relnotes.txt" "$SP_DIR/relnotes.txt"

# --- uniprot_sprot.fasta.gz（89.4 MB；6 路 Range 分块并行）---
FASTA="$SP_DIR/uniprot_sprot.fasta.gz"
URL="https://ftp.uniprot.org/pub/databases/uniprot/current_release/knowledgebase/complete/uniprot_sprot.fasta.gz"
SIZE=93801562
N=6
if [ ! -s "$FASTA" ]; then
  CHUNK=$(( (SIZE + N - 1) / N ))
  pids=()
  for i in 0 1 2 3 4 5; do
    s=$((i*CHUNK)); e=$(( (i+1)*CHUNK - 1 )); [ $e -ge $SIZE ] && e=$((SIZE-1))
    ( curl -s --retry 8 --retry-delay 5 --max-time 5400 -r "${s}-${e}" -o "$TMP/part_$i" "$URL" ) &
    pids+=($!)
  done
  fail=0
  for p in "${pids[@]}"; do wait "$p" || fail=1; done
  if [ $fail -ne 0 ]; then echo "FASTA chunk FAILED"; exit 1; fi
  cat "$TMP"/part_0 "$TMP"/part_1 "$TMP"/part_2 "$TMP"/part_3 "$TMP"/part_4 "$TMP"/part_5 > "$FASTA"
  got=$(stat -f%z "$FASTA")
  if [ "$got" != "$SIZE" ]; then echo "FASTA SIZE MISMATCH got=$got want=$SIZE"; exit 1; fi
  rm -f "$TMP"/part_*
  echo "FASTA downloaded $got bytes"
fi

# --- 校验与摘要 ---
echo "--- gzip integrity ---"
for f in "$FASTA" "$SIFTS_DIR"/*.gz; do gzip -t "$f" && echo "gzip OK: $(basename "$f")"; done
echo "--- sha256 ---"
shasum -a 256 "$FASTA" "$SP_DIR/relnotes.txt" \
  "$SIFTS_DIR/pdb_chain_uniprot.csv.gz" "$SIFTS_DIR/uniprot_pdb.csv.gz" "$SIFTS_DIR/uniprot_segments_observed.csv.gz" \
  "$WWPDB_DIR/pdb_entry_type.txt" "$WWPDB_DIR/entries.idx"
TZ=Asia/Shanghai date '+END %Y-%m-%d %H:%M:%S'
