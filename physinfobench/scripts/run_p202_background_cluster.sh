#!/bin/bash
# P2.02：背景 A（484,179）+ 192 个 FS 端点 的冻结参数聚类（mmseqs2 easy-cluster 0.30/0.70 cov-mode 0）
# 输出：data/interim/p202/l2_cluster.tsv（rep<TAB>member）；失败自动降级 linclust 并标注
set -e
ROOT=/Users/yuan/Documents/ChatGPT/PLM
W=$ROOT/data/interim/p202
MM=/Users/yuan/Documents/Codex/2026-09-08/jie/benchmark_step1/tools/mmseqs2-18-8cc5c/bin/mmseqs
mkdir -p "$W"
python3 - <<'PYEOF'
import csv, gzip
W="/Users/yuan/Documents/ChatGPT/PLM/data/interim/p202"
want=set()
with gzip.open("/Users/yuan/Documents/ChatGPT/PLM/data/curated/fold_switch_unlabeled_sequence.tsv.gz","rt") as f:
    for r in csv.DictReader(f, delimiter="\t"):
        want.add(r["uniprot_accession"])
n=0
with gzip.open("/Users/yuan/Documents/ChatGPT/PLM/data/raw/swissprot/2026-09-23/uniprot_sprot.fasta.gz","rt") as f, open(W+"/l2_input.fa","w") as out:
    keep=False
    for line in f:
        if line.startswith(">"):
            acc=line[3:].split("|")[1] if "|" in line else line[1:].split()[0]
            keep=acc in want
            if keep:
                out.write(f">{acc}\n"); n+=1
        elif keep:
            out.write(line)
print("bg seqs written:", n, "of want", len(want))
assert n==len(want), "fasta 过滤数与 curated 背景行数不一致"
# P2.02 追加：matched controls 中不在背景 A 的 accession（UniProt REST 取得，manifest 在册）
import os
mc_dir="/Users/yuan/Documents/ChatGPT/PLM/data/raw/p202_matched_controls/2026-09-24"
added=0
if os.path.isdir(mc_dir):
    for fn in sorted(os.listdir(mc_dir)):
        if fn.endswith(".fasta"):
            acc=fn[:-6]
            with open(f"{mc_dir}/{fn}") as fin, open("/Users/yuan/Documents/ChatGPT/PLM/data/interim/p202/l2_input.fa","a") as out:
                first=True
                for line in fin:
                    if line.startswith(">"):
                        if first: out.write(f">{acc}\n"); first=False
                    else: out.write(line)
                added+=1
print("matched controls appended:", added)
PYEOF
echo "fasta ready $(date '+%H:%M:%S')"
# 端点序列另用 join 表补入（避免上面占位循环）：
python3 - <<'PYEOF'
import csv
W="/Users/yuan/Documents/ChatGPT/PLM/data/interim/p202"
summ={(r["pdb_id"].lower(),r["requested_chain"].upper()):r["observed_sequence"] for r in csv.DictReader(open("/Users/yuan/Documents/Codex/2026-09-08/jie/benchmark_step1/manifests/fold_pair_endpoint_sequence_structure_summary.tsv"),delimiter="\t")}
g=list(csv.DictReader(open("/Users/yuan/Documents/ChatGPT/PLM/data/curated/fold_switch_global.tsv"),delimiter="\t"))
n=0
with open(W+"/l2_input.fa","a") as out:
    for r in g:
        for side,pdb,ch in (("A",r["pdb_a"],r["chain_a"]),("B",r["pdb_b"],r["chain_b"])):
            key=f"EP_{r['pair_id']}_{side}__{pdb.lower()}{ch.upper()}"
            out.write(f">{key}\n{summ[(pdb.lower(),ch.upper())]}\n"); n+=1
print("endpoint seqs appended:", n)
PYEOF
grep -c '^>' "$W/l2_input.fa"
set +e
"$MM" easy-cluster "$W/l2_input.fa" "$W/l2_easy" "$W/tmp_easy" --min-seq-id 0.3 -c 0.7 --cov-mode 0 > "$W/easy.log" 2>&1
RC=$?
set -e
if [ $RC -eq 0 ] && [ -s "$W/l2_easy_cluster.tsv" ]; then
  cp "$W/l2_easy_cluster.tsv" "$W/l2_cluster.tsv"
  echo "MODE=easy-cluster"
else
  echo "easy-cluster failed rc=$RC; fallback linclust"
  tail -3 "$W/easy.log" || true
  "$MM" easy-linclust "$W/l2_input.fa" "$W/l2_lin" "$W/tmp_lin" --min-seq-id 0.3 -c 0.7 --cov-mode 0 > "$W/lin.log" 2>&1
  cp "$W/l2_lin_cluster.tsv" "$W/l2_cluster.tsv"
  echo "MODE=easy-linclust"
fi
echo "clusters done $(date '+%H:%M:%S') lines=$(wc -l < "$W/l2_cluster.tsv")"
