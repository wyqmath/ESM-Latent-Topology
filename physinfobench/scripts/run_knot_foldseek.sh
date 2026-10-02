#!/bin/bash
#SBATCH --partition=NV_4090D
#SBATCH --gres=gpu:1
#SBATCH --job-name=p303knotfs
#SBATCH --output=slurm_%j.log
#SBATCH --time=12:00:00
#SBATCH -A cwu
cd /lenovofs1/home/jyma/PLM_benchmark/p303_knot_foldseek
set -e
FS=/lenovofs1/home/jyma/PLM_benchmark/tools_linux/foldseek/bin/foldseek
PY=/lenovofs1/home/jyma/.conda/envs/plm_bench/bin/python
US=./tools/USalign/USalign
echo "=== [1] 下载 mmCIF + 抽单链 $(date '+%F %T') ==="
$PY fetch_single_chain.py
echo "=== [2] Foldseek 全对全 $(date '+%F %T') ==="
if [ ! -s knot_hits.tsv ]; then
  $FS createdb singe_pdb knotdb
  $FS search knotdb knotdb knotaln tmp -e 0.001 -a
  $FS convertalis knotdb knotdb knotaln knot_hits.tsv --format-output query,target,evalue,bits,alntmscore,qtmscore,ttmscore
fi
echo "hits: $(wc -l < knot_hits.tsv)"
echo "=== [3] 候选预筛 $(date '+%F %T') ==="
$PY - << 'PYEOF'
rows = []
with open("knot_hits.tsv") as f:
    for line in f:
        p = line.rstrip("\n").split("\t")
        if len(p) < 7:
            continue
        a, b = p[0], p[1]
        if a >= b:
            continue
        s = max(float(p[4]), float(p[5]), float(p[6]))
        if s >= 0.5:
            rows.append((a, b, s))
rows.sort(key=lambda x: -x[2])
rows = rows[:30000]
with open("candidates.tsv", "w") as f:
    for a, b, _ in rows:
        f.write(f"{a}\t{b}\n")
print(f"candidate pairs: {len(rows)}")
PYEOF
echo "=== [4] US-align 确认 $(date '+%F %T') ==="
: > usalign_confirmed.tsv
while IFS=$'\t' read -r a b; do
  res=$("$US" "singe_pdb/$a.pdb" "singe_pdb/$b.pdb" 2>/dev/null | grep "TM-score=" | awk '{print $2}' | tr '\n' ' ')
  echo -e "$a\t$b\t$res" >> usalign_confirmed.tsv
done < candidates.tsv
echo "confirmed: $(wc -l < usalign_confirmed.tsv)"
echo "=== DONE $(date '+%F %T') ==="
