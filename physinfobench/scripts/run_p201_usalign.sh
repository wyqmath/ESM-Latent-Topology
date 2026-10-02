#!/bin/bash
# P2.01 步骤一：US-align 全对全（208 链：192 strict 端点 + 16 PN 代表链）
# 输出：data/interim/p201_step1/usalign/<a>__<b>.out（每链对一份原始输出）
set -e
CH=/Users/yuan/Documents/ChatGPT/PLM/data/interim/p201_step1/chains
OUT=/Users/yuan/Documents/ChatGPT/PLM/data/interim/p201_step1/usalign
USALIGN=/Users/yuan/Documents/Codex/2026-09-08/jie/benchmark_step1/tools/USalign-bin
mkdir -p "$OUT"
PAIRS=/Users/yuan/Documents/ChatGPT/PLM/data/interim/p201_step1/pairs.txt
ls "$CH" | sed 's/\.pdb$//' | sort > /Users/yuan/Documents/ChatGPT/PLM/data/interim/p201_step1/keys.txt
python3 - <<'EOF' > "$PAIRS"
keys=[l.strip() for l in open('/Users/yuan/Documents/ChatGPT/PLM/data/interim/p201_step1/keys.txt') if l.strip()]
n=len(keys)
lines=[]
for i in range(n):
    for j in range(i+1,n):
        lines.append(f"{keys[i]} {keys[j]}")
print("\n".join(lines))
EOF
TOTAL=$(wc -l < "$PAIRS" | tr -d ' ')
echo "pairs=$TOTAL start=$(date '+%H:%M:%S')"
cat > /tmp/usalign_one.sh <<'RUNNER'
#!/bin/bash
CH=/Users/yuan/Documents/ChatGPT/PLM/data/interim/p201_step1/chains
OUT=/Users/yuan/Documents/ChatGPT/PLM/data/interim/p201_step1/usalign
USALIGN=/Users/yuan/Documents/Codex/2026-09-08/jie/benchmark_step1/tools/USalign-bin
a=$1; b=$2
f="$OUT/${a}__${b}.out"
[ -s "$f" ] && exit 0
"$USALIGN" "$CH/${a}.pdb" "$CH/${b}.pdb" > "$f" 2>/dev/null || rm -f "$f"
RUNNER
chmod +x /tmp/usalign_one.sh
xargs -P 8 -L1 /tmp/usalign_one.sh < "$PAIRS"
DONE=$(ls "$OUT" | wc -l | tr -d ' ')
echo "done_files=$DONE end=$(date '+%H:%M:%S')"
