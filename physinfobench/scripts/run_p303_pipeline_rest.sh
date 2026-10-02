#!/bin/bash
# P3.03 剩余全自动管线：disorder 探针收尾 → FS_L2A 选择 → stage-B 构建/提取/打分
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"
export HF_ENDPOINT=https://hf-mirror.com
PYTHON=python3
D=data/interim/p303
echo "[pipe] $(date '+%H:%M:%S') 等待 disorder 探针完成..."
while [ ! -f results/probes/T-DISORDER-RES.selected.json ]; do sleep 120; done
echo "[pipe] $(date '+%H:%M:%S') 恢复 knots/FS summary 行（合并写）"
P303_ONLY=KNOT_PRESENCE,KNOT_TYPE,FS_REGION,FS_L1 "$PYTHON" scripts/run_probes_p303.py 2>&1 | grep -E 'probe\]'
echo "[pipe] $(date '+%H:%M:%S') 等待 stage-A 提取完成..."
while ! grep -q 'ALL_REMAINING_EXTRACTIONS_DONE' $D/extraction.log; do sleep 300; done
echo "[pipe] $(date '+%H:%M:%S') 运行 FS_L2A 选择"
P303_ONLY=FS_L2A "$PYTHON" scripts/run_probes_p303.py 2>&1 | grep -E 'l2A|probe\]'
echo "[pipe] $(date '+%H:%M:%S') 构建 stage-B 代表帧输入"
"$PYTHON" scripts/build_l2_stage_b_inputs.py
echo "[pipe] $(date '+%H:%M:%S') stage-B 提取开始（长跑）"
"$PYTHON" scripts/extract_representations.py --fasta "$D/l2_stage_b.fa" --out-dir "$D/emb/l2_stage_b" --mean-layers 5,11,17,23,29,33
echo "[pipe] $(date '+%H:%M:%S') stage-B 打分+传播+官方指标"
"$PYTHON" scripts/run_l2_stage_b.py
echo "PIPELINE_ALL_DONE $(date '+%H:%M:%S')"
