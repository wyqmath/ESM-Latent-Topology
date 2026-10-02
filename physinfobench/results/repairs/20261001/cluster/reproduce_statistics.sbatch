#!/bin/bash
#SBATCH --job-name=P5-statistics-repair
#SBATCH --partition=NV_4090D
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --output=statistics_reproduction-%j.out
set -euo pipefail
cd /lenovofs1/home/jyma/PLM_benchmark/repairs/20261001/statistics_reproduction
tar -xzf inputs.tar.gz
PY=/lenovofs1/home/jyma/PLM_benchmark/p303_resplit/venv/bin/python
"$PY" - <<'PY'
import hashlib,json
from pathlib import Path
for p,h in json.loads(Path('input_sha256.json').read_text()).items():
    assert hashlib.sha256(Path(p).read_bytes()).hexdigest()==h,p
print('All 13 transferred inputs verified')
PY
"$PY" -m unittest discover -s tests -v
"$PY" scripts/recompute_p5_statistics.py --component-map results/repairs/20261001/exposure/component_map.tsv --output remote_full_set &
FULL_PID=$!
"$PY" scripts/recompute_p5_statistics.py --component-map results/repairs/20261001/exposure/component_map.tsv --posthoc-exposure results/repairs/20261001/exposure/evaluated_sample_exposure.tsv --output remote_posthoc &
POST_PID=$!
wait "$FULL_PID"
wait "$POST_PID"
printf '%s\n' 'Both statistical reproductions completed'
