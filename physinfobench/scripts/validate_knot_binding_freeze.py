#!/usr/bin/env python3
"""KNOT-RESPLIT 步骤 1b：冻结校验（补算产物完整性+口径重现+覆盖声明）。

检查（全过 exit 0）：
  F1 输入版本记账：inputs_checksums.json 与当前文件 sha256 一致（防步骤间漂移）；
  F2 确认表完整性：3,854 行、全部可解析、无自对、无序对唯一；
  F3 链覆盖声明：端点 ⊆ 1,006 usable 链（(pdb,chain) 归一）；14 条无结构链列为未验证（不是通过）；
  F4 口径重现：min≥0.6 边=2,998；跨集合=679（301/230/148）；映射零未解析；
  F5 集合映射基线：全部 split 来自现行冻结 split_manifest.tsv（旧版基线，本任务未改动——
     以 git status 确认 data/splits/ 无改动 + split_qc.json run_ts 仍为 2026-09-24 03:27）。
输出：reports/knot_resplit/step1_validation_qc.json；失败 exit 1。
"""
import csv
import hashlib
import json
import os
import subprocess
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
FREEZE_DIR = os.path.join(ROOT, "reports/knot_resplit")


def die(m):
    print(f"[freeze FATAL] {m}", file=sys.stderr)
    sys.exit(1)


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


INPUTS = [
    "data/interim/p303/usalign_confirmed.tsv",
    "data/interim/p303/knot_chain_list.tsv",
    "data/curated/knots_sequences.tsv",
    "data/curated/knots_sequences_unavailable.tsv",
    "data/curated/knots.tsv",
    "data/splits/split_manifest.tsv",
    "scripts/fetch_single_chain.py",
    "scripts/run_knot_foldseek.sh",
    "scripts/knot_binding_check.py",
]
os.makedirs(FREEZE_DIR, exist_ok=True)
sums = {p: sha(os.path.join(ROOT, p)) for p in INPUTS}
ck_path = os.path.join(FREEZE_DIR, "inputs_checksums.json")
if os.path.exists(ck_path):
    old = json.load(open(ck_path))["checksums"]
    drifted = [p for p in INPUTS if old.get(p) and old[p] != sums[p]]
    if drifted:
        die(f"F1 输入漂移: {drifted}")
json.dump({"frozen_at": __import__("datetime").datetime.now().strftime("%Y-%m-%d %H:%M"),
           "checksums": sums}, open(ck_path, "w"), indent=1, sort_keys=True)

# F2 确认表
rows, fails = [], 0
seen = set()
for line in open(os.path.join(ROOT, "data/interim/p303/usalign_confirmed.tsv")):
    p = line.rstrip("\n").split("\t")
    if len(p) < 3 or not p[2].strip():
        continue
    a, b = p[0], p[1]
    try:
        tms = [float(x) for x in p[2].split()]
    except ValueError:
        fails += 1
        continue
    if len(tms) < 2:
        fails += 1
        continue
    if a == b:
        die(f"F2 自对 {a}")
    key = tuple(sorted([a, b]))
    if key in seen:
        die(f"F2 重复对 {key}")
    seen.add(key)
    rows.append((a, b, tms))
if len(rows) != 3854 or fails:
    die(f"F2 rows={len(rows)} fails={fails} != 3854/0")

# F3 覆盖
chain_rows = list(csv.DictReader(open(os.path.join(ROOT, "data/interim/p303/knot_chain_list.tsv")), delimiter="\t"))
chain_keys = {(r["pdb"].lower(), r["chain"]) for r in chain_rows}
endpoints = set()
for a, b, _ in rows:
    for x in (a, b):
        f, c = x.rsplit("_", 1)
        endpoints.add((f.lower(), c))
outside = endpoints - chain_keys
if outside:
    die(f"F3 端点超出 1,006 链集合: {sorted(outside)[:5]}")
unavail = list(csv.DictReader(open(os.path.join(ROOT, "data/curated/knots_sequences_unavailable.tsv")), delimiter="\t"))
if len(chain_rows) != 1006 or len(unavail) != 14:
    die(f"F3 1006/14 口径破坏: {len(chain_rows)}/{len(unavail)}")

# F4 口径重现（重跑 knot_binding_check 逻辑核心数）
man = {}
for r in csv.DictReader(open(os.path.join(ROOT, "data/splits/split_manifest.tsv")), delimiter="\t"):
    if r["task_area"] == "knot":
        suf = r["sample_id"].split(":", 1)[1]
        man[suf] = r["split"]
        man[suf.lower()] = r["split"]
by_pc = {}
for r in chain_rows:
    rid = r["record_id"]
    by_pc[(r["pdb"].lower(), r["chain"])] = r["split"] if r["split"] != "?" else (
        man.get(rid) or man.get(rid.lower()))
n_bind, cross, unresolved = 0, 0, 0
cc = __import__("collections").Counter()
for a, b, tms in rows:
    ka, kb = a.rsplit("_", 1), b.rsplit("_", 1)
    sa, sb = by_pc.get((ka[0].lower(), ka[1])), by_pc.get((kb[0].lower(), kb[1]))
    if sa is None or sb is None:
        die(f"F4 未解析 split: {a} {b}")
    if min(tms) >= 0.6:
        n_bind += 1
        if sa != sb:
            cross += 1
            cc[tuple(sorted([sa, sb]))] += 1
if n_bind != 2998 or cross != 679:
    die(f"F4 口径重现失败: bind={n_bind} cross={cross} != 2998/679")
if dict(cc) != {("confirmation", "development"): 301,
                ("development", "final_holdout"): 230,
                ("confirmation", "final_holdout"): 148}:
    die(f"F4 类别分布漂移: {dict(cc)}")

# F5 划分基线未动
qc = json.load(open(os.path.join(ROOT, "data/splits/split_qc.json")))
if qc.get("run_ts") != "2026-09-24 03:27":
    die(f"F5 split_qc run_ts 漂移: {qc.get('run_ts')}")
dirty = subprocess.run(["git", "status", "--porcelain", "data/splits/"],
                       cwd=ROOT, capture_output=True, text=True).stdout.strip()
if dirty:
    die(f"F5 data/splits 有未提交改动: {dirty}")

qc_out = {"checks": {"F1_inputs_pinned": len(INPUTS), "F2_pairs": len(rows),
                     "F3_covered": 1006, "F3_unverified": 14,
                     "F4_bind_edges": n_bind, "F4_cross_set": cross,
                     "F5_split_baseline": "2026-09-24 03:27 (frozen, untouched)"},
          "coverage_statement": "结构检查覆盖 1,006/1,020 usable 链；其余 14 条无结构链标记为未验证（不计为通过）"}
json.dump(qc_out, open(os.path.join(FREEZE_DIR, "step1_validation_qc.json"), "w"),
          ensure_ascii=False, indent=1, sort_keys=True)
print("[freeze] ALL PASSED:", json.dumps(qc_out["checks"]))
