#!/usr/bin/env python3
"""KNOT-RESPLIT 步骤 1a：打结结构绑定核算（可复现固化版；输入=冻结产物，只读）。

输入版本（sha256 见 reports/knot_resplit/inputs_checksums.json）：
  data/interim/p303/usalign_confirmed.tsv   集群作业 813027 产物（3,854 对 US-align 确认）
  data/interim/p303/knot_chain_list.tsv     1,006 usable 链清单（链/pdb/chain/split/presence/tier）
  data/splits/split_manifest.tsv            现行冻结划分（旧版基线；本任务不改它，步骤 4 才生成新版）
输出：
  results/probes/knot_cross_set_binding.tsv  跨集合绑定对全清单（679 行）
  results/probes/knot_binding_check_qc.json  汇总
口径：US-align 单链 TM≥0.6 min 对称化（与 FS 冻结绑定规则同口径）；链→split 经 (pdb,chain) 归一映射。
"""
import csv
import json
import os
import sys
from collections import Counter, defaultdict

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def die(m):
    print(f"[kbc FATAL] {m}", file=sys.stderr)
    sys.exit(1)


man = {}
for r in csv.DictReader(open(os.path.join(ROOT, "data/splits/split_manifest.tsv")), delimiter="\t"):
    if r["task_area"] == "knot":
        suf = r["sample_id"].split(":", 1)[1]
        man[suf] = r["split"]
        man[suf.lower()] = r["split"]
by_pc, tiers = {}, {}
for r in csv.DictReader(open(os.path.join(ROOT, "data/interim/p303/knot_chain_list.tsv")), delimiter="\t"):
    rid = r["record_id"]
    s = r["split"] if r["split"] != "?" else (man.get(rid) or man.get(rid.lower()))
    by_pc[(r["pdb"].lower(), r["chain"])] = s
    tiers[(r["pdb"].lower(), r["chain"])] = r["presence_target"]

rows_out, n_parsed, n_fail = [], 0, 0
seen_pairs = set()
for line in open(os.path.join(ROOT, "data/interim/p303/usalign_confirmed.tsv")):
    p = line.rstrip("\n").split("\t")
    if len(p) < 3 or not p[2].strip():
        continue
    a, b = p[0], p[1]
    try:
        tms = [float(x) for x in p[2].split()]
    except ValueError:
        n_fail += 1
        continue
    if len(tms) < 2:
        n_fail += 1
        continue
    n_parsed += 1
    key = tuple(sorted([a, b]))
    if key in seen_pairs:
        die(f"重复对 {key}")
    seen_pairs.add(key)
    if a == b:
        die(f"自对 {a}")
    ka, kb = a.rsplit("_", 1), b.rsplit("_", 1)
    sa = by_pc.get((ka[0].lower(), ka[1]))
    sb = by_pc.get((kb[0].lower(), kb[1]))
    if sa is None or sb is None:
        die(f"链无法解析 split: {a} {b}")
    if min(tms) >= 0.6 and sa != sb:
        rows_out.append({"chain_a": a, "chain_b": b, "split_a": sa, "split_b": sb,
                         "tm_norm_a": round(tms[0], 4), "tm_norm_b": round(tms[1], 4),
                         "tm_min": round(min(tms), 4), "tm_max": round(max(tms), 4),
                         "presence_a": tiers.get((ka[0].lower(), ka[1]), "?"),
                         "presence_b": tiers.get((kb[0].lower(), kb[1]), "?")})
rows_out.sort(key=lambda r: -r["tm_min"])
if n_parsed != 3854:
    die(f"解析对 {n_parsed} != 3854")
if len(rows_out) != 679:
    die(f"跨集合绑定对 {len(rows_out)} != 679")
with open(os.path.join(ROOT, "results/probes/knot_cross_set_binding.tsv"), "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows_out[0].keys()), delimiter="\t", lineterminator="\n")
    w.writeheader()
    w.writerows(rows_out)

cc = Counter(tuple(sorted([r["split_a"], r["split_b"]])) for r in rows_out)
uniq = sorted({c for r in rows_out for c in (r["chain_a"], r["chain_b"])})
by_set = Counter(by_pc.get((u.rsplit("_", 1)[0].lower(), u.rsplit("_", 1)[1])) for u in uniq)
tms = [r["tm_min"] for r in rows_out]
pos_involved = sorted({f"{r['chain_a']}({r['presence_a']})" for r in rows_out if r["presence_a"] == "1"} |
                      {f"{r['chain_b']}({r['presence_b']})" for r in rows_out if r["presence_b"] == "1"})
# 单侧相似（信息性）
near, seen_near = [], set()
for line in open(os.path.join(ROOT, "data/interim/p303/usalign_confirmed.tsv")):
    p = line.rstrip("\n").split("\t")
    if len(p) < 3 or not p[2].strip():
        continue
    a, b = p[0], p[1]
    try:
        tms = [float(x) for x in p[2].split()]
    except ValueError:
        continue
    if len(tms) >= 2 and max(tms) >= 0.6 > min(tms):
        ka, kb = a.rsplit("_", 1), b.rsplit("_", 1)
        sa, sb = by_pc.get((ka[0].lower(), ka[1])), by_pc.get((kb[0].lower(), kb[1]))
        key = tuple(sorted([a, b]))
        if sa != sb and key not in seen_near:
            seen_near.add(key)
            near.append((a, b))
summary = {
    "protocol": "US-align 单链 CA min TM>=0.6（与 FS 冻结绑定规则同口径）；候选=Foldseek e<=0.001 预筛",
    "universe_chains": 1006,
    "unverified_chains": 14,
    "usalign_pairs": n_parsed,
    "parse_failures": n_fail,
    "binding_level_pairs": 2998,
    "cross_set_binding_pairs": len(rows_out),
    "cross_set_by_class": {f"{k[0]}|{k[1]}": v for k, v in sorted(cc.items())},
    "unique_chains_involved": len(uniq),
    "unique_chains_by_set": dict(by_set),
    "tm_min_median": sorted(tms)[len(tms) // 2],
    "tm_min_max": max(tms),
    "positive_chains_involved": len(pos_involved),
    "unilateral_max06_cross_set_pairs": len(near),
    "verdict": "STOP_DECISION_POINT — 打结宇宙未通过 min 口径结构近邻绑定检查（用户已裁决方案 A）",
}
with open(os.path.join(ROOT, "results/probes/knot_binding_check_qc.json"), "w") as f:
    json.dump(summary, f, ensure_ascii=False, indent=1, sort_keys=True)
print(f"[kbc] OK cross={len(rows_out)} classes={summary['cross_set_by_class']} pos_involved={len(pos_involved)}")
