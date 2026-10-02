#!/usr/bin/env python3
"""P5.04 前置（final_lock 钉版协议）：保留集 fasta/残基域索引构建。

与 build_p501_confirmation_inputs.py 同协议（probes.yaml residue_storage_domain +
trunc1022 裁剪；build_p303_inputs.py 同源序列源）。纪律：本脚本只做输入构建；
运行时刻在 final_lock 注册之后、保留集标签结果级 first_read（P5.05）之前；
final_holdout 结果级读取另需用户明确授权门禁（P5.05 前置）。
输出：data/interim/p504/{knots_hold.fa, disorder_hold.fa, disorder_hold_resid_idx.tsv,
fs_hold_pairs.tsv, inputs_qc.json}。
注：fs holdout 无 strict 阳性（final_lock n_strict_locked=0）且不注册描述性
side-table，fs_hold_pairs.tsv 仅记录 pair 清单与端点（供覆盖率披露，不参与判定）。
"""
import csv
import hashlib
import json
import os
import sys
from collections import defaultdict
from datetime import datetime

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
OUT = os.path.join(ROOT, "data/interim/p504")
B = "/Users/yuan/Documents/Codex/2026-09-08/jie/benchmark_step1"


def die(m):
    print(f"[p504in FATAL] {m}", file=sys.stderr)
    sys.exit(1)


def rd(p, d="\t"):
    with open(p, newline="") as f:
        return list(csv.DictReader(f, delimiter=d))


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_fa(path, items):
    with open(path, "w") as f:
        for n, s in items:
            f.write(f">{n}\n{s}\n")


if not os.path.isdir(B):
    die(f"FS 端点序列源目录不存在（仓库外依赖）：{B}")
os.makedirs(OUT, exist_ok=True)
man = rd(os.path.join(ROOT, "data/splits/split_manifest.tsv"))
qc = {"built_at": datetime.now().strftime("%Y-%m-%d %H:%M"), "outputs": {}}

# ---- 1) knots 保留集 ----
kseq = {r["record_id"].lower(): r for r in rd(os.path.join(ROOT, "data/curated/knots_sequences.tsv"))}
kdb = json.load(open(os.path.join(ROOT, "data/raw/rcsb/2026-09-25/knot_entry_sequences.json")))
knots = {r["record_id"].lower(): r for r in rd(os.path.join(ROOT, "data/curated/knots.tsv"))}
conf_k = [r["sample_id"].split(":", 1)[1] for r in man
          if r["task_area"] == "knot" and r["split"] == "final_holdout"]
items, n_pos, n_neg, missing = [], 0, 0, []
for rec in conf_k:
    krec = knots.get(rec.lower())
    if krec is None:
        die(f"保留链不在 knots.tsv: {rec}")
    if krec["presence_mask"] != "1":
        continue
    s = kseq.get(rec.lower())
    if s is None:
        missing.append(rec)
        continue
    if krec["presence_target"] == "1":
        n_pos += 1
    elif krec["presence_target"] == "0":
        n_neg += 1
    else:
        die(f"{rec} presence_target 非法")
    e, ch = s["pdb"], s["chain"]
    if e not in kdb or ch not in kdb[e]:
        die(f"{rec} 序列库缺 {e}/{ch}")
    items.append((f"knot:{rec}", kdb[e][ch]))
if sorted(missing) != ["3pvm_B"]:
    die(f"无序列保留链 {sorted(missing)} != 预期 [3pvm_B]（final_lock 覆盖披露口径）")
if (n_pos, n_neg) != (25, 83):
    die(f"knots 保留可评价 {n_pos}/{n_neg} != final_lock 钉版 25/83")
qc["knots_hold"] = {"n_total_manifest": len(conf_k), "n_evaluable": len(items),
                    "n_pos": n_pos, "n_neg": n_neg, "no_sequence": sorted(missing)}
write_fa(os.path.join(OUT, "knots_hold.fa"), items)

# ---- 2) disorder 保留集 ----
dp = json.load(open(os.path.join(ROOT, "data/raw/disprot/2026-09-22/disprot_api_search.json")))
dpseq = {x["acc"]: x["sequence"] for x in dp["data"]}
id2acc = {r["disprot_id"]: r["uniprot_acc"] for r in rd(os.path.join(ROOT, "data/curated/disorder.tsv"))}
dom = defaultdict(list)
for r in rd(os.path.join(ROOT, "data/curated/disorder_masks.tsv")):
    if r["state"] in ("0", "1") and r["mask"] == "1":
        dom[r["disprot_id"]].extend(range(int(r["start"]), int(r["end"]) + 1))
hold_d = sorted(r["sample_id"].split(":", 1)[1] for r in man
                if r["task_area"] == "disorder" and r["split"] == "final_holdout")
dis_items, dis_ridx = [], []
n_empty = n_beyond = n_clip = 0
for did in hold_d:
    acc = id2acc.get(did)
    seq = dpseq.get(acc)
    if not seq:
        die(f"disorder {did}/{acc} 无序列")
    idx = sorted(set(dom.get(did, [])))
    if not idx:
        n_empty += 1
        continue
    clipped = [i for i in idx if i <= 1022]
    if len(clipped) < len(idx):
        n_clip += len(idx) - len(clipped)
        idx = clipped
    if not idx:
        n_beyond += 1
        continue
    if max(idx) > len(seq):
        die(f"disorder {did}: 索引域异常")
    dis_items.append((did, seq))
    dis_ridx.append({"name": did, "resid_indices": ",".join(map(str, idx))})
if len(dis_items) + n_empty + n_beyond != 513:
    die(f"disorder 保留 513 口径破坏: {len(dis_items)}+{n_empty}+{n_beyond}")
qc["disorder_hold"] = {"n_total_manifest": 513, "n_extracted": len(dis_items),
                       "n_empty_domain": n_empty, "n_beyond_trunc": n_beyond,
                       "n_clip_residues": n_clip,
                       "n_domain_residues": sum(len(x["resid_indices"].split(",")) for x in dis_ridx)}
write_fa(os.path.join(OUT, "disorder_hold.fa"), dis_items)
with open(os.path.join(OUT, "disorder_hold_resid_idx.tsv"), "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["name", "resid_indices"], delimiter="\t", lineterminator="\n")
    w.writeheader()
    w.writerows(dis_ridx)

# ---- 3) fs 保留对清单（覆盖率披露用；无 strict 阳性、无判定、无 side-table）----
fs = {r["pair_id"]: r for r in rd(os.path.join(ROOT, "data/curated/fold_switch_global.tsv"))}
hold_f = [r["sample_id"] for r in man
          if r["task_area"] == "fold_switch" and r["split"] == "final_holdout"]
n_strict = sum(1 for p in hold_f if fs[p]["target"] == "1" and fs[p]["valid_mask"] == "1")
if n_strict != 0 or len(hold_f) != 7:
    die(f"fs 保留 n_strict={n_strict}(应0) / 对数={len(hold_f)}(应7) 口径漂移")
with open(os.path.join(OUT, "fs_hold_pairs.tsv"), "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["pair_id", "tier", "valid_mask", "target"],
                       delimiter="\t", lineterminator="\n")
    w.writeheader()
    for p in hold_f:
        w.writerow({"pair_id": p, "tier": fs[p]["tier"], "valid_mask": fs[p]["valid_mask"],
                    "target": fs[p]["target"]})
qc["fs_hold"] = {"n_pairs": 7, "n_strict": 0,
                 "note": "无判定、无 side-table（final_lock 注册语义）"}

for fn in ["knots_hold.fa", "disorder_hold.fa", "disorder_hold_resid_idx.tsv", "fs_hold_pairs.tsv"]:
    qc["outputs"][fn] = sha256(os.path.join(OUT, fn))
qc["source_pins"] = {p: sha256(os.path.join(ROOT, p)) for p in [
    "data/splits/split_manifest.tsv", "data/curated/knots.tsv",
    "data/curated/knots_sequences.tsv", "data/curated/disorder_masks.tsv",
    "data/curated/disorder.tsv", "data/curated/fold_switch_global.tsv"]}
with open(os.path.join(OUT, "inputs_qc.json"), "w") as f:
    json.dump(qc, f, ensure_ascii=False, indent=1, sort_keys=True)
print("[p504in] OK", json.dumps({k: v for k, v in qc.items()
                                 if k in ("knots_hold", "disorder_hold", "fs_hold")},
                                ensure_ascii=False))
