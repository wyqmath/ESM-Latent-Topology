#!/usr/bin/env python3
"""P5.08 修复（一审 M1/M2）：排除链重建——UniProt accession 归一化排除 + 序列 sha 去重。

一审发现（agent_006a021f）：
  M1 design 注册的 sha256 去重未执行（kept 131 含 13 组 39 条重复序列）；
  M2 UniProt 排除空转（uniprot 列为 entry-name 格式如 Q6AMB9_DESPS，与 accession
     直接比对恒 0）。
修复规则（在 mmseqs 排除结果之上追加，顺序固定）：
  1) accession 归一化：uniprot 列 split('_')[0]，与 DisProt 3,337 acc 比对 → 排除；
  2) sha256 组内去重：同序列组保留 entry 字典序第一条 → 其余排除；
输出：data/interim/p508_final_set.json（重建，含 dropped 明细分类）。
"""
import csv
import hashlib
import json
import os
from collections import defaultdict

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
I = os.path.join(ROOT, "data", "interim")

# 旧 mmseqs 幸存名单（131）
old = json.load(open(f"{I}/p508_final_set.json"))
kept_old = set(old["kept"])

rows = {r["entry"]: r for r in csv.DictReader(open(f"{I}/p508_candidate_table.tsv"), delimiter="\t")}
labels = {m["name"]: m for m in json.load(open(f"{I}/p508_labels.json"))}
seqs = {}
name = None
for line in open(f"{I}/p508_seqs.fa"):
    if line.startswith(">"):
        name = line[1:].strip()
    else:
        seqs.setdefault(name, []).append(line.strip())
seqs = {k: "".join(v) for k, v in seqs.items()}

disprot_accs = {r["uniprot_acc"] for r in csv.DictReader(open(os.path.join(ROOT, "data/curated/disorder.tsv")), delimiter="\t")}

# entry → 链名映射（p508:{entry}_{asym}）
by_entry = defaultdict(list)
for nm in kept_old:
    e = nm.split(":")[1].rsplit("_", 1)[0]
    by_entry[e].append(nm)

# 排除链（顺序固定：acc 归一化排除 → sha 组内字典序去重）
drop_acc, drop_sha = {}, {}
kept = []
acc_of = {}
for e in sorted(by_entry):
    for nm in sorted(by_entry[e]):
        u = rows[e]["uniprot"]
        acc_of[nm] = u.split("_")[0] if u else ""

for e in sorted(by_entry):
    for nm in sorted(by_entry[e]):
        acc = acc_of.get(nm, "")
        if acc and acc in disprot_accs:
            drop_acc[nm] = acc
            continue
        sha = hashlib.sha256(seqs[nm].encode()).hexdigest()
        if sha in drop_sha:
            continue
        drop_sha[sha] = nm
        kept.append(nm)

json.dump({"kept": sorted(kept),
           "dropped_homology_mmseqs": old.get("dropped_homology", []),
           "dropped_uniprot_acc_normalized": drop_acc,
           "dropped_sequence_duplicate": sorted(set(kept_old) - set(kept) - set(drop_acc))},
          open(f"{I}/p508_final_set.json", "w"), indent=1, sort_keys=True)
uniq = len({hashlib.sha256(seqs[n].encode()).hexdigest() for n in kept})
print(f"重建 final_set: 保留 {len(kept)}（唯一序列 {uniq}）| acc 归一化排除 {len(drop_acc)}: {drop_acc}")
print(f"sha 重复排除 {len(set(kept_old) - set(kept) - set(drop_acc))}")
