#!/usr/bin/env python3
"""P3.03 stage-B 输入：dev 背景簇代表帧 fasta（47,604）。

实现语义修订（P3.03，相对 probes.yaml stage_b 原文"protein 级全宇宙 343,682"）：
  CPU 实测吞吐（disorder 提取 ≈0.2 蛋白/s）下 protein 级全宇宙 ≈2 周；修订为
  簇代表帧打分（47,604）+ 分数传播至成员（同簇成员同分），官方 L2 指标在传播后的
  全宇宙上计算。差异与理由进 P3.03 报告并经独立审核；传播敏感性=同簇分数同质性检查。
输出：data/interim/p303/l2_stage_b.fa + stage_b_manifest.tsv + inputs_stage_b_qc.json
"""
import csv
import gzip
import hashlib
import json
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
OUT = os.path.join(ROOT, "data/interim/p303")


def die(m):
    print(f"[stageb FATAL] {m}", file=sys.stderr)
    sys.exit(1)


B = "/Users/yuan/Documents/Codex/2026-09-08/jie/benchmark_step1"
reps = {}
members = {}
with gzip.open(os.path.join(ROOT, "data/splits/fs_l2_background_manifest.tsv.gz"), "rt") as f:
    for r in csv.DictReader(f, delimiter="\t"):
        if r["split"] != "development" or r["uniprot_accession"].startswith("EP_"):
            continue
        members.setdefault(r["cluster_rep"], []).append(r["uniprot_accession"])
        reps[r["cluster_rep"]] = r["uniprot_accession"]
rep_accs = sorted(reps)
if len(rep_accs) != 47604:
    die(f"dev 簇代表 {len(rep_accs)} != 47,604")
need = set(rep_accs)
# 簇代表本身可能是 EP_ 端点——从端点表取序列
summ = {(r["pdb_id"].lower(), r["requested_chain"].upper()): r["observed_sequence"]
        for r in csv.DictReader(open(B + "/manifests/fold_pair_endpoint_sequence_structure_summary.tsv"),
                                 delimiter="\t")}
fs = list(csv.DictReader(open(os.path.join(ROOT, "data/curated/fold_switch_global.tsv")), delimiter="\t"))
ep_map = {}
for r in fs:
    for side, pdb, ch in (("A", r["pdb_a"], r["chain_a"]), ("B", r["pdb_b"], r["chain_b"])):
        ep_map[f"EP_{r['pair_id']}_{side}__{pdb.lower()}{ch.upper()}"] = \
            summ[(pdb.lower(), ch.upper())]

sp_seq = {}
with gzip.open(os.path.join(ROOT, "data/raw/swissprot/2026-09-23/uniprot_sprot.fasta.gz"), "rt") as f:
    acc = None
    buf = []
    for line in f:
        if line.startswith(">"):
            if acc in need:
                sp_seq[acc] = "".join(buf)
            parts = line[1:].split("|")
            acc = parts[1] if len(parts) >= 2 else line[1:].split()[0]
            buf = []
        else:
            buf.append(line.strip())
    if acc in need:
        sp_seq[acc] = "".join(buf)

items, n_ep, n_missing = [], 0, []
for a in rep_accs:
    if a in sp_seq:
        seq = sp_seq[a]
    elif a in ep_map:
        seq = ep_map[a]
        n_ep += 1
    else:
        n_missing.append(a)
        continue
    items.append((a, seq))
if n_missing:
    die(f"缺序列的代表 {len(n_missing)}: {n_missing[:5]}")
with open(os.path.join(OUT, "l2_stage_b.fa"), "w") as f:
    for n, s in items:
        f.write(f">{n}\n{s}\n")
with open(os.path.join(OUT, "stage_b_manifest.tsv"), "w", newline="") as f:
    w = csv.writer(f, delimiter="\t", lineterminator="\n")
    w.writerow(["cluster_rep", "rep_accession", "n_members"])
    for a in rep_accs:
        if a in sp_seq or a in ep_map:
            w.writerow([a, a, len(members[a])])
qc = {"run_ts": __import__("datetime").datetime.now().strftime("%Y-%m-%d %H:%M"),
      "reps": len(items), "ep_reps": n_ep,
      "total_dev_proteins": sum(len(v) for v in members.values()),
      "largest_cluster": max(len(v) for v in members.values()),
      "seq_len_max": max(len(s) for _, s in items)}
json.dump(qc, open(os.path.join(OUT, "inputs_stage_b_qc.json"), "w"), indent=1, sort_keys=True)
print(f"[stageb] OK reps={len(items)} ep_reps={n_ep} members={qc['total_dev_proteins']}")
