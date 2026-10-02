#!/usr/bin/env python3
"""P3.03：构建提取输入（fasta + 残基索引 + L2 stage-A 抽样清单），全部可复现。

输入（冻结产物）：split_manifest.tsv、knots_sequences.tsv（preflight）、fold_switch_global.tsv、
fold_pair_endpoint_sequence_structure_summary.tsv、fold_switch_regions.tsv、disorder_masks.tsv、
disprot_api_search.json、fs_l2_background_manifest.tsv.gz、uniprot_sprot.fasta.gz
输出（data/interim/p303/）：knots_dev.fa、fs_endpoints.fa、fs_region.fa(+resid_idx)、
disorder_dev.fa(+resid_idx)、l2_stage_a.fa、stage_a_manifest.tsv、inputs_qc.json
断言：FS usable=3 对 6 行/fine_only=6 对 10 行（P1.15 口径）；disorder dev=2,380；
stage-A 抽样=RandomState(2026) accession 升序无放回 10,000/47,604；序列-长度/索引一致性。
"""
import csv
import gzip
import hashlib
import json
import os
import sys
from collections import defaultdict

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
B = "/Users/yuan/Documents/Codex/2026-09-08/jie/benchmark_step1"
OUT = os.path.join(ROOT, "data/interim/p303")
os.makedirs(OUT, exist_ok=True)


def die(m):
    print(f"[p303in FATAL] {m}", file=sys.stderr)
    sys.exit(1)


def rd(p, d="\t"):
    with open(p, newline="") as f:
        return list(csv.DictReader(f, delimiter=d))


def write_fa(path, items):  # items: [(name, seq)]
    with open(path, "w") as f:
        for n, s in items:
            f.write(f">{n}\n{s}\n")


man = rd(os.path.join(ROOT, "data/splits/split_manifest.tsv"))
qc = {}

# ---- 1) knots dev ∩ usable ----
kseq = {r["record_id"]: r for r in rd(os.path.join(ROOT, "data/curated/knots_sequences.tsv"))}
kdb = json.load(open(os.path.join(ROOT, "data/raw/rcsb/2026-09-25/knot_entry_sequences.json")))
knots_items = []
for r in man:
    if r["task_area"] == "knot" and r["split"] == "development":
        rec = r["sample_id"].split(":", 1)[1]
        if rec in kseq:  # usable（1,006 已解析序列）
            e, ch = kseq[rec]["pdb"], kseq[rec]["chain"]
            knots_items.append((r["sample_id"], kdb[e][ch]))
qc["knots_dev_usable"] = len(knots_items)
if not knots_items:
    die("knots dev usable 为空")
write_fa(os.path.join(OUT, "knots_dev.fa"), knots_items)

# ---- 2) FS strict 端点 ----
fs = rd(os.path.join(ROOT, "data/curated/fold_switch_global.tsv"))
summ = {(r["pdb_id"].lower(), r["requested_chain"].upper()): r["observed_sequence"]
        for r in rd(B + "/manifests/fold_pair_endpoint_sequence_structure_summary.tsv")}
strict = [r for r in fs if r["target"] == "1" and r["valid_mask"] == "1"]
if len(strict) != 10:
    die(f"strict {len(strict)} != 10")
fs_items = []
for r in strict:
    for side, pdb, ch in (("A", r["pdb_a"], r["chain_a"]), ("B", r["pdb_b"], r["chain_b"])):
        fs_items.append((f"{r['pair_id']}_{side}", summ[(pdb.lower(), ch.upper())]))
write_fa(os.path.join(OUT, "fs_endpoints.fa"), fs_items)
qc["fs_strict_endpoints"] = len(fs_items)

# ---- 3) FS region 链（fine 行；usable/fine_only 分类复算） ----
reg = rd(os.path.join(ROOT, "data/curated/fold_switch_regions.tsv"))
strict_ids = {r["pair_id"] for r in strict}
fine_rows = [r for r in reg if r["pair_id"] in strict_ids
             and r["region_task_tier"] == "fine_candidate"]
usable_rows = [r for r in fine_rows if r["lit_label_decision"] == "usable"]
fine_only_rows = [r for r in fine_rows if r["lit_label_decision"] != "usable"]
usable_pairs = {r["pair_id"] for r in usable_rows}
fine_only_pairs = {r["pair_id"] for r in fine_only_rows} - usable_pairs
if len(usable_pairs) != 3 or len(usable_rows) != 6:
    die(f"usable 分类 {len(usable_pairs)} 对/{len(usable_rows)} 行 != 3/6（P1.15）")
if len(fine_only_pairs) != 6 or len(fine_only_rows) != 10:
    die(f"fine_only 分类 {len(fine_only_pairs)} 对/{len(fine_only_rows)} 行 != 6/10（P1.15）")
if not usable_pairs == {"porter_20_5c1vA__5c1vB", "porter_61_4gqcC__4gqcB", "porter_62_4o0pA__4o01D"}:
    die(f"usable 对集异常: {usable_pairs}")
pair_ep = {r["pair_id"]: {"A": (r["pdb_a"].lower(), r["chain_a"].upper()),
                          "B": (r["pdb_b"].lower(), r["chain_b"].upper())} for r in fs}
# 权威逐残基映射（旧项目 fold_pair_endpoint_residue_mapping.tsv；行序=观测序列序；
# unmapped 行也占位（仍有 label_seq_id），行总数必须=观测残基数）
ep_label = defaultdict(list)
for m in rd(B + "/manifests/fold_pair_endpoint_residue_mapping.tsv"):
    key = (m["pdb_id"].lower(), m["requested_chain"].upper())
    v = str(m["label_seq_id"]).strip()
    ep_label[key].append(int(v) if v.isdigit() else None)
region_intervals = defaultdict(list)
for r in sorted(fine_rows, key=lambda x: (x["entry_id"], int(x["label_seq_start"]))):
    name = r["entry_id"]
    eps = pair_ep[r["pair_id"]]
    key = eps["A"] if name == eps["A"][0] + eps["A"][1] else (
        eps["B"] if name == eps["B"][0] + eps["B"][1] else None)
    if key is None:
        die(f"region entry {name} 无法映射 {r['pair_id']} 端点 {eps}")
    seq = summ[key]
    lab = ep_label[key]
    if len(lab) != len(seq):
        die(f"region {name}: 映射行 {len(lab)} != 观测残基 {len(seq)}")
    st, en = int(r["label_seq_start"]), int(r["label_seq_end"])
    pos = [i + 1 for i, v in enumerate(lab) if v is not None and st <= v <= en]
    if not pos:
        die(f"region {name}: label {st}-{en} 无观测残基")
    if pos[-1] - pos[0] + 1 != len(pos):
        die(f"region {name}: 观测区间不连续 {pos[0]}-{pos[-1]}")
    region_intervals[name].append({"seq": seq, "start": pos[0], "end": pos[-1],
                                   "pair_id": r["pair_id"],
                                   "label_decision": r["lit_label_decision"]})
region_items, ridx_rows, iv_rows = [], [], []
for name, ivs in sorted(region_intervals.items()):
    seq = ivs[0]["seq"]
    if max(v["end"] for v in ivs) > len(seq):
        die(f"region {name}: 区间超长 {max(v['end'] for v in ivs)}>{len(seq)}")
    region_items.append((name, seq))
    # FS-REGION 存储域=全长链（分母=整链：区段内 y=1，区段外 y=0）；区间另存 intervals 供标注
    ridx_rows.append({"name": name, "resid_indices": ",".join(map(str, range(1, len(seq) + 1)))})
    for v in ivs:
        iv_rows.append({"name": name, "pair_id": v["pair_id"], "start": v["start"],
                        "end": v["end"], "label_decision": v["label_decision"]})
with open(os.path.join(OUT, "fs_region_intervals.tsv"), "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["name", "pair_id", "start", "end", "label_decision"],
                       delimiter="\t", lineterminator="\n")
    w.writeheader()
    w.writerows(iv_rows)
write_fa(os.path.join(OUT, "fs_region.fa"), region_items)
with open(os.path.join(OUT, "fs_region_resid_idx.tsv"), "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["name", "resid_indices"], delimiter="\t", lineterminator="\n")
    w.writeheader()
    w.writerows(ridx_rows)
qc["fs_region_chains"] = len(region_items)
qc["fs_usable_pairs"] = sorted(usable_pairs)
qc["fs_fine_only_pairs"] = sorted(fine_only_pairs)

# ---- 4) disorder dev（序列 + 分母残基域索引） ----
dp = json.load(open(os.path.join(ROOT, "data/raw/disprot/2026-09-22/disprot_api_search.json")))
dpseq = {x["acc"]: x["sequence"] for x in dp["data"]}
man_dev = rd(os.path.join(ROOT, "data/splits/split_manifest.tsv"))
dis_dev = [r["sample_id"].split(":", 1)[1] for r in man_dev
           if r["task_area"] == "disorder" and r["split"] == "development"]
dis_rows = rd(os.path.join(ROOT, "data/curated/disorder.tsv"))
id2acc = {r["disprot_id"]: r["uniprot_acc"] for r in dis_rows}
masks = rd(os.path.join(ROOT, "data/curated/disorder_masks.tsv"))
dom = defaultdict(list)
for r in masks:
    if r["state"] in ("0", "1") and r["mask"] == "1":
        dom[r["disprot_id"]].extend(range(int(r["start"]), int(r["end"]) + 1))
dis_items, dis_ridx = [], []
n_empty_domain = 0
n_beyond_trunc = 0
n_clip_residues = 0
for did in sorted(dis_dev):
    acc = id2acc.get(did)
    seq = dpseq.get(acc)
    if not seq:
        die(f"disorder {did}/{acc} 无序列")
    idx = sorted(set(dom.get(did, [])))
    if not idx:
        # 掩码域全空（全 NA）——按 P0.05 NA 规则 EMPTY_EVALUABLE_SET 出分母，跳过提取并登记
        n_empty_domain += 1
        continue
    # 冻结截断语义（trunc1022）：1022 之外的域残基不可提取，裁剪并登记
    clipped = [i for i in idx if i <= 1022]
    if len(clipped) < len(idx):
        n_clip_residues += len(idx) - len(clipped)
        idx = clipped
    if not idx:
        n_beyond_trunc += 1
        continue
    if max(idx) > len(seq):
        die(f"disorder {did}: 索引域异常（{len(idx)} 残基, len={len(seq)}）")
    dis_items.append((did, seq))
    dis_ridx.append({"name": did, "resid_indices": ",".join(map(str, idx))})
if len(dis_items) + n_empty_domain + n_beyond_trunc != 2392:
    die(f"disorder dev 提取+空域+全域超界 {len(dis_items)}+{n_empty_domain}+{n_beyond_trunc} != 2392")
if sum(len(x["resid_indices"].split(",")) for x in dis_ridx) > 223660:
    die("disorder 域残基总数 > 223,660（P3.02 冻结存储域口径）")
write_fa(os.path.join(OUT, "disorder_dev.fa"), dis_items)
with open(os.path.join(OUT, "disorder_resid_idx.tsv"), "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["name", "resid_indices"], delimiter="\t", lineterminator="\n")
    w.writeheader()
    w.writerows(dis_ridx)
qc["disorder_dev_entries"] = 2380
qc["disorder_extracted"] = len(dis_items)
qc["disorder_empty_domain_skipped"] = n_empty_domain
qc["disorder_domain_residues"] = sum(len(x["resid_indices"].split(",")) for x in dis_ridx)
qc["disorder_entries_all_domain_beyond_trunc1022"] = n_beyond_trunc
qc["disorder_domain_residues_clipped_by_trunc1022"] = n_clip_residues

# ---- 5) L2 stage-A：dev 簇代表 RandomState(2026) 无放回 10,000 ----
reps = set()
with gzip.open(os.path.join(ROOT, "data/splits/fs_l2_background_manifest.tsv.gz"), "rt") as f:
    for r in csv.DictReader(f, delimiter="\t"):
        if r["split"] == "development" and not r["uniprot_accession"].startswith("EP_"):
            reps.add(r["cluster_rep"])
reps = sorted(reps)
if len(reps) != 47595:
    die(f"dev 簇代表 {len(reps)} != 47,595（KNOT-RESPLIT 新版划分口径）")
rng = np.random.RandomState(2026)
sample = sorted(rng.choice(len(reps), size=10000, replace=False))
sample_accs = [reps[i] for i in sample]
need = set(sample_accs)
sp_path = os.path.join(ROOT, "data/raw/swissprot/2026-09-23/uniprot_sprot.fasta.gz")
sp_seq = {}
with gzip.open(sp_path, "rt") as f:
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
missing = [a for a in sample_accs if a not in sp_seq]
if missing:
    # 簇代表可能是 FS 端点（EP_{pair_id}_{side}__{pdb}{chain}，audit_leakage 命名规则）——从端点表取序列
    ep_map = {}
    for r in fs:
        for side, pdb, ch in (("A", r["pdb_a"], r["chain_a"]), ("B", r["pdb_b"], r["chain_b"])):
            ep_map[f"EP_{r['pair_id']}_{side}__{pdb.lower()}{ch.upper()}"] = \
                summ[(pdb.lower(), ch.upper())]
    still = [a for a in missing if a not in ep_map]
    if still:
        die(f"Swiss-Prot fasta 缺 {len(still)} 个抽样 accession 且非 EP_ 端点: {still[:5]}")
    for a in missing:
        sp_seq[a] = ep_map[a]
l2_items = [(a, sp_seq[a]) for a in sample_accs]
write_fa(os.path.join(OUT, "l2_stage_a.fa"), l2_items)
with open(os.path.join(OUT, "stage_a_manifest.tsv"), "w", newline="") as f:
    w = csv.writer(f, delimiter="\t", lineterminator="\n")
    w.writerow(["accession", "cluster_rep", "sample_index"])
    pos = {a: i for i, a in enumerate(reps)}
    for i, a in enumerate(sample_accs):
        w.writerow([a, pos[a], i])
qc["l2_stage_a"] = len(l2_items)
qc["l2_stage_a_sha_head"] = hashlib.sha256(json.dumps(sample_accs).encode()).hexdigest()[:16]

json.dump(qc, open(os.path.join(OUT, "inputs_qc.json"), "w"), ensure_ascii=False, indent=1, sort_keys=True)
print(f"[p303in] OK {json.dumps({k: v for k, v in qc.items() if not isinstance(v, list)}, ensure_ascii=False)}")
