#!/usr/bin/env python3
"""P3.03 T-FS-L2-PU-RANK stage-B：代表帧打分 + 分数传播 + 官方 L2 指标。

实现语义（P3.03 修订，见报告）：stage-B 宇宙=dev 背景簇代表 47,604（CPU 算力实测下替代
protein 级 343,682 的直推打分），成员分数=所属代表分数（传播）；官方指标在传播后的
全宇宙（343,682 背景 + 10 strict pair 单元）上计算。同簇同分为设计内性质。
输出：results/probes/T-FS-L2-PU-RANK.stageB.{scores.tsv.gz,metrics.json,propagation_qc.json}
"""
import csv
import gzip
import json
import os
import sys
from collections import defaultdict
from datetime import datetime

import numpy as np
from sklearn.linear_model import LogisticRegression

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
EMB = os.path.join(ROOT, "data/interim/p303/emb")
OUTD = os.path.join(ROOT, "results/probes")
SEED = 2026


def die(m):
    print(f"[stageb FATAL] {m}", file=sys.stderr)
    sys.exit(1)


def rd(p, d="\t"):
    with open(p, newline="") as f:
        return list(csv.DictReader(f, delimiter=d))


sel = json.load(open(os.path.join(OUTD, "T-FS-L2-PU-RANK.stageA.selected.json")))
layer, C = sel["layer"], sel["C"]
print(f"[stageb] selected layer={layer} C={C}")

# 代表帧嵌入（KNOT-RESPLIT 新版）：旧 packed 矩阵 ∪ 增量 pack，过滤到新版 dev 代表帧
reps = {}
packed_old = os.path.join(EMB, "packed_l2_stage_b.npz")
packed_delta = os.path.join(EMB, "packed_l2sb_delta.npz")
for pth in (packed_old, packed_delta):
    if not os.path.exists(pth):
        continue
    with np.load(pth, allow_pickle=True) as z:
        X = z["X"].astype(np.float32)
        names = json.loads(str(z["names"]))
    for i, n in enumerate(names):
        reps[n] = X[i]
    print(f"[stageb] loaded {pth}: {X.shape}")
# 新版 dev 代表帧
new_dev_reps = set()
import gzip as _gz
with _gz.open(os.path.join(ROOT, "data/splits/fs_l2_background_manifest.tsv.gz"), "rt") as f:
    for r in csv.DictReader(f, delimiter="\t"):
        if r["split"] == "development" and not r["uniprot_accession"].startswith("EP_"):
            new_dev_reps.add(r["cluster_rep"])
missing = new_dev_reps - set(reps)
if missing:
    die(f"新版 dev 代表缺嵌入 {len(missing)}: {sorted(missing)[:5]}")
reps = {k: reps[k] for k in new_dev_reps}
if len(reps) != 47595:
    die(f"代表嵌入 {len(reps)} != 47,595（新版划分帧）")

# 阳性 pair 向量（两端点逐层均值；porter_87 同序列回退由 extract 端处理）
fe = rd(os.path.join(EMB, "fs_endpoints", "extract_manifest.tsv"))
key2e = {}
for row in fe:
    if row["key"] not in key2e:
        with np.load(os.path.join(EMB, "fs_endpoints", row["key"] + ".npz"), allow_pickle=False) as z:
            key2e[row["key"]] = z["mean_layers"].astype(np.float32)
e_by_name = {row["name"]: key2e[row["key"]] for row in fe}
fs = [r for r in rd(os.path.join(ROOT, "data/curated/fold_switch_global.tsv"))
      if r["target"] == "1" and r["valid_mask"] == "1"]
pos = {}
for r in fs:
    a = e_by_name[f"{r['pair_id']}_A"]
    b_name = f"{r['pair_id']}_B"
    if b_name not in e_by_name:
        die(f"fold-switch 阳性端点缺失: {b_name}")
    b = e_by_name[b_name]
    pos[r["pair_id"]] = (a + b) / 2

# PU 直推拟合（代表帧 + 10 阳性）
Xp = np.stack([pos[p][layer - 1] for p in sorted(pos)])
Xbg = np.stack([reps[k][layer - 1] for k in sorted(reps)])
X = np.concatenate([Xp, Xbg])
y = np.array([1] * len(Xp) + [0] * len(Xbg))
clf = LogisticRegression(C=C, class_weight="balanced", solver="liblinear",
                         max_iter=1000, random_state=SEED)
clf.fit(X, y)
rep_score = {k: float(v) for k, v in zip(sorted(reps), clf.predict_proba(Xbg)[:, 1])}

# 分数传播
member_of = defaultdict(list)
with gzip.open(os.path.join(ROOT, "data/splits/fs_l2_background_manifest.tsv.gz"), "rt") as f:
    for r in csv.DictReader(f, delimiter="\t"):
        if r["split"] == "development" and not r["uniprot_accession"].startswith("EP_"):
            member_of[r["cluster_rep"]].append(r["uniprot_accession"])
rows = []
for rep, mem in member_of.items():
    s = rep_score.get(rep)
    if s is None:
        die(f"代表 {rep} 无分数")
    for m in mem:
        rows.append((m, rep, s))
# 期望=新版 manifest 的 dev 非 EP_ 背景行数（动态推导，不硬编码旧值）
import gzip as _g
_expect = 0
with _g.open(os.path.join(ROOT, "data/splits/fs_l2_background_manifest.tsv.gz"), "rt") as f:
    for r in csv.DictReader(f, delimiter="\t"):
        if r["split"] == "development" and not r["uniprot_accession"].startswith("EP_"):
            _expect += 1
if len(rows) != _expect:
    die(f"传播后背景 {len(rows)} != manifest dev 背景 {_expect}")

# 官方指标（全宇宙=343,682 背景 + 10 strict pair 单元）
pos_scores = clf.predict_proba(Xp)[:, 1]
all_scores = np.array([s for _, _, s in rows] + pos_scores.tolist())
all_is_pos = np.array([0] * len(rows) + [1] * len(pos_scores))
order = np.argsort(-all_scores, kind="stable")
ranks = np.empty(len(all_scores), dtype=int)
ranks[order] = np.arange(1, len(all_scores) + 1)
pos_ranks = ranks[all_is_pos == 1]
ks = [10, 100, 1000, 10000]
metrics = {"layer": layer, "C": C, "universe": int(len(all_scores)),
           "run_ts": datetime.now().strftime("%Y-%m-%d %H:%M")}
for k in ks:
    metrics[f"recall@{k}"] = round(float((pos_ranks <= k).mean()), 6)
    metrics[f"enrichment@{k}"] = round(float((pos_ranks <= k).mean() / (k / len(all_scores))), 6)
metrics["positive_rank_percentile"] = round(float(np.mean(pos_ranks) / len(all_scores)), 6)
metrics["universe_bg_expected"] = _expect
metrics["pos_ranks"] = sorted(int(r) for r in pos_ranks)

with gzip.open(os.path.join(OUTD, "T-FS-L2-PU-RANK.stageB.scores.tsv.gz"), "wt", newline="") as f:
    w = csv.writer(f, delimiter="\t", lineterminator="\n")
    w.writerow(["uniprot_accession", "cluster_rep", "score"])
    w.writerows(rows)
json.dump(metrics, open(os.path.join(OUTD, "T-FS-L2-PU-RANK.stageB.metrics.json"), "w"),
          indent=1, sort_keys=True)

# 传播敏感性：同簇分数同质性=构造性成立；报告簇大小分布与代表覆盖率
sizes = sorted((len(v) for v in member_of.values()), reverse=True)
qc = {"clusters": len(member_of), "members": len(rows),
      "cluster_size_top10": sizes[:10],
      "single_member_clusters": sum(1 for s in sizes if s == 1),
      "positives_share_cache_note": "porter_87 两端同序列共享缓存条目（pair 向量=A+B 均值，退化为 A）",
      "semantics": "rep-frame scoring + propagation；同簇同分为构造性性质，非模型性质"}
json.dump(qc, open(os.path.join(OUTD, "T-FS-L2-PU-RANK.stageB.propagation_qc.json"), "w"),
          ensure_ascii=False, indent=1, sort_keys=True)
print(f"[stageb] OK {json.dumps({k: v for k, v in metrics.items() if k != 'pos_ranks'})}")
