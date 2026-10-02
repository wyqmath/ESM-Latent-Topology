#!/usr/bin/env python3
"""P2.06：数据偏差基线（L2 排序，先于 PLM 表示；evaluation_protocol 三层对照矩阵）。

宇宙=L2 development 划分（背景 dev 蛋白 343,682 + 10 strict positive）。
基线（预注册，仅样本元数据/序列派生，禁模型输出）：
  B0 random 排序（种子 2026）；
  B1 sequence_length（背景表 sequence_length）；
  B2 研究覆盖度（n_pdb_entries_crossref）；
  B3 logistic（特征=length/n_pdb；PU 语义 observed-positive vs unlabeled，全宇宙直推拟合——诊断性排序，非泛化预测）；
指标：recall_at_k / enrichment_at_k / positive_rank_percentile（k=10/100/1000/10000，P1.12 冻结）。
校准/ECE 关闭（L2 非校准任务）。统计不确定性=bootstrap 待统计参数冻结（P3.02 前），本轮仅点估计。
输出：results/baselines/l2_development.tsv + report md。
"""
import csv
import gzip
import json
import os
import random
import sys

import numpy as np
from collections import defaultdict

sys.path.insert(0, os.path.dirname(__file__))
from metrics import recall_at_k, enrichment_at_k, positive_rank_percentile

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SEED = 2026
KS = [10, 100, 1000, 10000]
OUT = os.path.join(ROOT, "results/baselines/l2_development.tsv")
OUT_MD = os.path.join(ROOT, "reports/baselines_l2_development.md")
QC = os.path.join(ROOT, "results/baselines/l2_development_qc.json")


def die(m):
    print(f"[p206 FATAL] {m}", file=sys.stderr)
    sys.exit(1)


# ---- 宇宙：L2 development ----
bg = {}
with gzip.open(os.path.join(ROOT, "data/splits/fs_l2_background_manifest.tsv.gz"), "rt") as f:
    for r in csv.DictReader(f, delimiter="\t"):
        u = r["uniprot_accession"]
        if r["split"] == "development" and not u.startswith("EP_"):  # 端点成员属有标注 pair，不入背景宇宙
            bg[u] = r["group_node"]
meta = {}
with gzip.open(os.path.join(ROOT, "data/curated/fold_switch_unlabeled_sequence.tsv.gz"), "rt") as f:
    for r in csv.DictReader(f, delimiter="\t"):
        if r["uniprot_accession"] in bg:
            meta[r["uniprot_accession"]] = r
# 补元数据：matched controls 中不在背景 A 的 7 个（UniProt REST fasta 实测长度；n_pdb 取 P1.11 候选表）
mc_extra = {}
pn = {r["uniprot_accession"]: r for r in csv.DictReader(open(os.path.join(ROOT, "data/curated/fold_switch_putative_negative_candidates.tsv"), newline=""), delimiter="\t")}
mc_dir = os.path.join(ROOT, "data/raw/p202_matched_controls/2026-09-24")
if os.path.isdir(mc_dir):
    for fn in sorted(os.listdir(mc_dir)):
        if fn.startswith("._") or not fn.endswith(".fasta"):
            continue
        acc = fn[:-6]
        n = sum(len(l.strip()) for l in open(os.path.join(mc_dir, fn)) if not l.startswith(">"))
        mc_extra[acc] = {"sequence_length": str(n),
                         "n_pdb_entries_crossref": pn.get(acc, {}).get("n_pdb_entries", "0")}
missing = set(bg) - set(meta) - set(mc_extra)
if missing:
    die(f"背景 dev 有 {len(missing)} 个 accession 无元数据")
pos = []
for r in csv.DictReader(open(os.path.join(ROOT, "data/curated/fold_switch_global.tsv"), newline=""), delimiter="\t"):
    if r["tier"] == "strict_state_candidate":
        pos.append(r["uniprot_a"])
# strict 的 L2 节点：positive 蛋白加入宇宙（不在背景 A）
universe = sorted(set(bg) | set(pos))
N = len(universe)
print(f"universe={N} (bg {len(bg)} + pos {len(pos)})")

# ---- 特征 ----
feat_len = {}
feat_pdb = {}
strict_len = {"B9W5G6": 179, "P00573": 883, "P19726": 345, "P38505": 134, "Q08209": 521,
              "Q12931": 704, "Q58AD3": 94, "Q8E473": 1310, "Q9RZA4": 755, "Q9YA14": 161}
for u in universe:
    m = meta.get(u) or mc_extra.get(u)
    feat_len[u] = int(m["sequence_length"]) if m else strict_len[u]
    feat_pdb[u] = int(m["n_pdb_entries_crossref"]) if m else 0

def rank_random(rng):
    ids = universe[:]
    rng.shuffle(ids)
    return ids

def rank_by(keyfun, desc=True):
    return sorted(universe, key=lambda u: (keyfun(u) * (-1 if desc else 1), u))

def rank_logistic():
    """预登记口径（任务树原文）：逻辑回归（observed-positive vs unlabeled）——全宇宙直推拟合。
    语义披露：这是数据偏差基线的诊断性排序（非泛化预测）；正例仅 10 个且不属背景簇，
    不存在可用的"标注训练折"（曾试折 0 拟合，5 折均无正例，已放弃并在此披露）。
    禁模型输出；特征=长度+结构数。"""
    from sklearn.linear_model import LogisticRegression
    X, y, ids = [], [], []
    for u in universe:
        X.append([feat_len[u], feat_pdb[u]])
        y.append(1 if u in set(pos) else 0)
        ids.append(u)
    clf = LogisticRegression(max_iter=1000, class_weight="balanced")
    clf.fit(np.array(X), y)
    scored = clf.predict_proba(np.array(X))[:, 1]
    return [u for _, u in sorted(zip(-scored, ids))]

baselines = {
    "B0_random": rank_random(random.Random(SEED)),
    "B1_length_desc": rank_by(lambda u: feat_len[u]),
    "B2_n_pdb_desc": rank_by(lambda u: feat_pdb[u]),
    "B3_logistic_len_pdb": rank_logistic(),
}

rows = []
for name, ranked in baselines.items():
    for k in KS:
        rows.append({"baseline": name, "k": k,
                     "recall_at_k": recall_at_k(ranked, pos, k),
                     "enrichment_at_k": enrichment_at_k(ranked, pos, k, N)})
    rows.append({"baseline": name, "k": "percentile",
                 "recall_at_k": "", "enrichment_at_k": "",
                 "positive_rank_percentile": positive_rank_percentile(ranked, pos)})

os.makedirs(os.path.dirname(OUT), exist_ok=True)
cols = ["baseline", "k", "recall_at_k", "enrichment_at_k", "positive_rank_percentile"]
with open(OUT, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=cols, delimiter="\t", lineterminator="\n")
    w.writeheader()
    for r in rows:
        r.setdefault("positive_rank_percentile", "")
        w.writerow(r)

qc = {"universe": N, "positives": len(pos), "seed": SEED,
      "baselines": list(baselines), "ks": KS,
      "note": "点估计；bootstrap 待统计参数冻结（P3.02 前）；无模型输出参与"}
with open(QC, "w") as f:
    json.dump(qc, f, ensure_ascii=False, indent=1, sort_keys=True)

md = ["# L2 数据偏差基线（P2.06；development 划分）", "",
      f"宇宙={N}（背景 dev {len(bg)} + strict 10）；基线只用元数据（长度/结构数/随机/logistic），无模型输出。", "",
      "| baseline | k=10 R/E | k=100 R/E | k=1000 R/E | k=10000 R/E | rank_pct |",
      "|---|---|---|---|---|---|"]
byb = defaultdict(dict)
for r in rows:
    byb[r["baseline"]][str(r["k"])] = r
for b, d in byb.items():
    def fmt(k, field):
        v = d.get(k, {}).get(field)
        return f"{v:.3f}" if isinstance(v, float) else "-"
    cells = ["{}/{}".format(fmt(k, "recall_at_k"), fmt(k, "enrichment_at_k")) for k in ("10", "100", "1000", "10000")]
    md.append(f"| {b} | " + " | ".join(cells) + f" | {fmt('percentile','positive_rank_percentile')} |")
md += ["", "判读边界：这些是偏差基线的点估计（bootstrap 待 P3.02 统计参数冻结）；"
       "PLM 相对基线的增益将来必须在共同样本交集上计算（协议 generic_rules）。"
       "校准/ECE 对 L2 关闭（PU 排序任务）。"]
with open(OUT_MD, "w") as f:
    f.write("\n".join(md) + "\n")
print(f"[p206] baselines={list(baselines)} universe={N} -> {OUT}")
