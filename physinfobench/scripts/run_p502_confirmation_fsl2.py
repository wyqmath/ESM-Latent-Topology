#!/usr/bin/env python3
"""P5.02 确认集复核（本地侧）：H-FSL2-RANK。

钉版语义（confirmation_lock）：
  1. n_strict 复算：manifest(task_area=fold_switch, split=confirmation) ∩
     fold_switch_global(target=1 ∧ valid_mask=1)；== lock 钉版值 0 → 按预注册规则
     verdict=insufficient_evidence（不构成失败）；主指标 NA。
  2. 描述性 side-table（注册在案、无阈值、无 verdict、不入结论）：6 对确认对端点均值向量
     （hidden 29）由冻结 stage-B 协议打分（dev 代表帧+10 dev strict 重拟合，liblinear
     确定性 seed 2026），重拟合分数与冻结 scores.tsv.gz 一致性断言
     （统计等价判据：抖动上界 max<=1e-4 / mean<=1e-6 + pearson>=1-1e-9 +
     秩变动与 percentile ±maxd 区间披露；lock change_log #3）后排序。
  3. first_read 守卫与集群侧一致；本运行不消费任何确认集标签值（确认对标签全空，
     n_strict 检查为构成统计，已登记 exposure_log P5.01 行）。
"""
import csv
import gzip
import json
import os
import subprocess
import sys
from datetime import datetime, timezone

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run_p502_confirmation_cluster import (die, first_read_guard, log, now_pair,
                                           rd, verify_lock)

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RUN_ID = "CONF-FSL2-RANK-P502-v1"
SEED = 2026


def load_packed_reps():
    reps = {}
    for pth in ("data/interim/p303/emb/packed_l2_stage_b.npz",
                "data/interim/p303/emb/packed_l2sb_delta.npz"):
        fp = os.path.join(ROOT, pth)
        if not os.path.exists(fp):
            continue
        with np.load(fp, allow_pickle=True) as z:
            X = z["X"].astype(np.float32)
            names = json.loads(str(z["names"]))
        for i, n in enumerate(names):
            reps[n] = X[i]
        log(f"loaded {pth}: {X.shape}")
    return reps


def ensure_conf_endpoints_emb():
    d = os.path.join(ROOT, "data/interim/p502/emb_fs_conf_endpoints")
    if os.path.exists(os.path.join(d, "extract_manifest.tsv")):
        return d
    log("FS 确认端点本地抽取（12 链，CPU）")
    os.makedirs(d, exist_ok=True)
    py = os.path.join(ROOT, ".venv_local/bin/python")
    if not os.path.exists(py):
        py = sys.executable
    subprocess.run([py, os.path.join(ROOT, "scripts/extract_representations.py"),
                    "--fasta", os.path.join(ROOT, "data/interim/p501/fs_conf_endpoints.fa"),
                    "--out-dir", d,
                    "--mean-layers", "5,11,17,23,29,33",
                    "--device", "cpu"], check=True, cwd=ROOT)
    return d


def main():
    lock = verify_lock()
    outd, now = first_read_guard(lock, RUN_ID)

    man = rd(os.path.join(ROOT, "data/splits/split_manifest.tsv"))
    fs = {r["pair_id"]: r for r in rd(os.path.join(ROOT, "data/curated/fold_switch_global.tsv"))}
    conf_pairs = [r["sample_id"] for r in man
                  if r["task_area"] == "fold_switch" and r["split"] == "confirmation"]
    n_strict = sum(1 for p in conf_pairs
                   if fs[p]["target"] == "1" and fs[p]["valid_mask"] == "1")
    locked_n = int(lock["confirmation_sets"]["fs_strict"]["n_strict_locked"])
    if n_strict != locked_n:
        die(f"n_strict 复算 {n_strict} != lock 钉版 {locked_n}（偏离须走 P5.03 回路）")
    if n_strict != 0:
        die(f"n_strict={n_strict}>0：完整评价路径未在 lock 注册（lock 钉版=0 分支）")
    verdict = "insufficient_evidence"
    log(f"n_strict={n_strict}（lock 钉版 {locked_n}）→ verdict={verdict}")

    # —— 描述性 side-table：冻结 stage-B 协议重拟合（确定性）+ 全量一致性断言 ——
    reps = load_packed_reps()
    new_dev_reps = set()
    with gzip.open(os.path.join(ROOT, "data/splits/fs_l2_background_manifest.tsv.gz"), "rt") as f:
        for r in csv.DictReader(f, delimiter="\t"):
            if r["split"] == "development" and not r["uniprot_accession"].startswith("EP_"):
                new_dev_reps.add(r["cluster_rep"])
    missing = new_dev_reps - set(reps)
    if missing:
        die(f"dev 代表缺嵌入 {len(missing)}")
    reps = {k: reps[k] for k in new_dev_reps}
    if len(reps) != 47595:
        die(f"代表嵌入 {len(reps)} != 47,595")

    fe = rd(os.path.join(ROOT, "data/interim/p303/emb/fs_endpoints/extract_manifest.tsv"))
    key2e = {}
    for row in fe:
        if row["key"] not in key2e:
            with np.load(os.path.join(ROOT, "data/interim/p303/emb/fs_endpoints",
                                      row["key"] + ".npz"), allow_pickle=False) as z:
                key2e[row["key"]] = z["mean_layers"].astype(np.float32)
    e_by_name = {row["name"]: key2e[row["key"]] for row in fe}
    strict = [r for r in rd(os.path.join(ROOT, "data/curated/fold_switch_global.tsv"))
              if r["target"] == "1" and r["valid_mask"] == "1"]
    dev_ids = {r["sample_id"] for r in man if r["split"] == "development"}
    for r in strict:
        if r["pair_id"] not in dev_ids:
            die(f"dev strict 混入非 dev: {r['pair_id']}")
    Xp = np.stack([(e_by_name[f"{r['pair_id']}_A"][4] +
                    e_by_name.get(f"{r['pair_id']}_B", e_by_name[f"{r['pair_id']}_A"])[4]) / 2
                   for r in strict])
    from sklearn.linear_model import LogisticRegression
    Xbg = np.stack([reps[k] for k in sorted(reps)])[:, 4]
    X = np.concatenate([Xp, Xbg])
    y = np.array([1] * len(Xp) + [0] * len(Xbg))
    clf = LogisticRegression(C=0.1, class_weight="balanced", solver="liblinear",
                             max_iter=1000, random_state=SEED)
    clf.fit(X, y)
    refit_rep_score = {k: float(v) for k, v in zip(sorted(reps), clf.predict_proba(Xbg)[:, 1])}

    stored_rep = {}
    n_rows = 0
    with gzip.open(os.path.join(ROOT,
                    "results/probes/T-FS-L2-PU-RANK.stageB.scores.tsv.gz"), "rt") as f:
        for r in csv.DictReader(f, delimiter="\t"):
            n_rows += 1
            s = stored_rep.setdefault(r["cluster_rep"], float(r["score"]))
            if abs(s - float(r["score"])) > 1e-9:
                die(f"存储分数同簇不一致 {r['cluster_rep']}")
    common = sorted(set(stored_rep) & set(refit_rep_score))
    a_st = np.array([stored_rep[k] for k in common])
    a_rf = np.array([refit_rep_score[k] for k in common])
    diffs = np.abs(a_st - a_rf)
    maxd = float(diffs.max())
    meand = float(diffs.mean())
    r = float(np.corrcoef(a_st, a_rf)[0, 1])
    # 统计等价判据（lock change_log #3）：点向抖动上界 + 高度相关；
    # 秩变动仅在近并列处发生且不改排序型结论——变动规模与对交付量的影响区间
    # （每对 percentile 的 ±maxd 上下界）如实量化披露，不做位级/序级硬断言。
    if maxd > 1e-4 or meand > 1e-6:
        die(f"重拟合与冻结分数抖动超界 max|Δ|={maxd:.2e} mean|Δ|={meand:.2e}")
    if r < 1 - 1e-9:
        die(f"重拟合与冻结分数相关性不足 pearson={r!r}")
    rank_st = np.empty(len(a_st), dtype=int)
    rank_st[np.argsort(-a_st, kind="stable")] = np.arange(1, len(a_st) + 1)
    rank_rf = np.empty(len(a_rf), dtype=int)
    rank_rf[np.argsort(-a_rf, kind="stable")] = np.arange(1, len(a_rf) + 1)
    n_rank_changed = int((rank_st != rank_rf).sum())
    max_rank_shift = int(np.abs(rank_st - rank_rf).max())
    log(f"重拟合一致性断言通过（{len(common)} 代表, max|Δ|={maxd:.2e}, "
        f"mean|Δ|={meand:.2e}, pearson={r:.10f}, 秩变动 {n_rank_changed} "
        f"(最大位移 {max_rank_shift})）")

    ced = ensure_conf_endpoints_emb()
    fe2 = rd(os.path.join(ced, "extract_manifest.tsv"))
    key2c = {}
    for row in fe2:
        if row["key"] not in key2c:
            with np.load(os.path.join(ced, row["key"] + ".npz"), allow_pickle=False) as z:
                key2c[row["key"]] = z["mean_layers"].astype(np.float32)
    c_by_name = {row["name"]: key2c[row["key"]] for row in fe2}
    desc = []
    stored_scores = np.array(list(stored_rep.values()))
    dev_strict_scores = clf.predict_proba(Xp)[:, 1]
    for p in conf_pairs:
        a = c_by_name[f"{p}_A"]
        b = c_by_name.get(f"{p}_B", a)
        sc = float(clf.predict_proba(((a[4] + b[4]) / 2).reshape(1, -1))[:, 1][0])
        pool = np.concatenate([stored_scores, dev_strict_scores, np.array([sc])])
        rank = int((pool > sc).sum()) + 1
        rank_lo = int((pool > sc + maxd).sum()) + 1   # 抖动上界扰动后的最好名次
        rank_hi = int((pool > sc - maxd).sum()) + 1   # 最差名次
        desc.append({"pair_id": p, "score": round(sc, 6), "rank": rank,
                     "percentile": round(rank / len(pool), 8), "pool_n": int(len(pool)),
                     "rank_interval_jitter": [rank_lo, rank_hi],
                     "percentile_interval_jitter": [round(rank_lo / len(pool), 8),
                                                    round(rank_hi / len(pool), 8)],
                     "tier_locked": fs[p]["tier"], "valid_mask_locked": fs[p]["valid_mask"]})

    metrics = {"run_id": RUN_ID, "hypothesis": "H-FSL2-RANK", "executed_at": now,
               "n_strict_recomputed": n_strict, "n_strict_locked": locked_n,
               "verdict": verdict,
               "primary_metrics": "NA（n_strict=0 → 预注册证据不足分支）",
               "descriptive_side_table_note": "无阈值无 verdict；仅供 P5.06 裁决上下文",
               "scorer_verification": {"stored_rows": n_rows, "reps_compared": len(common),
                                       "max_abs_diff": maxd, "mean_abs_diff": meand,
                                       "pearson": r,
                                       "n_rank_changed": n_rank_changed,
                                       "max_rank_shift": max_rank_shift,
                                       "criterion": "max<=1e-4 + mean<=1e-6 + pearson>=1-1e-9；秩变动与 percentile ±maxd 区间如实披露（change_log #3）"},
               "descriptive_pairs": desc}
    json.dump(metrics, open(os.path.join(outd, "metrics.json"), "w"), indent=1,
              ensure_ascii=False, sort_keys=True)
    with open(os.path.join(outd, "descriptive_pairs.tsv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["pair_id", "score", "rank", "percentile",
                                          "pool_n", "rank_interval_jitter",
                                          "percentile_interval_jitter",
                                          "tier_locked", "valid_mask_locked"],
                           delimiter="\t", lineterminator="\n")
        w.writeheader()
        w.writerows(desc)
    log(f"DONE {RUN_ID} verdict={verdict} 描述行={len(desc)}")


if __name__ == "__main__":
    main()
