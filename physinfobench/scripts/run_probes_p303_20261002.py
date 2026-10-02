#!/usr/bin/env python3
"""P3.03：多层线性探针（configs/probes.yaml 冻结配置；仅 development；确认/保留零接触）。

实现（冻结语义）：
  - 读取器=逻辑回归（C 网格 0.01/0.1/1/10，class_weight=balanced，liblinear，max_iter=1000）；
  - 选择=dev_fold 组级 CV（折值=split_manifest.dev_fold；FS 小任务=其组 dev_fold 留一）；
  - 指标=AUROC / Macro-F1（真值出现类）/ residue AUPRC（per-protein，P0.05）；
  - FS-L2 stage-A 选择宇宙=抽样 10,000 簇代表+10 strict（pair 向量=两端点逐层均值——实现语义，
    报告披露）；排序直推语义（P2.06 双披露延续）；选择指标=positive_rank_percentile；
  - FS-L1 潜能臂=描述性（两端点逐层余弦），无 CV 无主张。
环境变量 P303_ONLY="KNOT_PRESENCE,FS_REGION" 可只跑子集（emb 未就绪则整批要求就绪）。
输出：results/probes/{task}.selection.tsv、{task}.selected.json、summary.tsv、qc.json
"""
import csv
import glob
import json
import os
import sys
from collections import defaultdict
from datetime import datetime

import numpy as np
import yaml
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RUN_TS = datetime.now().strftime("%Y-%m-%d %H:%M")
EMB = os.path.join(ROOT, "data/interim/p303/emb_20261002")
OUTD = os.path.join(ROOT, "results/repairs/20261002/probes")
SEED = 2026
C_GRID = [0.01, 0.1, 1, 10]
ONLY = set(filter(None, os.environ.get("P303_ONLY", "").split(",")))
MAN = list(csv.DictReader(open(os.path.join(ROOT, "data/splits/split_manifest.tsv")), delimiter="\t"))
CFG = yaml.safe_load(open(os.path.join(ROOT, "configs/probes.yaml")))


def die(m):
    print(f"[p303probe FATAL] {m}", file=sys.stderr)
    sys.exit(1)


def want(tag):
    return not ONLY or tag in ONLY


def rd(p, d="\t"):
    with open(p, newline="") as f:
        return list(csv.DictReader(f, delimiter=d))


def load_emb(task_dir):
    """name→arrays（经 extract_manifest 的 name→key 解析；同序列共享 npz 的样本各自可解析）。"""
    man = rd(os.path.join(task_dir, "extract_manifest.tsv"))
    key2arr = {}
    for row in man:
        key = row["key"]
        if key not in key2arr:
            with np.load(os.path.join(task_dir, key + ".npz"), allow_pickle=False) as z:
                key2arr[key] = {k: z[k] for k in z.files if k != "meta"}
    return {row["name"]: key2arr[row["key"]] for row in man}


def cv_global(X, items, folds, fit_metric):
    """全局任务：每配置每折指标。items=[(name,y)]。"""
    out = []
    layers = X[next(iter(X))]["mean_layers"].shape[0]
    for layer in range(layers):
        for C in C_GRID:
            scores = []
            im = dict(items)
            for f in sorted(set(folds.values())):
                tr = [n for n, _ in items if folds[n] != f]
                te = [n for n, _ in items if folds[n] == f]
                if not tr or not te or len(set(im[n] for n in tr)) < 2 or len(set(im[n] for n in te)) < 2:
                    continue
                Xtr = np.stack([X[n]["mean_layers"][layer] for n in tr])
                Xte = np.stack([X[n]["mean_layers"][layer] for n in te])
                ytr = [im[n] for n in tr]
                yte = [im[n] for n in te]
                clf = LogisticRegression(C=C, class_weight="balanced",
                                         solver="lbfgs" if fit_metric == "macro_f1" else "liblinear",
                                         max_iter=1000, random_state=SEED)
                clf.fit(Xtr, ytr)
                if fit_metric == "auroc":
                    s = roc_auc_score(yte, clf.predict_proba(Xte)[:, 1])
                else:
                    s = f1_score(yte, clf.predict(Xte), average="macro", labels=sorted(set(yte)))
                scores.append(round(float(s), 6))
            val = scores or [None]
            out.append({"layer": layer + 1, "C": C, "metric": fit_metric,
                        "cv_mean": round(float(np.mean(scores)), 6) if scores else None,
                        "cv_std": round(float(np.std(scores)), 6) if scores else None,
                        "folds": scores})
    return out


def write_selection(fname, rows):
    with open(os.path.join(OUTD, fname), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()), delimiter="\t", lineterminator="\n")
        w.writeheader()
        w.writerows(rows)


def pick(rows, metric):
    cands = [r for r in rows if r.get("cv_mean") is not None]
    return max(cands, key=lambda r: (r[metric], -C_GRID.index(r["C"]), -r["layer"]))


SUMMARY = []
QC = {"run_ts": RUN_TS, "tasks": {}}
os.makedirs(OUTD, exist_ok=True)


def t_knot_presence():
    kseq = {r["record_id"]: r for r in rd(os.path.join(ROOT, "data/curated/knots_sequences.tsv"))}
    kX = load_emb(os.path.join(EMB, "knots_dev"))
    items = []
    for r in MAN:
        if r["task_area"] == "knot" and r["split"] == "development":
            rec = r["sample_id"].split(":", 1)[1]
            if rec in kseq and r["sample_id"] in kX:
                items.append((r["sample_id"], int(kseq[rec]["presence_target"])))
    if len(items) < 600:
        die(f"knots dev 样本 {len(items)} 异常")
    folds = {n: next(x["dev_fold"] for x in MAN if x["sample_id"] == n) for n, _ in items}
    rows = cv_global(kX, items, folds, "auroc")
    write_selection("T-KNOT-PRESENCE.selection.tsv", rows)
    best = pick(rows, "cv_mean")
    json.dump(best, open(os.path.join(OUTD, "T-KNOT-PRESENCE.selected.json"), "w"), indent=1)
    SUMMARY.append(["T-KNOT-PRESENCE", "AUROC", best["layer"], best["C"], best["cv_mean"],
                    best["cv_std"], len(items)])
    QC["tasks"]["T-KNOT-PRESENCE"] = {"n": len(items), "selected": best}


def t_knot_type():
    kt = {r["record_id"]: r for r in rd(os.path.join(ROOT, "data/curated/knots.tsv"))
          if r["type_task_tier"] == "eligible"}
    kX = load_emb(os.path.join(EMB, "knots_dev"))
    items = []
    for r in MAN:
        if r["task_area"] == "knot" and r["split"] == "development":
            rec = r["sample_id"].split(":", 1)[1]
            if rec in kt and r["sample_id"] in kX:
                items.append((r["sample_id"], kt[rec]["c2_primary"]))
    # 可评价口径=（manifest dev ∧ type-eligible ∧ 有已嵌入序列）；无序列链数如实登记
    dev_type_all = [r["sample_id"] for r in MAN
                    if r["task_area"] == "knot" and r["split"] == "development"
                    and r["sample_id"].split(":", 1)[1] in kt]
    no_seq = [s for s in dev_type_all if s not in kX]
    if len(items) != len(dev_type_all) - len(no_seq):
        die(f"knot type dev 可评价数 {len(items)} != 预期 {len(dev_type_all)}−{len(no_seq)}")
    folds = {n: next(x["dev_fold"] for x in MAN if x["sample_id"] == n) for n, _ in items}
    rows = cv_global(kX, items, folds, "macro_f1")
    write_selection("T-KNOT-TYPE.selection.tsv", rows)
    best = pick(rows, "cv_mean")
    json.dump(best, open(os.path.join(OUTD, "T-KNOT-TYPE.selected.json"), "w"), indent=1)
    SUMMARY.append(["T-KNOT-TYPE", "Macro-F1", best["layer"], best["C"], best["cv_mean"],
                    best["cv_std"], len(items)])
    QC["tasks"]["T-KNOT-TYPE"] = {"n": len(items), "selected": best}


IV_BY_CHAIN = defaultdict(list)


def t_fs_region():
    X = load_emb(os.path.join(EMB, "fs_region"))
    for r in rd(os.path.join(ROOT, "data/interim/p303/fs_region_intervals.tsv")):
        IV_BY_CHAIN[r["name"]].append((int(r["start"]), int(r["end"])))
    y_by_chain = {}
    for ch, z in X.items():
        L = z["resid_layers"].shape[1]
        y = np.zeros(L, dtype=int)
        for s, e in IV_BY_CHAIN[ch]:
            if e > L:
                die(f"region {ch}: 区间 {s}-{e} 超出链长 {L}")
            y[s - 1:e] = 1
        y_by_chain[ch] = y
    fold_of_pair = {r["sample_id"]: r["dev_fold"] for r in MAN
                    if r["task_area"] == "fold_switch" and r["split"] == "development"}
    pair_of_chain = {r["name"]: r["pair_id"]
                     for r in rd(os.path.join(ROOT, "data/interim/p303/fs_region_intervals.tsv"))}
    usable = sorted({r["name"] for r in rd(os.path.join(ROOT, "data/interim/p303/fs_region_intervals.tsv"))
                     if r["label_decision"] == "usable"})
    fine_only = sorted(set(X) - set(usable))
    for tag, chains, fname in (("usable_main", usable, "T-FS-REGION"),
                               ("fine_only_sensitivity", fine_only, "T-FS-REGION.sensitivity")):
        chain_fold = {c: fold_of_pair.get(pair_of_chain[c], "NA") for c in chains}
        layers = X[chains[0]]["resid_layers"].shape[0]
        rows = []
        for layer in range(layers):
            for C in C_GRID:
                per_fold = {}
                for f in sorted(set(chain_fold.values())):
                    tr = [c for c in chains if chain_fold[c] != f]
                    te = [c for c in chains if chain_fold[c] == f]
                    if not tr or not te:
                        continue
                    Xtr = np.concatenate([X[c]["resid_layers"][layer] for c in tr])
                    ytr = np.concatenate([y_by_chain[c] for c in tr])
                    if len(set(ytr.tolist())) < 2:
                        per_fold[f] = {}
                        continue
                    clf = LogisticRegression(C=C, class_weight="balanced", solver="liblinear",
                                             max_iter=1000, random_state=SEED)
                    clf.fit(Xtr, ytr)
                    scores = {}
                    for c in te:
                        yte = y_by_chain[c]
                        if len(set(yte)) < 2:
                            scores[c] = None
                            continue
                        s = clf.predict_proba(X[c]["resid_layers"][layer])[:, 1]
                        scores[c] = round(float(average_precision_score(yte, s)), 6)
                    per_fold[f] = scores
                valid = [s for sc in per_fold.values() for s in sc.values() if s is not None]
                rows.append({"layer": layer + 1, "C": C, "metric": "residue_auprc",
                             "cv_mean": round(float(np.mean(valid)), 6) if valid else None,
                             "cv_std": round(float(np.std(valid)), 6) if valid else None,
                             "per_fold": json.dumps(per_fold, ensure_ascii=False)})
        write_selection(fname + ".selection.tsv", rows)
        best = pick(rows, "cv_mean")
        json.dump(best, open(os.path.join(OUTD, fname + ".selected.json"), "w"), indent=1)
        SUMMARY.append([f"T-FS-REGION:{tag}", "residue AUPRC", best["layer"], best["C"],
                        best["cv_mean"], best["cv_std"], len(chains)])
        QC["tasks"]["T-FS-REGION:" + tag] = {"chains": chains, "selected": best}


def t_fs_l1():
    X = load_emb(os.path.join(EMB, "fs_endpoints"))
    fs = [r for r in rd(os.path.join(ROOT, "data/curated/fold_switch_global.tsv"))
          if r["target"] == "1" and r["valid_mask"] == "1"]
    rows = []
    for r in fs:
        a = X[f"{r['pair_id']}_A"]
        # porter_87 两端观测序列相同（P3.01 断言）→ 共享同一条目；仅该对允许回退
        b_name = f"{r['pair_id']}_B"
        if b_name not in X:
            die(f"fold-switch 阳性端点缺失: {b_name}")
        b = X[b_name]
        for layer in range(a["mean_layers"].shape[0]):
            va, vb = a["mean_layers"][layer].astype(float), b["mean_layers"][layer].astype(float)
            cos = float(va @ vb / (np.linalg.norm(va) * np.linalg.norm(vb)))
            rows.append({"pair_id": r["pair_id"], "layer": layer + 1, "cosine": round(cos, 6)})
    write_selection("T-FS-L1-PAIRED.potential_cosine.tsv", rows)
    QC["tasks"]["T-FS-L1-PAIRED"] = {"n_pairs": len(fs), "descriptive": True}
    SUMMARY.append(["T-FS-L1-PAIRED", "描述性（逐层余弦，无主张）", "-", "-", "-", "-", len(fs)])
    return X, fs


def t_disorder():
    X = load_emb(os.path.join(EMB, "disorder_dev"))
    ridx = {r["name"]: [int(x) for x in r["resid_indices"].split(",")]
            for r in rd(os.path.join(ROOT, "data/interim/p303/disorder_resid_idx.tsv"))}
    state_by_res = defaultdict(dict)
    for r in rd(os.path.join(ROOT, "data/curated/disorder_masks.tsv")):
        if r["state"] in ("0", "1") and r["mask"] == "1":
            for p in range(int(r["start"]), int(r["end"]) + 1):
                state_by_res[r["disprot_id"]][p] = int(r["state"])
    avail = sorted(n for n in X if n in state_by_res and state_by_res[n])
    # 提取宇宙=build_p303_inputs 产出的全部（空域/全域超 trunc1022 跳过项见其 qc）
    if len(avail) != len(X):
        die(f"disorder 有标签但未入探针 {len(X) - len(avail)} 条")
    # 残基断言：掩码域 ∩ trunc1022
    ridx_all = {n: [i for i in ridx[n] if i <= 1022] for n in avail}
    dis_fold = {r["sample_id"].split(":", 1)[1]: r["dev_fold"] for r in MAN
                if r["task_area"] == "disorder" and r["split"] == "development"}
    layers = X[avail[0]]["resid_layers"].shape[0]
    names, pos, ys, folds = [], [], [], []
    for n in avail:
        for i, p in enumerate(ridx_all[n]):
            names.append(n)
            pos.append((n, i))
            ys.append(state_by_res[n][p])
            folds.append(dis_fold.get(n, "NA"))
    y_arr = np.array(ys, dtype=int)
    fold_arr = np.array(folds)
    name_arr = np.array(names, dtype=object)
    rows = []
    for layer in range(layers):
        Xl = np.stack([X[n]["resid_layers"][layer][i] for n, i in pos]).astype(np.float32)
        for C in C_GRID:
            ps = []
            for f in sorted(set(fold_arr)):
                tr = fold_arr != f
                te = fold_arr == f
                if y_arr[tr].sum() == 0 or y_arr[te].sum() == 0 or y_arr[te].sum() == te.sum():
                    continue
                clf = LogisticRegression(C=C, class_weight="balanced", solver="liblinear",
                                         max_iter=1000, random_state=SEED)
                clf.fit(Xl[tr], y_arr[tr])
                s = clf.predict_proba(Xl[te])[:, 1]
                for n in sorted(set(name_arr[te])):
                    m = name_arr[te] == n
                    if len(set(y_arr[te][m])) < 2:
                        continue
                    ps.append(average_precision_score(y_arr[te][m], s[m]))
            rows.append({"layer": layer + 1, "C": C, "metric": "residue_auprc_per_protein",
                         "cv_mean": round(float(np.mean(ps)), 6),
                         "cv_std": round(float(np.std(ps)), 6), "n_proteins": len(ps)})
            print(f"[disorder] layer={layer+1} C={C} mean={rows[-1]['cv_mean']}", flush=True)
    write_selection("T-DISORDER-RES.selection.tsv", rows)
    best = pick(rows, "cv_mean")
    json.dump(best, open(os.path.join(OUTD, "T-DISORDER-RES.selected.json"), "w"), indent=1)
    SUMMARY.append(["T-DISORDER-RES", "residue AUPRC", best["layer"], best["C"],
                    best["cv_mean"], best["cv_std"], len(avail)])
    QC["tasks"]["T-DISORDER-RES"] = {"n_proteins": len(avail), "n_residues": len(y_arr),
                                     "selected": best}


def t_fs_l2a(eX=None, fs=None):
    if eX is None:
        eX = load_emb(os.path.join(EMB, "fs_endpoints"))
        fs = [r for r in rd(os.path.join(ROOT, "data/curated/fold_switch_global.tsv"))
              if r["target"] == "1" and r["valid_mask"] == "1"]
    sX = load_emb(os.path.join(EMB, "l2_stage_a"))
    pos_vec = {}
    for r in fs:
        a = eX[f"{r['pair_id']}_A"]["mean_layers"].astype(np.float32)
        b_name = f"{r['pair_id']}_B"
        if b_name not in eX:
            die(f"fold-switch 阳性端点缺失: {b_name}")
        b = eX[b_name]["mean_layers"].astype(np.float32)
        pos_vec[r["pair_id"]] = (a + b) / 2
    names = sorted(sX)
    bg = np.stack([sX[n]["mean_layers"] for n in names]).astype(np.float32)
    rows = []
    for layer in range(bg.shape[1]):
        Xbg = bg[:, layer, :]
        Xp = np.stack([pos_vec[p][layer] for p in sorted(pos_vec)])
        X = np.concatenate([Xp, Xbg])
        y = np.array([1] * len(Xp) + [0] * len(Xbg))
        for C in C_GRID:
            clf = LogisticRegression(C=C, class_weight="balanced", solver="liblinear",
                                     max_iter=1000, random_state=SEED)
            clf.fit(X, y)
            sc = clf.predict_proba(X)[:, 1]
            order = np.argsort(-sc)
            ranks = np.empty(len(y), dtype=int)
            ranks[order] = np.arange(1, len(y) + 1)
            pr = ranks[:len(Xp)]
            ks = [10, 100, 1000, 10000]
            rec = {k: round(float((pr <= k).mean()), 6) for k in ks}
            enr = {k: round(float(rec[k] / (k / len(y))), 6) for k in ks}
            pct = round(float(np.mean(pr) / len(y)), 6)
            rows.append({"layer": layer + 1, "C": C, **{f"recall@{k}": rec[k] for k in ks},
                         **{f"enrichment@{k}": enr[k] for k in ks},
                         "positive_rank_percentile": pct})
            print(f"[l2A] layer={layer+1} C={C} pct={pct} recall@10000={rec[10000]}", flush=True)
    write_selection("T-FS-L2-PU-RANK.stageA.selection.tsv", rows)
    best = min(rows, key=lambda r: (r["positive_rank_percentile"], C_GRID.index(r["C"]), r["layer"]))
    json.dump(best, open(os.path.join(OUTD, "T-FS-L2-PU-RANK.stageA.selected.json"), "w"), indent=1)
    SUMMARY.append(["T-FS-L2-PU-RANK:stageA", "rank percentile(选择)", best["layer"], best["C"],
                    best["positive_rank_percentile"], "-", len(names) + 10])
    QC["tasks"]["T-FS-L2-PU-RANK"] = {"stageA_universe": len(names) + 10, "selected": best}


def main():
    eX = fs = None
    if want("KNOT_PRESENCE") and not os.path.exists(os.path.join(OUTD, "T-KNOT-PRESENCE.selected.json")):
        t_knot_presence()
        print("[probe] KNOT_PRESENCE done", flush=True)
    if want("KNOT_TYPE") and not os.path.exists(os.path.join(OUTD, "T-KNOT-TYPE.selected.json")):
        t_knot_type()
        print("[probe] KNOT_TYPE done", flush=True)
    if want("FS_REGION") and not os.path.exists(os.path.join(OUTD, "T-FS-REGION.selected.json")):
        t_fs_region()
        print("[probe] FS_REGION done", flush=True)
    if want("FS_L1") and not os.path.exists(os.path.join(OUTD, "T-FS-L1-PAIRED.potential_cosine.tsv")):
        eX, fs = t_fs_l1()
        print("[probe] FS_L1 done", flush=True)
    if want("DISORDER") and not os.path.exists(os.path.join(OUTD, "T-DISORDER-RES.selected.json")):
        t_disorder()
        print("[probe] DISORDER done", flush=True)
    if want("FS_L2A") and not os.path.exists(os.path.join(OUTD, "T-FS-L2-PU-RANK.stageA.selected.json")):
        t_fs_l2a(eX, fs)
        print("[probe] FS_L2A done", flush=True)
    # summary/qc 合并写（P303_ONLY 分跑不互相覆盖；task 列为主键）
    old_sum = {}
    sp = os.path.join(OUTD, "summary.tsv")
    if os.path.exists(sp):
        for row in csv.DictReader(open(sp), delimiter="\t"):
            old_sum[row["task"]] = [row[c] for c in ("task", "metric", "layer", "C", "cv_mean", "cv_std", "n")]
    for row in SUMMARY:
        old_sum[row[0]] = row
    with open(sp, "w", newline="") as f:
        w = csv.writer(f, delimiter="\t", lineterminator="\n")
        w.writerow(["task", "metric", "layer", "C", "cv_mean", "cv_std", "n"])
        w.writerows(old_sum.values())
    qp = os.path.join(OUTD, "qc.json")
    old_qc = {}
    if os.path.exists(qp):
        old_qc = json.load(open(qp))
    old_qc.setdefault("tasks", {}).update(QC.get("tasks", {}))
    for f in sorted(glob.glob(os.path.join(OUTD, "*.selected.json"))):
        name = os.path.basename(f).replace(".selected.json", "")
        old_qc.setdefault("tasks", {}).setdefault(name, {"selected": json.load(open(f))})
    old_qc["run_ts"] = RUN_TS
    old_qc["note"] = ("全部数字为 development 内 CV 点估计；bootstrap 未运行（后续任务依冻结统计参数执行）；"
                      "FS-L2 为 stage-A 选择宇宙内直推排序（乐观值，官方指标在 stage-B 全宇宙）。")
    json.dump(old_qc, open(qp, "w"), ensure_ascii=False, indent=1, sort_keys=True)
    print("[p303probe] DONE")


if __name__ == "__main__":
    main()
