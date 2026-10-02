#!/usr/bin/env python3
"""P4.02：逐残基与固定聚合比较（configs/aggregation.yaml frozen_2026-09-26_p401；claims C-AG1）。

两阶段 CLI（臂级断点续跑；每臂产物独立落盘，重跑自动跳过已完成臂）：
  run_p402_aggregation.py arms --task disorder --arms ID,WIN4,WIN16,WIN64,MEAN,LAST,ID_h23,MEAN_h23
  run_p402_aggregation.py arms --task knots   --arms POOLED,LAST,WIN4,WIN16,WIN64,MIL_mean,MIL_max,POOLED_h29,LAST_h29,MIL_mean_h29
  run_p402_aggregation.py compare             # 读臂产物→推断族+敏感性 bootstrap+qc 汇总

推断族（sig99fam，尾侧 0.005/3）：fixed_residue=3 比较（池化残基 AUPRC 差）；
fixed_global=3 比较（AUROC 差）。per-protein 双类 n=4=描述性不入族。
窗口/C 皆内部 CV 选优（tie-break 更小值），选择曝光登记。
"""
import argparse
import csv
import gzip
import json
import os
import sys
from collections import defaultdict
from datetime import datetime

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
EMB303 = os.path.join(ROOT, "data/interim/p303/emb")
OUTD = os.path.join(ROOT, "results/aggregation/fixed")
ARMD = os.path.join(OUTD, "arms")
C_GRID = [0.01, 0.1, 1, 10]
BOOT_SEED = 2026
NF = 3  # 推断族比较数
LAYER_ROW = {"disorder": {"h33": 2, "h23": 1}, "knots": {"h33": 2, "h29": 1}}
RUN_TS = datetime.now().strftime("%Y-%m-%d %H:%M")


def die(m):
    print(f"[FATAL] {m}", flush=True)
    sys.exit(1)


def rd(p, d="\t"):
    with open(p) as f:
        return list(csv.DictReader(f, delimiter=d))


def load_emb(manifest_path, npz_dir):
    man = rd(manifest_path)
    key2arr, out = {}, {}
    for row in man:
        if row["key"] not in key2arr:
            p = os.path.join(npz_dir, row["key"] + ".npz")
            if not os.path.exists(p):
                die(f"npz 缺失（禁止静默子集）: {p}")
            with np.load(p, allow_pickle=False) as z:
                key2arr[row["key"]] = {k: z[k] for k in z.files if k != "meta"}
        out[row["name"]] = key2arr[row["key"]]
    return out


def logreg(Xtr, ytr, Xte, C):
    clf = LogisticRegression(C=C, class_weight="balanced", solver="liblinear",
                             max_iter=1000, random_state=2026)
    clf.fit(Xtr, ytr)
    return clf.predict_proba(Xte)[:, 1]


def cv_global(ubf, feat_of, tag):
    y_of = {u[0]: int(u[2]) for fo in ubf for u in ubf[fo]}
    logs, by_C = [], {}
    for C in C_GRID:
        oof = {}
        for f in sorted(ubf):
            tr = [u for fo in sorted(ubf) if fo != f for u in ubf[fo]]
            te = list(ubf[f])
            if not tr or not te:
                continue
            ytr = np.array([int(u[2]) for u in tr])
            if len(set(ytr.tolist())) < 2:
                continue
            Xtr = np.concatenate([feat_of(u[1]) for u in tr])
            s = logreg(Xtr, ytr, np.concatenate([feat_of(u[1]) for u in te]), C)
            for u, si in zip(te, s):
                oof[u[0]] = float(si)
        if not oof:
            continue
        ys = np.array([y_of[u] for u in sorted(oof)])
        ss = np.array([oof[u] for u in sorted(oof)])
        m = roc_auc_score(ys, ss) if len(set(ys.tolist())) > 1 else float("nan")
        print(f"    [{tag}] C={C} pooled_auroc={m:.4f}", flush=True)
        logs.append({"C": C, "pooled_auroc": round(float(m), 6)})
        by_C[C] = (m, oof)
    valid = {c: v for c, v in by_C.items() if np.isfinite(v[0])}
    if not valid:
        die(f"{tag}: 全部 C 退化")
    best = max(sorted(valid), key=lambda c: valid[c][0])
    return best, valid[best][1], logs


def cv_residue(ubf, feat_of, tag):
    y_of = {u[0]: np.asarray(u[2]).ravel().astype(int) for fo in ubf for u in ubf[fo]}
    logs, by_C = [], {}
    for C in C_GRID:
        oof = {}
        for f in sorted(ubf):
            tr = [u for fo in sorted(ubf) if fo != f for u in ubf[fo]]
            te = list(ubf[f])
            if not tr or not te:
                continue
            ytr = np.concatenate([np.asarray(u[2]).ravel() for u in tr])
            if len(set(ytr.tolist())) < 2:
                continue
            Xtr = np.concatenate([feat_of(u[1]) for u in tr])
            s = logreg(Xtr, ytr, np.concatenate([feat_of(u[1]) for u in te]), C)
            off = 0
            for u in te:
                n_i = len(y_of[u[0]])
                oof[u[0]] = s[off:off + n_i]
                off += n_i
        if not oof:
            continue
        ys = np.concatenate([y_of[u] for u in sorted(oof)])
        ss = np.concatenate([oof[u] for u in sorted(oof)])
        m = average_precision_score(ys, ss) if len(set(ys.tolist())) > 1 else float("nan")
        print(f"    [{tag}] C={C} pooled_auprc={m:.4f}", flush=True)
        logs.append({"C": C, "pooled_auprc": round(float(m), 6)})
        by_C[C] = (m, oof)
    valid = {c: v for c, v in by_C.items() if np.isfinite(v[0])}
    if not valid:
        die(f"{tag}: 全部 C 退化")
    best = max(sorted(valid), key=lambda c: valid[c][0])
    flat = {}
    for p in sorted(valid[best][1]):
        s = valid[best][1][p]
        for i in range(len(y_of[p])):
            flat[f"{p}#{i}"] = (int(y_of[p][i]), float(s[i]))
    return best, flat, logs


def win_mean(pos, X, w):
    n = len(pos)
    out = np.empty(X.shape, dtype=np.float32)
    lo = 0
    for i in range(n):
        while pos[lo] < pos[i] - w:
            lo += 1
        hi = lo
        while hi < n and pos[hi] <= pos[i] + w:
            hi += 1
        out[i] = X[lo:hi].mean(axis=0)
    return out


def save_arm(tag, scores, meta):
    os.makedirs(ARMD, exist_ok=True)
    import io
    buf = io.StringIO()
    w = csv.writer(buf, delimiter="\t", lineterminator="\n")
    w.writerow(["unit", "y", "score"])
    for u, (y, s) in sorted(scores.items()):
        yy = int(y) if np.isscalar(y) or isinstance(y, int) else json.dumps(np.atleast_1d(y).tolist())
        w.writerow([u, yy, f"{s:.6g}"])
    with open(os.path.join(ARMD, f"{tag}.scores.tsv.gz"), "wb") as f:
        f.write(gzip.compress(buf.getvalue().encode(), mtime=0))
    json.dump(meta, open(os.path.join(ARMD, f"{tag}.meta.json"), "w"),
              ensure_ascii=False, indent=1, sort_keys=True)
    print(f"  [saved] {tag} ({len(scores)} units)", flush=True)


def load_arm(tag):
    scores = {}
    with gzip.open(os.path.join(ARMD, f"{tag}.scores.tsv.gz"), "rt") as f:
        for r in csv.DictReader(f, delimiter="\t"):
            yv = r["y"]
            y = json.loads(yv) if yv.startswith("[") else int(yv)
            scores[r["unit"]] = (y, float(r["score"]))
    meta = json.load(open(os.path.join(ARMD, f"{tag}.meta.json")))
    print(f"  [cache] {tag} ({len(scores)} units)", flush=True)
    return scores, meta


def arm_cached(tag):
    return os.path.exists(os.path.join(ARMD, f"{tag}.scores.tsv.gz"))


# ---------------------------------------------------------------- disorder arms
def run_disorder_arms(arms):
    X = load_emb(os.path.join(ROOT, "data/interim/p303/emb_disorder_dev/extract_manifest.tsv"),
                 os.path.join(EMB303, "disorder_dev"))
    if len(X) != 2279:
        die(f"disorder 宇宙断言失败 {len(X)}")
    ridx_all = {r["name"]: [int(x) for x in r["resid_indices"].split(",")]
                for r in rd(os.path.join(ROOT, "data/interim/p303/disorder_resid_idx.tsv"))}
    ridx = {n: [x for x in v if x <= 1022] for n, v in ridx_all.items()}
    state_by_res = defaultdict(dict)
    for r in rd(os.path.join(ROOT, "data/curated/disorder_masks.tsv")):
        if r["state"] in ("0", "1") and r["mask"] == "1":
            for p in range(int(r["start"]), int(r["end"]) + 1):
                state_by_res[r["disprot_id"]][p] = int(r["state"])
    man_rows = rd(os.path.join(ROOT, "data/splits/split_manifest.tsv"))
    dev_ids = {r["sample_id"] for r in man_rows if r["split"] == "development"}
    fold_of = {r["sample_id"]: r["dev_fold"] for r in man_rows if r["split"] == "development"}
    avail = sorted(n for n in X if n in state_by_res and n in ridx and state_by_res[n])
    if len(avail) != len(X):
        die(f"disorder 有标签但未入实验 {len(X) - len(avail)}")
    for n in avail:
        if f"disorder:{n}" not in dev_ids:
            die(f"非 dev 样本混入 {n}")
    prot = {}
    for n in avail:
        pos, yv = [], []
        for i, p in enumerate(ridx[n]):
            if p in state_by_res[n]:
                pos.append(i)
                yv.append(state_by_res[n][p])
        if yv:
            base_row = LAYER_ROW["disorder"]["h33"]
            prot[n] = {"pos": np.array(pos), "y": np.array(yv, dtype=int),
                       "fold": fold_of[f"disorder:{n}"],
                       "pos_ids": np.array([ridx[n][i] for i in pos])}
            prot[n]["X"] = {row: X[n]["resid_layers"][row][np.array(pos)].astype(np.float32)
                            for row in set(LAYER_ROW["disorder"].values())}
    dual = [n for n in avail if len(set(prot[n]["y"].tolist())) == 2]
    if len(dual) != 4:
        die(f"disorder 双类断言失败 {len(dual)}")
    y_prot = {n: int(prot[n]["y"].mean() >= 0.5) for n in prot}
    fold_lists = {f: [n for n in prot if prot[n]["fold"] == f]
                  for f in sorted({prot[n]["fold"] for n in prot})}
    # 短序列人工核对（一次性；naive 独立实现）
    marker = os.path.join(ARMD, ".handcheck_disorder.json")
    if not os.path.exists(marker):
        os.makedirs(ARMD, exist_ok=True)
        hand = []
        for n in [n for n in avail if len(prot[n]["y"]) <= 50][:3]:
            pos_ids, Xl = prot[n]["pos_ids"], prot[n]["X"][LAYER_ROW["disorder"]["h33"]]
            for w in (4, 16):
                ref = np.stack([Xl[[j for j, q in enumerate(pos_ids) if abs(q - p) <= w]].mean(axis=0)
                                for p in pos_ids])
                hand.append({"name": n, "check": f"win{w}",
                             "max_abs_diff": float(np.abs(ref - win_mean(pos_ids, Xl, w)).max())})
            hand.append({"name": n, "check": "mean", "max_abs_diff":
                         float(np.abs(Xl.mean(axis=0) - prot[n]["X"][2].mean(axis=0)).max())})
            hand.append({"name": n, "check": "last", "max_abs_diff":
                         float(np.abs(Xl[-1] - prot[n]["X"][2][-1]).max())})
        if max(h["max_abs_diff"] for h in hand) > 1e-4:
            die("短序列人工核对超差")
        json.dump({"detail": hand, "max_diff": max(h["max_abs_diff"] for h in hand)},
                  open(marker, "w"), indent=1)
        print("  短序列人工核对通过", flush=True)

    for arm in arms:
        if arm_cached(f"D-{arm}"):
            print(f"  [skip] D-{arm} 已有产物", flush=True)
            continue
        base, _, layer_sfx = arm.partition("_h")
        row = LAYER_ROW["disorder"].get(f"h{layer_sfx}", LAYER_ROW["disorder"]["h33"]) if layer_sfx else LAYER_ROW["disorder"]["h33"]
        if base in ("ID",) or base.startswith("WIN"):
            w = int(base[3:]) if base.startswith("WIN") else 0
            def feat(o, row=row, w=w):
                Xl = prot[o]["X"][row]
                return Xl if w == 0 else win_mean(prot[o]["pos_ids"], Xl, w)
            ubf = {f: [(n, n, prot[n]["y"], f) for n in fold_lists[f]] for f in fold_lists}
            best_C, flat, logs = cv_residue(ubf, feat, f"D-{arm}")
            scores = flat
        elif base in ("MEAN", "LAST"):
            def feat(o, row=row, base=base):
                Xl = prot[o]["X"][row]
                return Xl.mean(axis=0, keepdims=True) if base == "MEAN" else Xl[-1:]
            ubf = {f: [(n, n, y_prot[n], f) for n in fold_lists[f]] for f in fold_lists}
            best_C, oof, logs = cv_global(ubf, feat, f"D-{arm}")
            scores = {}
            for n, s in oof.items():
                for i in range(len(prot[n]["y"])):
                    scores[f"{n}#{i}"] = (int(prot[n]["y"][i]), s)
        else:
            die(f"未知 disorder 臂 {arm}")
        save_arm(f"D-{arm}", scores, {"arm": arm, "task": "disorder", "best_C": best_C,
                                      "grid": logs, "dual_n": len(dual),
                                      "y_prot_positive": sum(y_prot.values())})


# ---------------------------------------------------------------- knots arms
def knot_context():
    kdir = os.path.join(ROOT, "data/interim/p402/emb_knots_resid")
    kman = os.path.join(kdir, "extract_manifest.tsv")
    if not os.path.exists(kman):
        die("knots 残基包缺失（先跑抽取作业）")
    kX = load_emb(kman, kdir)
    kseq = {r["record_id"]: r for r in rd(os.path.join(ROOT, "data/curated/knots_sequences.tsv"))}
    man_rows = rd(os.path.join(ROOT, "data/splits/split_manifest.tsv"))
    dev_ids = {r["sample_id"] for r in man_rows if r["split"] == "development"}
    fold_of = {r["sample_id"]: r["dev_fold"] for r in man_rows if r["split"] == "development"}
    units = []
    for r in man_rows:
        if r["task_area"] == "knot" and r["split"] == "development":
            rec = r["sample_id"].split(":", 1)[1]
            if rec in kseq and kseq[rec]["presence_target"] in ("0", "1"):
                if r["sample_id"] not in kX:
                    die(f"knots 表示缺失（禁止静默子集）: {r['sample_id']}")
                units.append({"sid": r["sample_id"], "rec": rec,
                              "y": int(kseq[rec]["presence_target"]),
                              "fold": fold_of[r["sample_id"]]})
    if len(units) != 750:
        die(f"knots presence 宇宙断言失败 {len(units)}")
    # K-POOLED 一致性断言（≥200 抽样 vs knots_mean_legacy hidden33）
    marker = os.path.join(ARMD, ".pooled_consistency_knots.json")
    if not os.path.exists(marker):
        kmean_old = load_emb(os.path.join(EMB303, "knots_dev", "extract_manifest.tsv"),
                             os.path.join(EMB303, "knots_dev"))
        common = [u for u in units if u["sid"] in kmean_old][:200]
        if len(common) < 200:
            die(f"池化一致性抽样不足 {len(common)}/200（legacy 库可用交集过小）")
        maxd = max(float(np.abs(kmean_old[u["sid"]]["mean_layers"][5].astype(np.float32)
                                - kX[u["sid"]]["resid_layers"][2].astype(np.float32).mean(
                                    axis=0)).max()) for u in common)
        json.dump({"max_abs_diff": round(maxd, 6), "n_sampled": len(common)},
                  open(marker, "w"), indent=1)
        if maxd > 1e-2:
            die(f"K-POOLED 一致性断言失败 {maxd}")
        print(f"  K-POOLED 一致性断言通过（max|Δ|={maxd:.4f}, n={len(common)}）", flush=True)
    return kX, units


def run_knots_arms(arms):
    kX, units = knot_context()
    y_of = {u["sid"]: u["y"] for u in units}
    fold_by = {u["sid"]: u["fold"] for u in units}
    ubf = defaultdict(list)
    for u in units:
        ubf[u["fold"]].append((u["sid"], u["sid"], u["y"], u["fold"]))
    ubf = dict(ubf)

    for arm in arms:
        if arm_cached(f"K-{arm}"):
            print(f"  [skip] K-{arm} 已有产物", flush=True)
            continue
        base, _, layer_sfx = arm.partition("_h")
        row = LAYER_ROW["knots"].get(f"h{layer_sfx}", LAYER_ROW["knots"]["h33"]) if layer_sfx else LAYER_ROW["knots"]["h33"]
        cache = {}
        def feat(sid, base=base, row=row, cache=cache):
            key = (sid, base, row)
            if key not in cache:
                R = kX[sid]["resid_layers"][row].astype(np.float32)
                if base == "POOLED":
                    cache[key] = R.mean(axis=0, keepdims=True)
                elif base == "LAST":
                    cache[key] = R[-1:]
                elif base.startswith("WIN"):
                    cache[key] = win_mean(np.arange(1, R.shape[0] + 1), R, int(base[3:])).mean(
                        axis=0, keepdims=True)
                elif base == "RESID":
                    cache[key] = R
            return cache[key]
        if base.startswith("MIL"):
            reduction = "mean" if base.endswith("mean") else "max"
            # 冻结协议（P4.02-M1 整改）：每 C 跨折收集 OOF 蛋白分数→池化 AUROC 全局选 C
            per_C = {C: {} for C in C_GRID}
            logs = []
            n_train_res, pos_rate = [], []
            for f in sorted(ubf):
                tr = [u for fo in sorted(ubf) if fo != f for u in ubf[fo]]
                te = list(ubf[f])
                Xtr = np.concatenate([feat(u[0], base="RESID") for u in tr])
                ytr = np.concatenate([np.full(feat(u[0], base="RESID").shape[0], u[2]) for u in tr])
                Xte = np.concatenate([feat(u[0], base="RESID") for u in te])
                n_train_res.append(int(Xtr.shape[0]))
                pos_rate.append(round(float(ytr.mean()), 6))
                for C in C_GRID:
                    sc = logreg(Xtr, ytr, Xte, C)
                    off = 0
                    for u in te:
                        n_i = feat(u[0], base="RESID").shape[0]
                        seg = sc[off:off + n_i]
                        off += n_i
                        per_C[C][u[0]] = float(seg.mean() if reduction == "mean" else seg.max())
            sel = []
            for C in C_GRID:
                oof = per_C[C]
                ys = [y_of[u] for u in sorted(oof)]
                ss = [oof[u] for u in sorted(oof)]
                m = roc_auc_score(ys, ss) if len(set(ys)) > 1 else float("nan")
                print(f"    [K-{arm}] C={C} pooled_auroc={m:.4f}", flush=True)
                logs.append({"C": C, "pooled_auroc": round(float(m), 6)})
                sel.append((m, C))
            best_m, best_C = max(sorted(sel, key=lambda t: t[1]), key=lambda t: t[0])  # 按 C 升序后取首个最大值=平局取更小 C（复审修正，与 cv_global 一致；本轮无平局）
            mil = per_C[best_C]
            save_arm(f"K-{arm}", {u: (y_of[u], s) for u, s in mil.items()},
                     {"arm": arm, "task": "knots", "reduction": reduction, "best_C": best_C,
                      "grid": logs, "mil_train_residues_per_fold": n_train_res,
                      "mil_train_pos_rate_per_fold": pos_rate})
        elif base in ("POOLED", "LAST") or base.startswith("WIN"):
            best_C, oof, logs = cv_global(ubf, lambda o: feat(o), f"K-{arm}")
            save_arm(f"K-{arm}", {u: (y_of[u], s) for u, s in oof.items()},
                     {"arm": arm, "task": "knots", "best_C": best_C, "grid": logs})
        else:
            die(f"未知 knots 臂 {arm}")


# ---------------------------------------------------------------- compare
def boot_compare(map_ref, map_arm, kind, n_family=NF, B=2000):
    by_prot = defaultdict(list)
    for u in map_ref:
        by_prot[u.split("#")[0]].append(u)
    prots = sorted(by_prot)

    def metric(smap, plist):
        ys, ss = [], []
        for p in plist:
            for u in by_prot[p]:
                y, s = smap[u]
                if isinstance(y, list):
                    ys.extend(y)
                    ss.append(s)
                else:
                    ys.append(y)
                    ss.append(s)
        if len(set(ys)) < 2:
            return None
        f = roc_auc_score if kind == "auroc" else average_precision_score
        return float(f(np.array(ys), np.array(ss)))

    fa, fb = metric(map_ref, prots), metric(map_arm, prots)
    if fa is None or fb is None:
        die(f"boot 点估计退化 ref={fa} arm={fb}")
    rng = np.random.RandomState(BOOT_SEED)
    deltas, degen = [], 0
    for _ in range(B):
        pick = rng.choice(prots, size=len(prots), replace=True)
        va, vb = metric(map_ref, pick), metric(map_arm, pick)
        if va is None or vb is None:
            degen += 1
            continue
        deltas.append(vb - va)
    if len(deltas) < 1000:
        die(f"boot 有效抽样 {len(deltas)}<1000")
    deltas.sort()
    cut = 0.005 / n_family
    lo = deltas[int(np.floor(cut * len(deltas)))]
    hi = deltas[int(np.ceil((1 - cut) * len(deltas))) - 1]
    return {"delta_point": round(fb - fa, 6), "ref": round(fa, 6), "arm": round(fb, 6),
            "ci99fam": [round(lo, 6), round(hi, 6)],
            "sig99fam": bool((lo > 0 and hi > 0) or (lo < 0 and hi < 0)),
            "boot_valid": len(deltas), "boot_degenerate": degen, "n_units": len(prots)}


def per_protein_auprc(smap):
    by = defaultdict(lambda: ([], []))
    for u, (y, s) in smap.items():
        if isinstance(y, list):
            by[u.split("#")[0]][0].extend(y)
            by[u.split("#")[0]][1].append(s)
        else:
            by[u.split("#")[0]][0].append(y)
            by[u.split("#")[0]][1].append(s)
    return {p: float(average_precision_score(np.array(ys), np.array(ss)))
            for p, (ys, ss) in by.items() if len(set(ys)) == 2}


def compare():
    qc = {"run_ts": RUN_TS, "config": "configs/aggregation.yaml frozen_2026-09-26_p401",
          "families": {}, "sensitivity": {}, "selection": {}, "coverage": {}}
    def get(tag):
        return load_arm(tag)
    # disorder
    d = {a: get(f"D-{a}") for a in ["ID", "MEAN", "LAST", "WIN4", "WIN16", "WIN64", "ID_h23", "MEAN_h23"]}
    # 窗口选择：按各窗口 best_C 行的池化 AUPRC 取最大（tie-break 更小 w）
    sel_metric = {w: max(x["pooled_auprc"] for x in d[f"WIN{w}"][1]["grid"]) for w in (4, 16, 64)}
    win_sel_d = min((w for w in (4, 16, 64) if sel_metric[w] == max(sel_metric.values())), key=lambda w: w)
    qc["selection"]["disorder_win"] = {"metric": sel_metric, "selected": win_sel_d}
    bootD = None
    ref = d["ID"][0]
    fam = []
    for arm in ["MEAN", "LAST", f"WIN{win_sel_d}"]:
        c = boot_compare(ref, d[arm][0], "auprc", NF)
        pp_ref, pp_arm = per_protein_auprc(ref), per_protein_auprc(d[arm][0])
        common = sorted(set(pp_ref) & set(pp_arm))
        dpp = [pp_arm[p] - pp_ref[p] for p in common]
        fam.append({"comparison": f"D-{arm} vs D-ID", "pooled_auprc": c,
                    "descriptive_per_protein_dual": {"n_common": len(common),
                                                     "delta_mean": round(float(np.mean(dpp)), 6),
                                                     "note": "双类 n=4 描述性不入族"}})
        print(f"  D-{arm}: Δ={c['delta_point']} CI99fam={c['ci99fam']}", flush=True)
    qc["families"]["aggregation_fixed_residue"] = fam
    for w in (4, 16, 64):
        if w != win_sel_d:
            qc["sensitivity"][f"D-WIN{w}_vs_D-ID"] = boot_compare(ref, d[f"WIN{w}"][0], "auprc", NF)
    qc["sensitivity"]["disorder_h23_ID_vs_ID_h33"] = boot_compare(ref, d["ID_h23"][0], "auprc", NF)
    qc["sensitivity"]["disorder_h23_MEAN_vs_MEAN_h33"] = boot_compare(d["MEAN"][0], d["MEAN_h23"][0], "auprc", NF)
    # knots
    k = {a: get(f"K-{a}") for a in ["POOLED", "LAST", "WIN4", "WIN16", "WIN64",
                                    "MIL_mean", "MIL_max", "POOLED_h29", "LAST_h29", "MIL_mean_h29"]}
    sel_metric_k = {w: max(x["pooled_auroc"] for x in k[f"WIN{w}"][1]["grid"]) for w in (4, 16, 64)}
    win_sel_k = min((w for w in (4, 16, 64) if sel_metric_k[w] == max(sel_metric_k.values())), key=lambda w: w)
    qc["selection"]["knots_win"] = {"metric": sel_metric_k, "selected": win_sel_k}
    refk = k["POOLED"][0]
    fam_g = []
    for arm in ["MIL_mean", "LAST", f"WIN{win_sel_k}"]:
        c = boot_compare(refk, k[arm][0], "auroc", NF)
        fam_g.append({"comparison": f"K-{arm} vs K-POOLED", "auroc": c})
        print(f"  K-{arm}: Δ={c['delta_point']} CI99fam={c['ci99fam']}", flush=True)
    qc["families"]["aggregation_fixed_global"] = fam_g
    for w in (4, 16, 64):
        if w != win_sel_k:
            qc["sensitivity"][f"K-WIN{w}_vs_K-POOLED"] = boot_compare(refk, k[f"WIN{w}"][0], "auroc", NF)
    qc["sensitivity"]["K-MIL_max_vs_K-POOLED"] = boot_compare(refk, k["MIL_max"][0], "auroc", NF)
    for a in ["POOLED_h29", "LAST_h29", "MIL_mean_h29"]:
        qc["sensitivity"][f"K-{a}_vs_K-POOLED_h33"] = boot_compare(refk, k[a][0], "auroc", NF)
    # KNOT-TYPE 描述性矩阵（引用 P3.03 registered 结果；本实验不重跑 multiclass 头）
    qc["coverage"]["disorder_units"] = len(ref)
    qc["coverage"]["knots_proteins"] = len(refk)
    qc["coverage"]["knot_type_note"] = "T-KNOT-TYPE 同矩阵为描述性（config §2 secondary）；Macro-F1 见 P3.03 registered 结果"
    json.dump(qc, open(os.path.join(OUTD, "aggregation_fixed_qc.json"), "w"),
              ensure_ascii=False, indent=1, sort_keys=True)
    print("[p402-compare] DONE", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["arms", "compare"])
    ap.add_argument("--task", choices=["disorder", "knots"], default="")
    ap.add_argument("--arms", default="")
    a = ap.parse_args()
    if a.stage == "arms":
        os.makedirs(ARMD, exist_ok=True)
        arms = [x for x in a.arms.split(",") if x]
        if a.task == "disorder":
            run_disorder_arms(arms)
        else:
            run_knots_arms(arms)
    else:
        compare()


if __name__ == "__main__":
    main()
