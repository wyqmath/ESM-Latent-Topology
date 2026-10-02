#!/usr/bin/env python3
"""P4.03：参数受控的可学习聚合（configs/aggregation.yaml frozen_2026-09-26_p401 §4；claims C-AG2）。

双臂：L-ATT-KNOTS（knots 750，y=presence，vs K-POOLED）；
      L-ATT-DISORDER（disorder 2,279 全量，y=y_prot，vs D-MEAN）。
聚合器=加性注意力池化（W_a 1280×64 + b_a 64 + w_a 64 = 82,048 参数，count_parameters 对账）。
训练：外层=dev_fold 5 折组级 OOF；内层=训练折内纯随机 90/10 早停（patience 10，≤100 epochs，
Adam lr1e-3 batch256 蛋白）；头=冻结聚合器输出 z 上的 sklearn logistic（C 网格 OOF 选优，
与固定臂同协议）。推断族 aggregation_learned（2 比较）：Δ=seed-mean OOF 池化 metric 差，
配对 bootstrap 按蛋白，尾侧 0.005/2（sig99fam）；三种子逐个并列。
归因规则：容量对照=P3.04 MLP-on-pooled；disorder 侧宇宙错配降为方向性参考（config §4）。
"""
import csv
import gzip
import json
import os
import sys
from collections import defaultdict
from datetime import datetime

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
EMB303 = os.path.join(ROOT, "data/interim/p303/emb")
OUTD = os.path.join(ROOT, "results/aggregation/learned")
C_GRID = [0.01, 0.1, 1, 10]
SEEDS = [13, 42, 2026]
B_BOOT = 2000
BOOT_SEED = 2026
BATCH = 256
EPOCHS = 100
PATIENCE = 10
LR = 1e-3
PARAM_EXPECT = 82048
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


class AttentionPool(torch.nn.Module):
    """e_i = w_a^T tanh(W_a h_i + b_a)；α=softmax over valid；z=Σ α_i h_i。82,048 参数。"""

    def __init__(self, d=1280, h=64):
        super().__init__()
        self.W_a = torch.nn.Linear(d, h)
        self.w_a = torch.nn.Linear(h, 1, bias=False)

    def forward(self, X, mask):
        e = self.w_a(torch.tanh(self.W_a(X))).squeeze(-1)      # (B, L)
        e = e.masked_fill(~mask, -1e9)
        a = torch.softmax(e, dim=1)                             # (B, L)
        return (a.unsqueeze(-1) * X).sum(dim=1)                 # (B, D)


def count_params(m):
    return sum(p.numel() for p in m.parameters())


def train_attention(Xs, ys, seed):
    """Xs: list of (n_i, D) fp32；ys: list of int。训练折内 90/10 随机早停。
    返回训练好的 AttentionPool（含辅助线性输出训练，输出层不入参数账）。"""
    torch.manual_seed(seed)
    np.random.seed(seed)
    agg = AttentionPool()
    aux = torch.nn.Linear(1280, 1)  # 辅助输出（训练用；参数账只记聚合器）
    n_params = count_params(agg)
    if n_params != PARAM_EXPECT:
        die(f"参数对账失败 {n_params} != {PARAM_EXPECT}")
    opt = torch.optim.Adam(list(agg.parameters()) + list(aux.parameters()), lr=LR)
    lossf = torch.nn.BCEWithLogitsLoss()
    idx = np.random.RandomState(seed).permutation(len(Xs))
    n_val = max(2, int(0.1 * len(Xs)))
    vi, ti = idx[:n_val], idx[n_val:]
    best, best_state, patience = -1.0, None, 0
    for epoch in range(EPOCHS):
        agg.train()
        perm = np.random.RandomState(seed + epoch + 1).permutation(len(ti))
        for b in range(0, len(ti), BATCH):
            sel = [int(x) for x in ti[b:b + BATCH]]
            X, m = _pack([Xs[i] for i in sel])
            y = torch.tensor([ys[i] for i in sel], dtype=torch.float32)
            opt.zero_grad()
            z = agg(X, m)
            loss = lossf(aux(z).squeeze(-1), y)
            loss.backward()
            opt.step()
        if len(vi) and len(set(np.array(ys)[vi].tolist())) > 1:
            agg.eval()
            with torch.no_grad():
                Xv, mv = _pack([Xs[i] for i in vi])
                zv = agg(Xv, mv)
                sv = torch.sigmoid(aux(zv).squeeze(-1)).numpy()
            sc = roc_auc_score(np.array(ys)[vi], sv)
            if sc > best:
                best, patience = sc, 0
                best_state = {k: v.clone() for k, v in agg.state_dict().items()}
            else:
                patience += 1
                if patience >= PATIENCE:
                    break
    if best_state is not None:
        agg.load_state_dict(best_state)
    agg.eval()
    return agg, {"n_params": n_params, "best_val": round(float(best), 6), "epochs": epoch + 1}


def _pack(lst):
    L = max(x.shape[0] for x in lst)
    X = np.zeros((len(lst), L, lst[0].shape[1]), dtype=np.float32)
    M = np.zeros((len(lst), L), dtype=bool)
    for i, x in enumerate(lst):
        X[i, :x.shape[0]] = x
        M[i, :x.shape[0]] = True
    return torch.tensor(X), torch.tensor(M)


def agg_scores(agg, Xs, bs=512):
    out = []
    with torch.no_grad():
        for b in range(0, len(Xs), bs):
            X, m = _pack(Xs[b:b + bs])
            out.append(agg(X, m).numpy())
    return np.concatenate(out)


def logreg_head(z_tr, y_tr, z_te, C):
    clf = LogisticRegression(C=C, class_weight="balanced", solver="liblinear",
                             max_iter=1000, random_state=2026)
    clf.fit(z_tr, y_tr)
    return clf.predict_proba(z_te)[:, 1]


def main():
    os.makedirs(OUTD, exist_ok=True)
    man_rows = rd(os.path.join(ROOT, "data/splits/split_manifest.tsv"))
    dev_ids = {r["sample_id"] for r in man_rows if r["split"] == "development"}
    fold_of = {r["sample_id"]: r["dev_fold"] for r in man_rows if r["split"] == "development"}
    qc = {"run_ts": RUN_TS, "config": "configs/aggregation.yaml frozen_2026-09-26_p401",
          "asserts": {}, "accounting": [], "families": {}, "sensitivity": {}}


    # ---------- L-ATT-KNOTS ----------
    print("== L-ATT-KNOTS ==", flush=True)
    kdir = os.path.join(ROOT, "data/interim/p402/emb_knots_resid")
    if not os.path.exists(os.path.join(kdir, "extract_manifest.tsv")):
        die("knots 残基包缺失（先跑 P4.02 前置抽取）")
    kX = load_emb(os.path.join(kdir, "extract_manifest.tsv"), kdir)
    kseq = {r["record_id"]: r for r in rd(os.path.join(ROOT, "data/curated/knots_sequences.tsv"))}
    units = []
    for r in man_rows:
        if r["task_area"] == "knot" and r["split"] == "development":
            rec = r["sample_id"].split(":", 1)[1]
            if rec in kseq and kseq[rec]["presence_target"] in ("0", "1"):
                if r["sample_id"] not in kX:
                    die(f"knots 表示缺失 {r['sample_id']}")
                units.append({"sid": r["sample_id"], "y": int(kseq[rec]["presence_target"]),
                              "fold": fold_of[r["sample_id"]],
                              "X": kX[r["sample_id"]]["resid_layers"][2].astype(np.float32)})
    qc["asserts"]["knots_n"] = len(units)
    if len(units) != 750:
        die(f"knots 断言失败 {len(units)}")
    fold_by_sid = {u["sid"]: u["fold"] for u in units}
    results = {}
    for seed in SEEDS:
        cached = _load_ckpt("knots", seed)
        if cached is not None:
            results[seed] = {"z": cached, "cached": True}
            print(f"  [ckpt] L-ATT-KNOTS seed{seed} 已缓存", flush=True)
            continue
        z = {}
        acc = []
        for f in sorted({u["fold"] for u in units}):
            tr = [u for u in units if u["fold"] != f]
            te = [u for u in units if u["fold"] == f]
            agg, info = train_attention([u["X"] for u in tr], [u["y"] for u in tr], seed)
            acc.append({"fold": f, **info})
            zt = agg_scores(agg, [u["X"] for u in te])
            for u, zi in zip(te, zt):
                z[u["sid"]] = (u["y"], zi)
        results[seed] = {"z": z, "accounting": acc}
        _save_ckpt("knots", seed, {u: [int(y), [float(x) for x in s]]
                                   for u, (y, s) in z.items()})
        qc["accounting"].append({"arm": "L-ATT-KNOTS", "seed": seed, "folds": acc})
    # 头 C 选优 + 池化 OOF AUROC（与固定臂同协议）
    knot_arm = {}
    for seed in SEEDS:
        z = results[seed]["z"]
        ids = sorted(z)
        yv = np.array([z[i][0] for i in ids])
        Z = np.stack([z[i][1] for i in ids])
        fold_arr = np.array([fold_by_sid[i] for i in ids])
        best = None
        for C in C_GRID:
            oof = np.zeros(len(ids))
            for f in sorted(set(fold_arr.tolist())):
                tr_m, te_m = fold_arr != f, fold_arr == f
                oof[te_m] = logreg_head(Z[tr_m], yv[tr_m], Z[te_m], C)
            m = roc_auc_score(yv, oof)
            if best is None or m > best[0]:
                best = (m, C, oof.copy())
        knot_arm[seed] = {"C": best[1], "pooled_auroc": round(best[0], 6),
                          "scores": {i: (int(yv[j]), float(best[2][j])) for j, i in enumerate(ids)}}
        print(f"  seed{seed}: C={best[1]} pooled={best[0]:.4f}", flush=True)

    # ---------- L-ATT-DISORDER ----------
    print("== L-ATT-DISORDER ==", flush=True)
    X = load_emb(os.path.join(ROOT, "data/interim/p303/emb_disorder_dev/extract_manifest.tsv"),
                 os.path.join(EMB303, "disorder_dev"))
    ridx_all = {r["name"]: [int(x) for x in r["resid_indices"].split(",")]
                for r in rd(os.path.join(ROOT, "data/interim/p303/disorder_resid_idx.tsv"))}
    ridx = {n: [x for x in v if x <= 1022] for n, v in ridx_all.items()}
    state_by_res = defaultdict(dict)
    for r in rd(os.path.join(ROOT, "data/curated/disorder_masks.tsv")):
        if r["state"] in ("0", "1") and r["mask"] == "1":
            for p in range(int(r["start"]), int(r["end"]) + 1):
                state_by_res[r["disprot_id"]][p] = int(r["state"])
    avail = sorted(n for n in X if n in state_by_res and n in ridx and state_by_res[n])
    if len(avail) != 2279:
        die(f"disorder 断言失败 {len(avail)}")
    dunits = []
    for n in avail:
        pos = [i for i, p in enumerate(ridx[n]) if p in state_by_res[n]]
        yv = np.array([state_by_res[n][ridx[n][i]] for i in pos], dtype=int)
        if not len(yv):
            continue
        dunits.append({"name": n, "fold": fold_of[f"disorder:{n}"],
                       "y_prot": int(yv.mean() >= 0.5),
                       "y": yv, "n": len(yv),
                       "X": X[n]["resid_layers"][2][np.array(pos)].astype(np.float32)})
    qc["asserts"]["disorder_n"] = len(dunits)
    dis_results = {}
    for seed in SEEDS:
        cached = _load_ckpt("disorder", seed)
        if cached is not None:
            dis_results[seed] = {"z": cached, "cached": True}
            print(f"  [ckpt] L-ATT-DISORDER seed{seed} 已缓存", flush=True)
            continue
        z = {}
        acc = []
        for f in sorted({u["fold"] for u in dunits}):
            tr = [u for u in dunits if u["fold"] != f]
            te = [u for u in dunits if u["fold"] == f]
            agg, info = train_attention([u["X"] for u in tr], [u["y_prot"] for u in tr], seed)
            acc.append({"fold": f, **info})
            zt = agg_scores(agg, [u["X"] for u in te])
            for u, zi in zip(te, zt):
                z[u["name"]] = (u["y"], u["y_prot"], zi)
        dis_results[seed] = {"z": z, "accounting": acc}
        _save_ckpt("disorder", seed,
                   {n: [[int(x) for x in yv], int(yp), [float(t) for t in zv]]
                    for n, (yv, yp, zv) in
                    ((n, (v[0], v[1], v[2])) for n, v in z.items())})
        qc["accounting"].append({"arm": "L-ATT-DISORDER", "seed": seed, "folds": acc})
    dis_arm = {}
    for seed in SEEDS:
        z = dis_results[seed]["z"]
        names = sorted(z)
        yprot = np.array([z[n][1] for n in names])
        Z = np.stack([z[n][2] for n in names])
        fold_arr = np.array([{u["name"]: u["fold"] for u in dunits}[n] for n in names])
        best = None
        for C in C_GRID:
            oof = np.zeros(len(names))
            for f in sorted(set(fold_arr.tolist())):
                tr_m, te_m = fold_arr != f, fold_arr == f
                oof[te_m] = logreg_head(Z[tr_m], yprot[tr_m], Z[te_m], C)
            m = roc_auc_score(yprot, oof)
            if best is None or m > best[0]:
                best = (m, C, oof.copy())
        # 残基级键空间（与 fixed 产物一致：name#i → 残基 y + 广播分数）
        scores = {}
        for j, n in enumerate(names):
            yv = z[n][0]
            for i in range(len(yv)):
                scores[f"{n}#{i}"] = (int(yv[i]), float(best[2][j]))
        dis_arm[seed] = {"C": best[1], "pooled_auroc_yprot": round(best[0], 6),
                         "scores": scores}
        print(f"  disorder seed{seed}: C={best[1]} yprot-AUROC={best[0]:.4f}", flush=True)

    # ---------- 推断：seed-mean Δ + 配对 bootstrap（族 aggregation_learned，尾侧 0.005/2） ----------
    # knots 对照 = P4.02 K-POOLED OOF；disorder 对照 = P4.02 D-MEAN OOF（读 fixed 产物）
    fixed_dir = os.path.join(os.path.dirname(OUTD), "fixed", "arms")
    ref_k = _read_arm_scores(os.path.join(fixed_dir, "K-POOLED.scores.tsv.gz"))
    ref_d = _read_arm_scores(os.path.join(fixed_dir, "D-MEAN.scores.tsv.gz"))
    NF = 2
    cmps = {}
    for label, arm_maps, ref_map, kind in [
            ("L-ATT-KNOTS vs K-POOLED", {s: {u: v for u, v in knot_arm[s]["scores"].items()}
                                         for s in SEEDS}, ref_k, "auroc"),
            ("L-ATT-DISORDER vs D-MEAN", {s: {u: v for u, v in dis_arm[s]["scores"].items()}
                                          for s in SEEDS}, ref_d, "auprc")]:
        seed_stats = []
        for s in SEEDS:
            seed_stats.append(_boot_pair(ref_map, arm_maps[s], kind, 0.0))
        # seed-mean 判定量：各蛋白分数取 seed 均值后配对
        mean_map = {}
        ids = sorted(arm_maps[SEEDS[0]])
        for u in ids:
            y = arm_maps[SEEDS[0]][u][0]
            mean_map[u] = (y, float(np.mean([arm_maps[s][u][1] for s in SEEDS])))
        judged = _boot_pair(ref_map, mean_map, kind, 0.005 / NF)
        cmps[label] = {"seedwise": seed_stats, "seed_mean_judged": judged}
        print(f"  {label}: seed-mean Δ={judged['delta']} CI99fam={judged['ci99fam']}", flush=True)
    qc["families"]["aggregation_learned"] = cmps

    # ---------- 落盘 ----------
    for seed in SEEDS:
        for tag, m in [("knots", knot_arm[seed]["scores"]),
                       ("disorder", dis_arm[seed]["scores"])]:
            import io
            buf = io.StringIO()
            w = csv.writer(buf, delimiter="\t", lineterminator="\n")
            w.writerow(["unit", "y", "score"])
            for u, (y, s) in sorted(m.items()):
                w.writerow([u, np.atleast_1d(y)[0] if isinstance(y, np.ndarray) else y,
                            f"{s:.6g}"])
            with open(os.path.join(OUTD, f"{tag}_latt_seed{seed}.tsv.gz"), "wb") as f:
                f.write(gzip.compress(buf.getvalue().encode(), mtime=0))
    json.dump(qc, open(os.path.join(OUTD, "aggregation_learned_qc.json"), "w"),
              ensure_ascii=False, indent=1, sort_keys=True)
    print("[p403] DONE", flush=True)


def _read_arm_scores(path):
    """读 fixed/arms/{TAG}.scores.tsv.gz（unit/y/score 三列）。"""
    with gzip.open(path, "rt") as f:
        return {r["unit"]: (int(r["y"]), float(r["score"]))
                for r in csv.DictReader(f, delimiter="\t")}


def _ckpt_path(kind, seed):
    return os.path.join(OUTD, f"ckpt_{kind}_seed{seed}.json.gz")


def _save_ckpt(kind, seed, obj):
    os.makedirs(OUTD, exist_ok=True)
    with open(_ckpt_path(kind, seed), "wb") as f:
        f.write(gzip.compress(json.dumps(obj).encode(), mtime=0))


def _load_ckpt(kind, seed):
    p = _ckpt_path(kind, seed)
    if not os.path.exists(p):
        return None
    with gzip.open(p, "rt") as f:
        return {u: tuple(v) for u, v in json.load(f).items()}


def _boot_pair(ref_map, arm_map, kind, cut, B=B_BOOT):
    by_prot = defaultdict(list)
    for u in ref_map:
        by_prot[u.split("#")[0]].append(u)
    prots = sorted(by_prot)

    def metric(smap, plist):
        ys, ss = [], []
        for p in plist:
            for u in by_prot[p]:
                y, s = smap[u]
                if isinstance(s, np.ndarray):
                    ys.extend(np.atleast_1d(y).tolist())
                    ss.extend(np.asarray(s).ravel().tolist())
                else:
                    ys.append(y)
                    ss.append(s)
        if len(set(ys)) < 2:
            return None
        f = roc_auc_score if kind == "auroc" else average_precision_score
        return float(f(np.array(ys), np.array(ss)))

    fa, fb = metric(ref_map, prots), metric(arm_map, prots)
    if fa is None or fb is None:
        return {"delta": None, "note": "退化"}
    rng = np.random.RandomState(BOOT_SEED)
    deltas, degen = [], 0
    for _ in range(B):
        pick = rng.choice(prots, size=len(prots), replace=True)
        va, vb = metric(ref_map, pick), metric(arm_map, pick)
        if va is None or vb is None:
            degen += 1
            continue
        deltas.append(vb - va)
    deltas.sort()
    if cut > 0:
        lo = deltas[int(np.floor(cut * len(deltas)))]
        hi = deltas[int(np.ceil((1 - cut) * len(deltas))) - 1]
    else:
        lo = hi = None
    return {"delta": round(fb - fa, 6), "ref": round(fa, 6), "arm": round(fb, 6),
            "ci99fam": [round(lo, 6), round(hi, 6)] if lo is not None else None,
            "boot_valid": len(deltas), "boot_degenerate": degen}


if __name__ == "__main__":
    main()
