#!/usr/bin/env python3
"""P3.04：受控非线性读取器容量对照（BN4；configs/probes.yaml readers.nonlinear 冻结语义）。v4。

统计核心（两轮审核整改后定稿）：
  Δ = 两臂各自 pooled metric 之差（Δmetric）的配对 bootstrap——按独立单位组（蛋白/链）重采样
  B=2000 种子 2026；退化抽样（重采样内单类→metric 未定义）弃除并计数 boot_degenerate；
  95% 与 99%（Bonferroni 族水平，capacity_gain 族 5 比较）双 CI；bootstrap-CI 框架无 p 值，
  不适用 Holm（probes.yaml 冻结 Holm 文本的偏离已在 P3.04 报告登记）。不使用单位级分数差
  （校准敏感伪像，一审 BLOCKER 已废）。
宇宙（新版划分 68,875 组，全部 development）：
  KNOT_PRESENCE dev usable 750（AUROC）；KNOT_TYPE dev 71 可评价（3_1 vs other 二值化语义，AUROC）；
  FS_REGION usable 6 双类链（池化残基 AUPRC；敏感性 10 链不跑非线性——冻结明文）；
  DISORDER500 dev 2,279 内双类保底抽样 500（池化残基 AUPRC；500 名单落盘）；
  FS_L2A stage-A 宇宙 10,010（2-半交叉拟合 OOF ΔAUROC；直配乐观 caveat 随行）。
  T-FS-L1 无读取器目标→不适用（P3.03 登记随行）。
"""
import csv
import json
import os
import sys
from collections import defaultdict
from datetime import datetime

import numpy as np
import torch
import yaml
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
EMB = os.path.join(ROOT, "data/interim/p303/emb")
PROB = os.path.join(ROOT, "results/probes")
OUTD = os.path.join(PROB, "capacity")
SEEDS = [13, 42, 2026]
C_GRID = [0.01, 0.1, 1, 10]
B_BOOT = 2000
BOOT_SEED = 2026
RUN_TS = datetime.now().strftime("%Y-%m-%d %H:%M")
torch.set_num_threads(max(1, (os.cpu_count() or 8) // 2))


def die(m):
    print(f"[p304 FATAL] {m}", file=sys.stderr)
    sys.exit(1)


def rd(p, d="\t"):
    with open(p, newline="") as f:
        return list(csv.DictReader(f, delimiter=d))


def load_emb(task_dir):
    man = rd(os.path.join(task_dir, "extract_manifest.tsv"))
    key2arr = {}
    for row in man:
        if row["key"] not in key2arr:
            with np.load(os.path.join(task_dir, row["key"] + ".npz"), allow_pickle=False) as z:
                key2arr[row["key"]] = {k: z[k] for k in z.files if k != "meta"}
    return {row["name"]: key2arr[row["key"]] for row in man}


def sel_layer(fname):
    return json.load(open(os.path.join(PROB, fname)))["layer"] - 1


def metric_fn(kind):
    return roc_auc_score if kind == "auroc" else average_precision_score


def pooled(kind, pairs):
    ys = [p[1] for p in pairs]
    if len(set(ys)) < 2:
        return None  # 退化：调用方弃除该轮
    return metric_fn(kind)(ys, [p[2] for p in pairs])


class MLP(torch.nn.Module):
    def __init__(self, din, hidden, depth, dout):
        super().__init__()
        layers, d = [], din
        for _ in range(depth):
            layers += [torch.nn.Linear(d, hidden), torch.nn.ReLU(), torch.nn.Dropout(0.1)]
            d = hidden
        layers.append(torch.nn.Linear(d, dout))
        self.net = torch.nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def fit_mlp(Xtr, ytr, hidden, depth, seed, binary):
    torch.manual_seed(seed)
    np.random.seed(seed)
    classes = sorted(set(ytr))
    c2i = {c: i for i, c in enumerate(classes)}
    y = np.array([c2i[v] for v in ytr])
    rng = np.random.RandomState(seed)
    idx = rng.permutation(len(y))
    n_val = max(2, int(0.1 * len(y)))
    vi, ti = idx[:n_val], idx[n_val:]
    if len(set(y[vi].tolist())) < 2:
        vi = np.array([], dtype=int)
    model = MLP(Xtr.shape[1], hidden, depth, 1 if binary else len(classes))
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    lossf = torch.nn.BCEWithLogitsLoss() if binary else torch.nn.CrossEntropyLoss()
    Xt = torch.tensor(Xtr, dtype=torch.float32)
    yt = torch.tensor(y, dtype=torch.float32 if binary else torch.long)
    best, best_state, patience = -1.0, None, 0
    for epoch in range(100):
        model.train()
        g = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(Xt[ti], yt[ti]),
                                        batch_size=256, shuffle=True)
        for xb, yb in g:
            opt.zero_grad()
            loss = lossf(model(xb).squeeze(-1), yb)
            loss.backward()
            opt.step()
        if len(vi):
            model.eval()
            with torch.no_grad():
                ov = model(torch.tensor(Xtr[vi], dtype=torch.float32))
            if binary:
                sc = roc_auc_score(y[vi], torch.sigmoid(ov.squeeze(-1)).numpy())
            else:
                sc = f1_score(y[vi], ov.argmax(1).numpy(), average="macro",
                              labels=sorted(set(y[vi].tolist())))
            if sc > best:
                best, best_state, patience = sc, {k: v.clone() for k, v in model.state_dict().items()}, 0
            else:
                patience += 1
                if patience >= 10:
                    break
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    return model


def mlp_scores(model, X, binary):
    with torch.no_grad():
        out = model(torch.tensor(X, dtype=torch.float32))
    if binary:
        return torch.sigmoid(out.squeeze(-1)).numpy()
    return torch.softmax(out, 1)[:, -1].numpy()


def run_universe(tag, units, kind, binary):
    """折基宇宙：units=[(name, X(1,D) 全局|(n,D) 残基, y 标量|向量, fold)]。"""
    is_residue = isinstance(units[0][2], (np.ndarray, list))
    fs = sorted({u[3] for u in units})

    def rows_of(us):
        X = np.concatenate([u[1] for u in us])
        if is_residue:
            y = np.concatenate([np.asarray(u[2]).ravel() for u in us])
        else:
            y = np.array([u[2] for u in us])
        return X, y

    def iter_units(us):
        if is_residue:
            for u in us:
                yv = np.asarray(u[2]).ravel()
                for i in range(len(yv)):
                    yield f"{u[0]}#{i}", int(yv[i]), u[1][i]
        else:
            for u in us:
                yield u[0], int(u[2]), u[1][0]

    lin_by_C = {}
    for C in C_GRID:
        oof = []
        for f in fs:
            tr = [u for u in units if u[3] != f]
            te = [u for u in units if u[3] == f]
            if not tr or not te:
                continue
            Xtr, ytr = rows_of(tr)
            if len(set(ytr.tolist())) < 2:
                continue
            clf = LogisticRegression(C=C, class_weight="balanced", solver="liblinear",
                                     max_iter=1000, random_state=2026)
            clf.fit(Xtr, ytr)
            for k, y, x in iter_units(te):
                oof.append((k, y, float(clf.predict_proba(x.reshape(1, -1))[:, 1][0])))
        m = pooled(kind, oof)
        if m is not None:
            lin_by_C[C] = (m, oof)
    if not lin_by_C:
        die(f"{tag}: linear 臂失败")
    best_C = max(lin_by_C, key=lambda c: lin_by_C[c][0])
    lin_m, lin_oof = lin_by_C[best_C]

    nl_best = {}
    for seed in SEEDS:
        best = None
        for hidden in (64, 256):
            for depth in (1, 2):
                oof = []
                for f in fs:
                    tr = [u for u in units if u[3] != f]
                    te = [u for u in units if u[3] == f]
                    if not tr or not te:
                        continue
                    Xtr, ytr = rows_of(tr)
                    if len(set(ytr.tolist())) < 2:
                        continue
                    model = fit_mlp(Xtr, ytr, hidden, depth, seed, binary)
                    for k, y, x in iter_units(te):
                        oof.append((k, y, float(mlp_scores(model, x.reshape(1, -1), binary)[0])))
                m = pooled(kind, oof)
                if m is not None and (best is None or m > best[0]):
                    best = (m, hidden, depth, oof)
        if best is None:
            die(f"{tag}: nonlinear 臂失败 seed={seed}")
        nl_best[seed] = best

    # Δmetric 配对 bootstrap（独立单位组重采样；退化弃除计数）
    lin_map = {k: (y, s) for k, y, s in lin_oof}
    nl_maps = {seed: {k: (y, s) for k, y, s in nl_best[seed][3]} for seed in SEEDS}
    unit_rows = defaultdict(list)
    for k in lin_map:
        unit_rows[k.split("#")[0]].append(k)
    units_sorted = sorted(unit_rows)
    rng = np.random.RandomState(BOOT_SEED)
    per_seed = {}
    for seed in SEEDS:
        nmap = nl_maps[seed]
        nl_m = pooled(kind, [(k, *nmap[k]) for k in lin_map if k in nmap])
        delta_point = round(nl_m - lin_m, 6)
        dboot, n_degenerate = [], 0
        for _ in range(B_BOOT):
            pick = rng.choice(units_sorted, size=len(units_sorted), replace=True)
            lp, npp = [], []
            for u in pick:
                for k in unit_rows[u]:
                    yl, sl = lin_map[k]
                    lp.append((k, yl, sl))
                    if k in nmap:
                        npp.append((k, *nmap[k]))
            nv, lv = pooled(kind, npp), pooled(kind, lp)
            if nv is None or lv is None:
                n_degenerate += 1
                continue
            dboot.append(nv - lv)
        if len(dboot) < B_BOOT * 0.5:
            die(f"{tag}: bootstrap 退化抽样过多（{n_degenerate}）")
        dboot.sort()
        ci95 = [round(dboot[int(0.025 * len(dboot))], 6), round(dboot[int(0.975 * len(dboot))], 6)]
        lo99 = dboot[int(max(1, round(0.005 * len(dboot))))]
        hi99 = dboot[int(min(len(dboot) - 1, round(0.995 * len(dboot)) - 1))]
        bh, bd, _ = nl_best[seed][1], nl_best[seed][2], None
        per_seed[seed] = {"hidden": bh, "depth": bd, "nl_pooled": round(nl_m, 6),
                          "delta_point": delta_point, "ci95": ci95,
                          "ci99": [round(lo99, 6), round(hi99, 6)],
                          "boot_valid": len(dboot), "boot_degenerate": n_degenerate}
        print(f"[{tag}] seed{seed} Δmetric={delta_point} CI95={ci95}", flush=True)
    with open(os.path.join(OUTD, f"{tag}.capacity.tsv"), "w", newline="") as f:
        w = csv.writer(f, delimiter="\t", lineterminator="\n")
        w.writerow(["arm", "config", "metric"])
        w.writerow(["linear", f"C={best_C}", round(lin_m, 6)])
        for seed in SEEDS:
            v = per_seed[seed]
            w.writerow(["nonlinear", f"h{v['hidden']}d{v['depth']}_seed{seed}", v["nl_pooled"]])
        for seed in SEEDS:
            v = per_seed[seed]
            w.writerow([f"delta_point_seed{seed}", f"h{v['hidden']}d{v['depth']}", v["delta_point"]])
            w.writerow([f"delta_ci95_seed{seed}", f"h{v['hidden']}d{v['depth']}", str(v["ci95"])])
            w.writerow([f"delta_ci99_seed{seed}", f"h{v['hidden']}d{v['depth']}", str(v["ci99"])])
            w.writerow([f"boot_degenerate_seed{seed}", f"h{v['hidden']}d{v['depth']}", v["boot_degenerate"]])
        dmean = round(float(np.mean([v["delta_point"] for v in per_seed.values()])), 6)
        env95 = [min(v["ci95"][0] for v in per_seed.values()), max(v["ci95"][1] for v in per_seed.values())]
        env99 = [min(v["ci99"][0] for v in per_seed.values()), max(v["ci99"][1] for v in per_seed.values())]
        sig99 = all((ci[0] > 0 and ci[1] > 0) or (ci[0] < 0 and ci[1] < 0)
                    for v in per_seed.values() for ci in [v["ci99"]])
        w.writerow(["delta_seedmean", "-", dmean])
        w.writerow(["ci95_envelope", "-", str([round(x, 6) for x in env95])])
        w.writerow(["ci99_envelope", "-", str([round(x, 6) for x in env99])])
        w.writerow(["sig99_all_seeds", "-", sig99])
    print(f"[{tag}] linear={lin_m:.4f}(C{best_C}) Δseedmean={dmean} env95={env95} sig99={sig99}", flush=True)
    return {"tag": tag, "linear": round(lin_m, 6), "delta_seedmean": dmean,
            "ci95_envelope": [round(x, 6) for x in env95], "ci99_envelope": [round(x, 6) for x in env99],
            "significant_99_all_seeds": sig99, "n_units": len(lin_map)}


def run_direct_oof(tag, units):
    """FS_L2A：2-半交叉拟合 OOF（5 次重复×2 方向=10 折块），ΔAUROC 配对 bootstrap。"""
    X_all = np.concatenate([u[1] for u in units])
    y_all = np.array([u[2] for u in units])
    n = len(y_all)
    rng0 = np.random.RandomState(2026)
    blocks = []
    for rep in range(5):
        perm = rng0.permutation(n)
        half = n // 2
        blocks.append((perm[:half], perm[half:]))
        blocks.append((perm[half:], perm[:half]))
    lin_oof = {}
    best_C, best_auc = None, -1.0
    for C in C_GRID:
        oof = np.zeros(n)
        for ti, si in blocks:
            clf = LogisticRegression(C=C, class_weight="balanced", solver="liblinear",
                                     max_iter=1000, random_state=2026)
            clf.fit(X_all[ti], y_all[ti])
            oof[si] = clf.predict_proba(X_all[si])[:, 1]
        auc = roc_auc_score(y_all, oof)
        if auc > best_auc:
            best_auc, best_C, lin_oof = auc, C, oof.copy()
    nl_oofs = {}
    for seed in SEEDS:
        best = None
        for hidden in (64, 256):
            for depth in (1, 2):
                oof = np.zeros(n)
                for ti, si in blocks:
                    model = fit_mlp(X_all[ti], y_all[ti], hidden, depth, seed, True)
                    oof[si] = mlp_scores(model, X_all[si], True)
                auc = roc_auc_score(y_all, oof)
                if best is None or auc > best[0]:
                    best = (auc, hidden, depth, oof.copy())
        nl_oofs[seed] = best
        print(f"[{tag}] seed{seed} h{best[1]}d{best[2]} OOF-AUROC={best[0]:.4f}", flush=True)
    per_seed = {}
    rng = np.random.RandomState(BOOT_SEED)
    for seed in SEEDS:
        auc, h, d, nl_oof = nl_oofs[seed]
        delta_point = round(auc - best_auc, 6)
        dboot, n_deg = [], 0
        for _ in range(B_BOOT):
            pick = rng.randint(0, n, n)
            av, lv = roc_auc_score(y_all[pick], nl_oof[pick]), roc_auc_score(y_all[pick], lin_oof[pick])
            if not (np.isfinite(av) and np.isfinite(lv)):
                n_deg += 1
                continue
            dboot.append(av - lv)
        dboot.sort()
        per_seed[seed] = {"hidden": h, "depth": d, "nl_oof_auc": round(auc, 6),
                          "delta_point": delta_point,
                          "ci95": [round(dboot[int(0.025 * len(dboot))], 6),
                                   round(dboot[int(0.975 * len(dboot))], 6)],
                          "boot_valid": len(dboot), "boot_degenerate": n_deg}
        print(f"[{tag}] seed{seed} ΔAUROC={delta_point} CI95={per_seed[seed]['ci95']}", flush=True)
    with open(os.path.join(OUTD, f"{tag}.capacity.tsv"), "w", newline="") as f:
        w = csv.writer(f, delimiter="\t", lineterminator="\n")
        w.writerow(["arm", "config", "metric"])
        w.writerow(["linear", f"C={best_C}", round(best_auc, 6)])
        for seed in SEEDS:
            v = per_seed[seed]
            w.writerow(["nonlinear", f"h{v['hidden']}d{v['depth']}_seed{seed}", v["nl_oof_auc"]])
        for seed in SEEDS:
            v = per_seed[seed]
            w.writerow([f"delta_point_seed{seed}", f"h{v['hidden']}d{v['depth']}", v["delta_point"]])
            w.writerow([f"delta_ci95_seed{seed}", f"h{v['hidden']}d{v['depth']}", str(v["ci95"])])
            w.writerow([f"boot_degenerate_seed{seed}", f"h{v['hidden']}d{v['depth']}", v["boot_degenerate"]])
        dmean = round(float(np.mean([v["delta_point"] for v in per_seed.values()])), 6)
        w.writerow(["delta_seedmean", "-", dmean])
        w.writerow(["caveat", "-", "direct-fit 2-half cross-fitting OOF; optimistic; "
                                   "selection universe reused; ceiling regime"])
    print(f"[{tag}] linear={best_auc:.4f} Δseedmean={dmean}", flush=True)
    return {"tag": tag, "linear": round(best_auc, 6), "delta_seedmean": dmean,
            "ci95_envelope": "2-half cross-fitting OOF（直配乐观）",
            "ci99_envelope": "同左", "significant_99_all_seeds": "n/a（直配无 bootstrap）",
            "n_units": n}


def rebuild_summary():
    """从既有五份 TSV 重建 capacity_summary.tsv（跨运行合并；修复手拼/覆写问题）。"""
    rows_out = []
    for tag in ("KNOT_PRESENCE", "KNOT_TYPE_3_1_vs_other", "FS_REGION",
                "DISORDER500", "FS_L2A_rank"):
        p = os.path.join(OUTD, f"{tag}.capacity.tsv")
        if not os.path.exists(p):
            print(f"[rebuild] 缺 {tag}.capacity.tsv——跳过")
            continue
        get = {}
        for line in open(p):
            pr = line.rstrip("\n").split("\t")
            if len(pr) >= 3:
                get[pr[0]] = pr[2]
        rows_out.append({"tag": tag, "linear": get.get("linear", ""),
                         "delta_seedmean": get.get("delta_seedmean", ""),
                         "ci95_envelope": get.get("ci95_envelope", ""),
                         "ci99_envelope": get.get("ci99_envelope", ""),
                         "significant_99_all_seeds": get.get("sig99_all_seeds", ""),
                         "n_units": get.get("n_pairs", "")})
    with open(os.path.join(OUTD, "capacity_summary.tsv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["tag", "linear", "delta_seedmean", "ci95_envelope",
                                          "ci99_envelope", "significant_99_all_seeds", "n_units"],
                           delimiter="\t", lineterminator="\n")
        w.writeheader()
        w.writerows(rows_out)
    print(f"[rebuild] summary rebuilt from {len(rows_out)} TSVs")


def main():
    man = rd(os.path.join(ROOT, "data/splits/split_manifest.tsv"))
    dev_ids = {r["sample_id"] for r in man if r["split"] == "development"}
    fold_of = {r["sample_id"]: r["dev_fold"] for r in man if r["split"] == "development"}
    results = []

    # KNOT_PRESENCE
    kX = load_emb(os.path.join(EMB, "knots_dev"))
    kseq = {r["record_id"]: r for r in rd(os.path.join(ROOT, "data/curated/knots_sequences.tsv"))}
    L = sel_layer("T-KNOT-PRESENCE.selected.json")
    units = []
    for r in man:
        if r["task_area"] == "knot" and r["split"] == "development":
            rec = r["sample_id"].split(":", 1)[1]
            if rec in kseq and r["sample_id"] in kX:
                units.append((r["sample_id"],
                              kX[r["sample_id"]]["mean_layers"][L].reshape(1, -1).astype(np.float32),
                              int(kseq[rec]["presence_target"]), fold_of[r["sample_id"]]))
    if not os.path.exists(os.path.join(OUTD, "KNOT_PRESENCE.capacity.tsv")):
        results.append(run_universe("KNOT_PRESENCE", units, "auroc", True))
    else:
        print("[skip] KNOT_PRESENCE 已有产物")

    # KNOT_TYPE
    kt = {r["record_id"]: r for r in rd(os.path.join(ROOT, "data/curated/knots.tsv"))
          if r["type_task_tier"] == "eligible"}
    L = sel_layer("T-KNOT-TYPE.selected.json")
    units = []
    for r in man:
        if r["task_area"] == "knot" and r["split"] == "development":
            rec = r["sample_id"].split(":", 1)[1]
            if rec in kt and r["sample_id"] in kX:
                units.append((r["sample_id"],
                              kX[r["sample_id"]]["mean_layers"][L].reshape(1, -1).astype(np.float32),
                              1 if kt[rec]["c2_primary"] == "3_1" else 0, fold_of[r["sample_id"]]))
    if not os.path.exists(os.path.join(OUTD, "KNOT_TYPE_3_1_vs_other.capacity.tsv")):
        results.append(run_universe("KNOT_TYPE_3_1_vs_other", units, "auroc", True))
    else:
        print("[skip] KNOT_TYPE 已有产物")

    # FS_REGION
    fX = load_emb(os.path.join(EMB, "fs_region"))
    ivc = defaultdict(list)
    for r in rd(os.path.join(ROOT, "data/interim/p303/fs_region_intervals.tsv")):
        ivc[r["name"]].append((int(r["start"]), int(r["end"])))
    pair_of_chain = {r["name"]: r["pair_id"] for r in rd(
        os.path.join(ROOT, "data/interim/p303/fs_region_intervals.tsv"))}
    fold_of_pair = {r["sample_id"]: r["dev_fold"] for r in man
                    if r["task_area"] == "fold_switch" and r["split"] == "development"}
    L = sel_layer("T-FS-REGION.selected.json")
    # 冻结容量宇宙=双满足 usable 6 双类链；敏感性 10 链不跑非线性（probes.yaml 明文）
    usable_chains = {r["name"] for r in rd(os.path.join(ROOT, "data/interim/p303/fs_region_intervals.tsv"))
                     if r["label_decision"] == "usable"}
    units = []
    for ch, z in fX.items():
        if ch not in usable_chains:
            continue
        R = z["resid_layers"][L].astype(np.float32)
        y = np.zeros(R.shape[0], dtype=int)
        for s, e in ivc[ch]:
            y[s - 1:e] = 1
        if len(set(y.tolist())) < 2:
            continue
        units.append((ch, R, y, fold_of_pair.get(pair_of_chain[ch], "NA")))
    if not os.path.exists(os.path.join(OUTD, "FS_REGION.capacity.tsv")):
        results.append(run_universe("FS_REGION", units, "auprc", True))
    else:
        print("[skip] FS_REGION 已有产物")

    # DISORDER500
    dX = load_emb(os.path.join(EMB, "disorder_dev"))
    ridx = {r["name"]: [int(x) for x in r["resid_indices"].split(",")]
            for r in rd(os.path.join(ROOT, "data/interim/p303/disorder_resid_idx.tsv"))}
    state_by_res = defaultdict(dict)
    for r in rd(os.path.join(ROOT, "data/curated/disorder_masks.tsv")):
        if r["state"] in ("0", "1") and r["mask"] == "1":
            for p in range(int(r["start"]), int(r["end"]) + 1):
                state_by_res[r["disprot_id"]][p] = int(r["state"])
    dev_dis = {r["sample_id"].split(":", 1)[1] for r in man
               if r["task_area"] == "disorder" and r["split"] == "development"}
    avail = sorted(n for n in dX if n in dev_dis and n in state_by_res and state_by_res[n])
    if len(avail) != 2279:
        die(f"DISORDER500 dev 过滤后 {len(avail)} != 2279（断言，一审 MAJOR-4）")
    dual = [n for n in avail if len(set(state_by_res[n].values())) >= 2]
    rest = [n for n in avail if n not in set(dual)]
    rng = np.random.RandomState(2026)
    fill = sorted(np.array(rest)[rng.choice(len(rest), size=500 - len(dual), replace=False)])
    pick = sorted(dual + list(fill))
    with open(os.path.join(ROOT, "data/interim/p303/disorder500_sample.tsv"), "w", newline="") as f:
        w = csv.writer(f, delimiter="\t", lineterminator="\n")
        w.writerow(["disprot_id", "is_dual_class"])
        for n in pick:
            w.writerow([n, n in set(dual)])
    L = sel_layer("T-DISORDER-RES.selected.json")
    units = []
    for n in pick:
        pos = [i for i, p in enumerate(ridx[n]) if p <= 1022]
        R = dX[n]["resid_layers"][L][pos].astype(np.float32)
        y = np.array([state_by_res[n][ridx[n][i]] for i in pos], dtype=int)
        units.append((n, R, y, dis_fold_of(man, n)))
    if not os.path.exists(os.path.join(OUTD, "DISORDER500.capacity.tsv")):
        results.append(run_universe("DISORDER500", units, "auprc", True))
    else:
        print("[skip] DISORDER500 已有产物")

    # FS_L2A
    sX = load_emb(os.path.join(EMB, "l2_stage_a"))
    eX = load_emb(os.path.join(EMB, "fs_endpoints"))
    fs = [r for r in rd(os.path.join(ROOT, "data/curated/fold_switch_global.tsv"))
          if r["target"] == "1" and r["valid_mask"] == "1"]
    L = sel_layer("T-FS-L2-PU-RANK.stageA.selected.json")
    units = []
    for n in sorted(sX):
        units.append((n, sX[n]["mean_layers"][L].reshape(1, -1).astype(np.float32), 0, "A"))
    for r in fs:
        a = eX[f"{r['pair_id']}_A"]["mean_layers"][L].astype(np.float32)
        b = eX.get(f"{r['pair_id']}_B", eX[f"{r['pair_id']}_A"])["mean_layers"][L].astype(np.float32)
        units.append((f"POS_{r['pair_id']}", ((a + b) / 2).reshape(1, -1), 1, "P"))
    if not os.path.exists(os.path.join(OUTD, "FS_L2A_rank.capacity.tsv")):
        results.append(run_direct_oof("FS_L2A_rank", units))
    else:
        print("[skip] FS_L2A 已有产物")

    # summary（仅 results 非空；防破坏性覆写）
    if not results:
        rebuild_summary()
        print("[p304] 全部宇宙已有产物——summary 已从 TSV 重建（防破坏性覆写）")
        return
    with open(os.path.join(OUTD, "capacity_summary.tsv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["tag", "linear", "delta_seedmean", "ci95_envelope",
                                          "ci99_envelope", "significant_99_all_seeds", "n_units"],
                           delimiter="\t", lineterminator="\n", extrasaction="ignore")
        w.writeheader()
        w.writerows(results)
    json.dump({"run_ts": RUN_TS, "results": results,
               "note": "全部宇宙 development；KNOT_TYPE 为 3_1-vs-other 二值化语义；FS_L2A 为 2-半交叉"
                       "拟合 OOF ΔAUROC（直配乐观语义）；Δ=两臂 pooled metric 之差的配对 bootstrap"
                       "（独立单位组重采样 B=2000 种子 2026，退化抽样弃除并计数）；多重性=capacity_gain "
                       "族 5 比较→99% CI Bonferroni 水平（bootstrap-CI 无 p 值不适用 Holm——probes.yaml "
                       "冻结 Holm 文本的偏离已在 P3.04 报告登记）；显著性=all seeds 99% CI 同号。"},
              open(os.path.join(OUTD, "capacity_qc.json"), "w"), ensure_ascii=False, indent=1, sort_keys=True)
    print("[p304] DONE")


def dis_fold_of(man, dis_id):
    for r in man:
        if r["sample_id"] == f"disorder:{dis_id}":
            return r["dev_fold"]
    return "NA"


if __name__ == "__main__":
    main()
