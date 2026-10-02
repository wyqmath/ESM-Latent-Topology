#!/usr/bin/env python3
"""P5.07 类型探针（development 诊断）：扩库平衡后的打结类型可读性。

协议（P3.03 T-KNOT-TYPE 同款 + 扩库宇宙）：
  - 宇宙 = 现有 dev type-eligible（77）+ P5.07 新验证链（111，绑定图分量并入既有折；
    44 条跨折冲突已剔除）；
  - 特征 = hidden 23/29/33 全残基均值（资源约束下 3 层网格，P3.03 为 6 层，如实披露）；
  - 读取器 = 多类 LogisticRegression(C 网格 0.01/0.1/1/10, class_weight=balanced,
    liblinear, max_iter=1000, seed 2026)；选择 = 分组 5 折 CV macro-F1；
  - 输出 = 选中配置的 OOF 混淆矩阵 + 逐类 recall/precision/F1/support + macro-F1
    （链级 bootstrap B=2000 CI）+ 旧宇宙（77）同协议对照（复现 0.506 基准）；
  - 5_1（n=1）不参与选择与逐类召回统计（样本量边界，如实披露），仅登记。
"""
import ast
import csv
import json
import os
import sys
from collections import defaultdict, Counter

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

ROOT = "/lenovofs1/home/jyma/PLM_benchmark/p303_resplit/proj"
SEED = 2026
C_GRID = [0.01, 0.1, 1, 10]
LAYERS = {"h23": 0, "h29": 1, "h33": 2}  # resid_layers 索引


def die(m):
    print(f"[p507probe FATAL] {m}", file=sys.stderr)
    sys.exit(1)


def rd(p, d="\t"):
    with open(p, newline="") as f:
        return list(csv.DictReader(f, delimiter=d))


def load_emb(manifest_path):
    man = rd(manifest_path)
    d = os.path.dirname(manifest_path)
    key2arr = {}
    for row in man:
        key = row["key"]
        if key not in key2arr:
            with np.load(os.path.join(d, key + ".npz"), allow_pickle=False) as z:
                key2arr[key] = {k: z[k] for k in z.files if k != "meta"}
    return {row["name"]: key2arr[row["key"]] for row in man}


def main():
    uni = json.load(open(f"{ROOT}/data/interim/p507_probe_universe.json"))
    fold_of_new = uni["fold_of"]
    types_new = uni["types"]

    # 现有 dev type-eligible（77）
    knots = {r["record_id"].upper(): r for r in rd(f"{ROOT}/data/curated/knots.tsv")}
    man = rd(f"{ROOT}/data/splits/split_manifest.tsv")
    old = {}
    for r in man:
        if r["task_area"] == "knot" and r["split"] == "development":
            rec = knots.get(r["sample_id"].split(":", 1)[1].upper())
            if rec and rec["type_task_tier"] == "eligible":
                old[r["sample_id"]] = {"type": rec["c2_primary"], "fold": r["dev_fold"],
                                       "store": ("knots_resid", r["sample_id"])}

    # 新链
    new = {}
    for c in uni["final_survivors"]:
        new[f"p507:{c}"] = {"type": types_new[c], "fold": fold_of_new[c],
                            "store": ("p507_emb", f"p507:{c}")}

    # 特征装载
    X, y, folds, names = defaultdict(list), {}, {}, []
    stores = {}
    for sid, info in list(old.items()) + list(new.items()):
        sname = info["store"][0]
        if sname not in stores:
            if sname == "knots_resid":
                stores[sname] = load_emb(f"{ROOT}/data/interim/p402/emb_knots_resid/extract_manifest.tsv")
            else:
                stores[sname] = load_emb(f"{ROOT}/data/interim/p507_emb/extract_manifest.tsv")
        store = stores[sname]
        key = info["store"][1]
        if key not in store:
            die(f"表示缺失（禁止静默子集）: {key}")
        X[sid] = {ln: store[key]["resid_layers"][li].astype(np.float32).mean(axis=0)
                  for ln, li in LAYERS.items()}
        y[sid] = info["type"]
        folds[sid] = str(info["fold"])
        names.append(sid)
    print(f"宇宙 {len(names)} = 旧 {len(old)} + 新 {len(new)}", flush=True)
    print("类型分布:", dict(Counter(y.values())), flush=True)

    # 分组 5 折（组=折，已有 dev_fold 语义）
    fold_ids = sorted({folds[n] for n in names})
    print("折:", fold_ids, flush=True)
    main_types = sorted({t for t in y.values() if t not in ("5_1",)})

    def oof_for(layer, C):
        preds = {}
        for f in fold_ids:
            tr = [n for n in names if folds[n] != f and y[n] in main_types]
            te = [n for n in names if folds[n] == f and y[n] in main_types]
            if not tr or not te:
                continue
            clf = LogisticRegression(C=C, class_weight="balanced", solver="liblinear",
                                     max_iter=1000, random_state=SEED)
            clf.fit(np.stack([X[n][layer] for n in tr]), [y[n] for n in tr])
            for n in te:
                preds[n] = clf.predict(X[n][layer].reshape(1, -1))[0]
        return preds

    selection = []
    for layer in LAYERS:
        for C in C_GRID:
            preds = oof_for(layer, C)
            yt = [y[n] for n in preds]
            macro = f1_score(yt, [preds[n] for n in preds], average="macro",
                             labels=sorted(set(yt)))
            selection.append({"layer": layer, "C": C, "macro_f1": round(float(macro), 6),
                              "n_oof": len(preds)})
    best = max(selection, key=lambda r: (r["macro_f1"], -C_GRID.index(r["C"])))
    print(f"选中: {best}", flush=True)

    preds = oof_for(best["layer"], best["C"])
    labels = main_types
    cm = np.zeros((len(labels), len(labels)), dtype=int)
    for n, p in preds.items():
        cm[labels.index(y[n]), labels.index(p)] += 1
    per_type = {}
    for i, t in enumerate(labels):
        tp = cm[i, i]
        recall = tp / cm[i].sum() if cm[i].sum() else float("nan")
        prec = tp / cm[:, i].sum() if cm[:, i].sum() else float("nan")
        per_type[t] = {"support": int(cm[i].sum()), "recall": round(float(recall), 4),
                       "precision": round(float(prec), 4)}
    yt = [y[n] for n in preds]
    yp = [preds[n] for n in preds]
    macro = f1_score(yt, yp, average="macro", labels=labels)
    # 链级 bootstrap CI（B=2000，seed 2026）
    rng = np.random.RandomState(SEED)
    items = list(preds)
    draws = []
    for _ in range(2000):
        pick = rng.randint(0, len(items), size=len(items))
        yt2 = [y[items[i]] for i in pick]
        yp2 = [preds[items[i]] for i in pick]
        draws.append(f1_score(yt2, yp2, average="macro", labels=labels))
    boot = {"ci95_low": round(float(np.percentile(draws, 2.5)), 4),
            "ci95_high": round(float(np.percentile(draws, 97.5)), 4)}

    # 旧宇宙对照（同协议，77 链）
    old_names = [n for n in names if n in old]
    old_types_main = sorted({y[n] for n in old_names} - {"5_1"})
    sel_old = []
    for layer in LAYERS:
        for C in C_GRID:
            pr = {}
            for f in fold_ids:
                tr = [n for n in old_names if folds[n] != f and y[n] in old_types_main]
                te = [n for n in old_names if folds[n] == f and y[n] in old_types_main]
                if not tr or not te:
                    continue
                clf = LogisticRegression(C=C, class_weight="balanced", solver="liblinear",
                                         max_iter=1000, random_state=SEED)
                clf.fit(np.stack([X[n][layer] for n in tr]), [y[n] for n in tr])
                for n in te:
                    pr[n] = clf.predict(X[n][layer].reshape(1, -1))[0]
            if pr:
                yt2 = [y[n] for n in pr]
                sel_old.append({"layer": layer, "C": C,
                                "macro_f1": round(float(f1_score(yt2, [pr[n] for n in pr],
                                                                 average="macro",
                                                                 labels=sorted(set(yt2)))), 6)})
    best_old = max(sel_old, key=lambda r: r["macro_f1"]) if sel_old else None

    out = {"universe": {"old": len(old), "new": len(new),
                        "types": dict(Counter(y[n] for n in names))},
           "selection_grid": selection, "selected": best,
           "oof_confusion_matrix": {"labels": labels, "matrix": cm.tolist()},
           "per_type": per_type,
           "macro_f1": round(float(macro), 4), "macro_f1_boot95": boot,
           "old_universe_same_protocol": best_old,
           "notes": ["5_1 n=1 不入选择与逐类统计（样本量边界）",
                     "层网格 3 层（23/29/33）为资源约束，P3.03 原网格 6 层，如实披露",
                     "5_1/6_1 的逐类召回无统计意义（n<3）"],
           "executed_at": {"utc": __import__("datetime").datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")}}
    with open(f"{ROOT}/data/interim/p507_type_probe_results.json", "w") as f:
        json.dump(out, f, indent=1, ensure_ascii=False, sort_keys=True)
    print(json.dumps({"selected": best, "macro_f1": out["macro_f1"], "boot": boot,
                      "per_type": per_type}, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
