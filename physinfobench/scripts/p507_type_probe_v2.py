#!/usr/bin/env python3
"""P5.07 修复版类型探针（预注册设计执行：configs/p507_type_probe_design.yaml）。

输入（全部冻结/预注册）：
  - data/interim/p507_final_universe.json：终版宇宙（冻结 86+旧 71 节点、分量、
    LCO 折、类型）——由本地"终版图构建器"在 814558 边并入后生成并同步；
  - 嵌入：data/interim/p507_emb/（新链）+ data/interim/p402/emb_knots_resid/（旧链）。
输出：results/p507_type_probe_v2/{selection_grid.tsv, lco_predictions.tsv,
  confusion_chain.tsv, confusion_component.tsv, per_type.tsv, metrics.json}
纪律：4_1/5_2 案例级标注；macro-F1 仅描述；禁止同分量拆分训练/测试（分量原子进出）。
"""
import csv
import argparse
import json
import os
import sys
from collections import defaultdict, Counter

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, balanced_accuracy_score, confusion_matrix

ROOT = "/lenovofs1/home/jyma/PLM_benchmark/p303_resplit/proj"
OUTD = f"{ROOT}/results/p507_type_probe_v2"
SEED = 2026
C_GRID = [0.01, 0.1, 1, 10]
LAYER = {"h23": 0, "h29": 1, "h33": 2}
CLASSES = ["3_1", "4_1", "5_2"]  # 5_1/6_1 不可评价（预注册边界）


def die(m):
    print(f"[p507v2 FATAL] {m}", file=sys.stderr)
    sys.exit(1)


def load_emb(manifest_path):
    man = list(csv.DictReader(open(manifest_path), delimiter="\t"))
    d = os.path.dirname(manifest_path)
    key2arr = {}
    for row in man:
        key = row["key"]
        if key not in key2arr:
            with np.load(os.path.join(d, key + ".npz"), allow_pickle=False) as z:
                key2arr[key] = {k: z[k] for k in z.files if k != "meta"}
                key2arr[key]["_layer_set"] = json.loads(z["meta"].item())["layer_set"].split("|", 1)[0]
    return {row["name"]: key2arr[row["key"]] for row in man}


def main():
    global OUTD
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output', required=True, help='New correction/development output; never overwrite historical results')
    args = ap.parse_args()
    OUTD = args.output
    if os.path.isdir(OUTD) and os.listdir(OUTD):
        ap.error('Output exists and is nonempty; refuse overwrite')
    uni = json.load(open(f"{ROOT}/data/interim/p507_final_universe.json"))
    nodes = uni["nodes"]  # sid -> {"type","component","source","store_key"}
    stores = {}
    for sname, path in [("new", f"{ROOT}/data/interim/p507_emb/extract_manifest.tsv"),
                        ("old", f"{ROOT}/data/interim/p402/emb_knots_resid/extract_manifest.tsv")]:
        stores[sname] = load_emb(path)

    X, y, comp, sid_list = {}, {}, {}, []
    for sid, info in nodes.items():
        st = stores["new" if info["store"] == "new" else "old"]
        if sid not in st:
            die(f"表示缺失（禁止静默子集）: {sid}")
        if info["store"] == "old":
            if st[sid]["_layer_set"].split('+')[-1] != 'resid_23_29_33':
                die(f"残基层元数据与23/29/33索引不一致: {sid}")
            X[sid] = {ln: st[sid]["resid_layers"][li].astype(np.float32).mean(axis=0)
                      for ln, li in LAYER.items()}          # resid 3 层：23/29/33
        else:
            if st[sid]["_layer_set"] != 'mean_5_11_17_23_29_33':
                die(f"均值层元数据与5/11/17/23/29/33索引不一致: {sid}")
            MEAN_IDX = {"h23": 3, "h29": 4, "h33": 5}       # mean 6 层中的 23/29/33
            X[sid] = {ln: st[sid]["mean_layers"][MEAN_IDX[ln]].astype(np.float32)
                      for ln in LAYER}
        y[sid] = info["type"]
        comp[sid] = info["component"]
        sid_list.append(sid)
    print(f"宇宙 {len(sid_list)} | 类型 {dict(Counter(y.values()))}", flush=True)

    comps = sorted(set(comp.values()))
    comps_major = [c for c in comps
                   if any(y[s] == "3_1" for s in sid_list if comp[s] == c)]

    def lco_predict(layer, C):
        preds = {}
        for test_c in comps:
            tr = [s for s in sid_list if comp[s] != test_c]
            te = [s for s in sid_list if comp[s] == test_c]
            if not tr:
                continue
            clf = LogisticRegression(C=C, class_weight="balanced", solver="lbfgs",
                                     max_iter=1000, random_state=SEED)  # 多类用 lbfgs（P3.03 macro_f1 同款；liblinear 新版不支持多类）
            clf.fit(np.stack([X[s][layer] for s in tr]), [y[s] for s in tr])
            for s in te:
                preds[s] = clf.predict(X[s][layer].reshape(1, -1))[0]
        return preds

    # 同一轮LCO内选择：分量内链级二元准确率的等权均值。报告为开发性描述。
    grid = []
    for layer in LAYER:
        for C in C_GRID:
            preds = lco_predict(layer, C)
            pred3 = {s: ("3_1" if p == "3_1" else "other") for s, p in preds.items()}
            y3 = {s: ("3_1" if y[s] == "3_1" else "other") for s in preds}
            by_comp = defaultdict(list)
            for s in preds:
                by_comp[comp[s]].append(pred3[s] == y3[s])
            comp_acc = np.mean([np.mean(v) for v in by_comp.values()])
            grid.append({"layer": layer, "C": C,
                         "component_mean_chain_binary_accuracy": round(float(comp_acc), 4),
                         "n_pred": len(preds)})
    best = max(grid, key=lambda r: (r["component_mean_chain_binary_accuracy"], -C_GRID.index(r["C"])))
    print(f"选中 {best}", flush=True)

    preds = lco_predict(best["layer"], best["C"])
    # 链级混淆
    labels = CLASSES
    yt = [y[s] for s in preds]
    yp = [preds[s] for s in preds]
    cm_chain = confusion_matrix(yt, yp, labels=labels)
    # 分量级（多数票）
    comp_pred, comp_true = {}, {}
    for s in preds:
        c = comp[s]
        comp_pred.setdefault(c, Counter())[preds[s]] += 1
        comp_true.setdefault(c, y[s])
    cpt, cpp = [], []
    for c in comp_pred:
        if comp_true[c] not in labels:
            continue
        cpt.append(comp_true[c])
        cpp.append(comp_pred[c].most_common(1)[0][0])
    cm_comp = confusion_matrix(cpt, cpp, labels=labels)

    per_type = {}
    for i, t in enumerate(labels):
        tp = cm_chain[i, i]
        per_type[t] = {
            "support_chains": int(cm_chain[i].sum()),
            "support_components": int(sum(1 for x in cpt if x == t)),
            "recall": round(float(tp / cm_chain[i].sum()), 4) if cm_chain[i].sum() else None,
            "precision": round(float(tp / cm_chain[:, i].sum()), 4) if cm_chain[:, i].sum() else None,
            "f1": round(float(f1_score(yt, yp, average="macro", labels=[t])), 4) if cm_chain[i].sum() else None,
            "component_recall": round(float(cm_comp[i, i] / cm_comp[i].sum()), 4) if cm_comp[i].sum() else None,
            "claim_level": "经选择的开发性分量描述" if t == "3_1" else "案例级（绑定分量数量有限）"}

    macro_f1 = f1_score(yt, yp, average="macro", labels=labels)
    bacc = balanced_accuracy_score(yt, yp)
    # 分量级 bootstrap（3_1 类 recall 的分量重采样）
    rng = np.random.RandomState(SEED)
    t3 = [i for i, t in enumerate(cpt) if t == "3_1"]
    draws = []
    for _ in range(2000):
        pick = rng.randint(0, len(t3), size=len(t3))
        hits = [1 if cpp[t3[i]] == "3_1" else 0 for i in pick]
        draws.append(float(np.mean(hits)))
    # 置换基线（分量级标签置换 ×100：整分量类型重排——分量内置换在类型纯分量上零效应）
    comp_members = defaultdict(list)
    for s in preds:
        comp_members[comp[s]].append(s)
    comp_ids = sorted(comp_members)
    comp_true_type = {c: y[comp_members[c][0]] for c in comp_ids}
    perm_macro = []
    for i in range(100):
        prng = np.random.RandomState(SEED + i)
        shuffled = prng.permutation([comp_true_type[c] for c in comp_ids])
        y_perm = {}
        for c, t in zip(comp_ids, shuffled):
            for s in comp_members[c]:
                y_perm[s] = t
        perm_macro.append(f1_score([y_perm[s] for s in preds], yp, average="macro", labels=labels))
    majority_macro = f1_score(yt, [Counter(yt).most_common(1)[0][0]] * len(yt),
                              average="macro", labels=labels)

    os.makedirs(OUTD, exist_ok=True)
    with open(f"{OUTD}/lco_predictions.tsv", "w", newline="") as f:
        w = csv.writer(f, delimiter="\t", lineterminator="\n")
        w.writerow(["sid", "true", "pred", "component", "source"])
        for s in preds:
            w.writerow([s, y[s], preds[s], comp[s], nodes[s]["source"]])
    with open(f"{OUTD}/selection_grid.tsv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(grid[0].keys()), delimiter="\t", lineterminator="\n")
        w.writeheader(); w.writerows(grid)
    metrics = {"selected": best,
               "selection_scope": "Same nonnested LCO used for selection and reporting; descriptive only",
               "recall_unit": "per_type.recall is chain-level; component_recall and bootstrap interval are component-level",
               "n_nodes": len(sid_list),
               "n_components": len(comps),
               "confusion_chain": {"labels": labels, "matrix": cm_chain.tolist()},
               "confusion_component": {"labels": labels, "matrix": cm_comp.tolist()},
               "per_type": per_type,
               "macro_f1_descriptive": round(float(macro_f1), 4),
               "balanced_accuracy_descriptive": round(float(bacc), 4),
               "recall_3_1_component_boot95": [round(float(np.percentile(draws, 2.5)), 4),
                                               round(float(np.percentile(draws, 97.5)), 4)],
               "permutation_macro_f1": {"mean": round(float(np.mean(perm_macro)), 4),
                                        "max": round(float(np.max(perm_macro)), 4)},
               "majority_macro_f1": round(float(majority_macro), 4),
               "claim_boundaries": ["4_1/5_2=案例级（预注册）", "5_1/6_1 不可评价",
                                    "macro-F1 仅描述不作泛化主张"]}
    json.dump(metrics, open(f"{OUTD}/metrics.json", "w"), indent=1, ensure_ascii=False)
    print(json.dumps({k: metrics[k] for k in ("selected", "macro_f1_descriptive",
                                              "permutation_macro_f1", "per_type")},
                     ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
