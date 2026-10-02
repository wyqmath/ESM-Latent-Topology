#!/usr/bin/env python3
"""P5.08 集群执行：mmseqs 同源排除 → 嵌入抽取 → 冻结读出外部评估（一体，断点续跑）。

步骤（configs/p508_disorder_contrast_design.yaml 预注册）：
  1) mmseqs easy-cluster(p508_seqs.fa + disprot_all.fa, ≥0.3/cov0.7)——
     与任一 DisProt 成员共簇的 p508 链整链排除；
  2) 抽取幸存链 hidden33 逐残基（resid-index=截断域）；
  3) 冻结读出（dev 2,279 蛋白分母域训练，C=0.01/liblinear/seed2026）逐蛋白评估：
     AUPRC(主)/balanced-acc/MCC + 池化(次)+蛋内置换×100+区段 IoU(描述)。
输出：results/p508_disorder_contrast/{per_protein.tsv, pooled_metrics.json, qc.json}
"""
import csv
import json
import os
from pathlib import Path
import re
import subprocess
import sys
from collections import defaultdict

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score

ROOT = os.environ.get("PLM_PROJECT_ROOT", str(Path(__file__).resolve().parents[1]))
I = f"{ROOT}/data/interim"
RES = f"{ROOT}/results/repairs/20261001/p508_h33_correction"
MMSEQS = "/lenovofs1/home/jyma/PLM_benchmark/tools_linux/mmseqs/bin/mmseqs"
VENV_PY = "/lenovofs1/home/jyma/PLM_benchmark/p303_resplit/venv/bin/python"
SEED = 2026
EXPECTED_MODEL = "facebook/esm2_t33_650M_UR50D"
EXPECTED_REVISION = "08e4846e537177426273712802403f7ba8261b6c"


def die(m):
    print(f"[p508 FATAL] {m}", file=sys.stderr)
    sys.exit(1)


def rd(p):
    with open(p, newline="") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def residual_layer(archive, manifest_row, target_layer=33):
    """Resolve a residual layer from metadata and verify manifest/cache agreement."""
    if "meta" not in archive or "resid_layers" not in archive or "resid_indices" not in archive:
        raise ValueError("Residual cache metadata/indices missing")
    meta = json.loads(str(np.asarray(archive["meta"]).item()))
    if (meta.get("model"), meta.get("revision")) != (EXPECTED_MODEL, EXPECTED_REVISION):
        raise ValueError("Residual cache model/revision differs from frozen ESM-2")
    for field in ("key", "layer_set", "revision", "seq_sha256"):
        if field not in manifest_row or meta.get(field) != manifest_row[field]:
            raise ValueError(f"Residual manifest/cache {field} mismatch")
    match = re.search(r"(?:^|\+)resid_([0-9_]+)(?:\||$)", meta["layer_set"])
    if not match:
        raise ValueError("Residual layer_set metadata unparseable")
    layers = [int(x) for x in match.group(1).split("_")]
    array = np.asarray(archive["resid_layers"])
    indices = np.asarray(archive["resid_indices"])
    if array.ndim != 3 or len(layers) != array.shape[0] or len(set(layers)) != len(layers):
        raise ValueError("Residual layer metadata/array shape mismatch")
    if len(indices) != array.shape[1] or int(meta["n_resid_stored"]) != len(indices):
        raise ValueError("Residual index metadata/array shape mismatch")
    if target_layer not in layers:
        raise ValueError(f"Requested hidden{target_layer} absent; no fallback permitted")
    slot = layers.index(target_layer)
    return array[slot], {"key": meta["key"], "layer_set": meta["layer_set"],
                         "target_layer": target_layer, "selected_cache_slot": slot,
                         "model": meta["model"], "revision": meta["revision"],
                         "seq_sha256": meta["seq_sha256"]}


def step1_exclude():
    correction = os.environ.get("PLM_P508_CORRECTION_SET", f"{ROOT}/results/repairs/20261001/p508/correction_set.json")
    if not os.path.exists(correction):
        die("True accession identity correction set is required; legacy final_set cannot bypass repair")
    cohort = json.load(open(correction))
    kept = cohort["kept"]
    forbidden = set(cohort["quarantined_identity"]) | set(cohort["excluded_accession_overlap"])
    if len(kept) != len(set(kept)) or not kept or set(kept) & forbidden:
        die("Correction cohort contains duplicate/empty/excluded identity records")
    return sorted(kept)


def step2_embed(kept):
    man = f"{I}/p508_emb/extract_manifest.tsv"
    if not os.path.exists(man):
        die("Historical external cache manifest missing; correction runner does not overwrite/extract caches")
    got = {r["name"] for r in rd(man)}
    if not set(kept) <= got:
        die(f"Correction cohort missing cached representations: {sorted(set(kept)-got)}")


def step3_eval(kept):
    if os.path.exists(RES) and os.listdir(RES):
        die("Nonempty correction output; refuse overwrite")
    labels = {m["name"]: m["labels"] for m in json.load(open(f"{I}/p508_labels.json"))}
    # —— 冻结读出：dev 训练（P5.02 同款）——
    man = rd(f"{ROOT}/data/interim/p303/emb_disorder_dev/extract_manifest.tsv")
    d = f"{ROOT}/data/interim/p303/emb_disorder_dev"
    key2arr = {}
    layer_audit = {"target_layer": 33, "development": {}, "external": {}}
    for row in man:
        if row["key"] not in key2arr:
            with np.load(os.path.join(d, row["key"] + ".npz"), allow_pickle=False) as z:
                features, audit = residual_layer(z, row, 33)
                key2arr[row["key"]] = {"resid_target": features, "resid_indices": z["resid_indices"]}
                layer_audit["development"][row["key"]] = audit
        else:
            if any(layer_audit["development"][row["key"]][field] != row[field]
                   for field in ("key", "layer_set", "revision", "seq_sha256")):
                raise ValueError("Development alias manifest metadata differs from cached identity")
    dev_store = {row["name"]: key2arr[row["key"]] for row in man}
    if len(dev_store) != 2279:
        die(f"dev 宇宙断言失败 {len(dev_store)}")
    ridx_dev = {r["name"]: [int(x) for x in r["resid_indices"].split(",")]
                for r in rd(f"{ROOT}/data/interim/p303/disorder_resid_idx.tsv")}
    for nm, array in dev_store.items():
        if array["resid_indices"].tolist() != ridx_dev[nm]:
            raise ValueError(f"Development cached index ordering differs from frozen index table: {nm}")
    state = defaultdict(dict)
    for r in rd(f"{ROOT}/data/curated/disorder_masks.tsv"):
        if r["state"] in ("0", "1") and r["mask"] == "1":
            for p in range(int(r["start"]), int(r["end"]) + 1):
                state[r["disprot_id"]][p] = int(r["state"])
    Xl, yl = [], []
    for nm in sorted(dev_store):
        pos_idx = [i for i, p in enumerate(ridx_dev[nm]) if p in state[nm]]
        Xl.append(dev_store[nm]["resid_target"].astype(np.float32)[np.array(pos_idx)])
        yl.append([state[nm][ridx_dev[nm][i]] for i in pos_idx])
    clf = LogisticRegression(C=0.01, class_weight="balanced", solver="liblinear",
                             max_iter=1000, random_state=SEED)
    clf.fit(np.concatenate(Xl), np.array(sum(yl, [])))
    print("[p508] dev 读出拟合完成", flush=True)

    # —— 外部评估 ——
    man2 = [r for r in rd(f"{I}/p508_emb/extract_manifest.tsv") if r["name"] in set(kept)]
    if {r["name"] for r in man2} != set(kept):
        die("External cached manifest and requested correction cohort differ")
    d2 = f"{I}/p508_emb"
    key2 = {}
    for row in man2:
        if row["key"] not in key2:
            with np.load(os.path.join(d2, row["key"] + ".npz"), allow_pickle=False) as z:
                features, audit = residual_layer(z, row, 33)
                key2[row["key"]] = {"resid_target": features}
                layer_audit["external"][row["key"]] = audit
        else:
            if any(layer_audit["external"][row["key"]][field] != row[field]
                   for field in ("key", "layer_set", "revision", "seq_sha256")):
                raise ValueError("External alias manifest metadata differs from cached identity")
    ext = {row["name"]: key2[row["key"]] for row in man2}
    ridx_ext = {r["name"]: len(r["resid_indices"].split(","))
                for r in rd(f"{I}/p508_final_ridx.tsv")}

    def iou_segments(y, s, min_len=5):
        def segs(v):
            out, st = [], None
            for i, x in enumerate(v):
                if x == 1 and st is None:
                    st = i
                elif x != 1 and st is not None:
                    if i - st >= min_len:
                        out.append((st, i))
                    st = None
            if st is not None and len(v) - st >= min_len:
                out.append((st, len(v)))
            return out
        P, G = segs(s), segs(y)
        if not G:
            return None
        used, hits = set(), 0
        for a, b in G:
            best = None
            for j, (c, d_) in enumerate(P):
                if j in used:
                    continue
                inter = min(b, d_) - max(a, c)
                if inter > 0:
                    iou = inter / (max(b, d_) - min(a, c))
                    if best is None or iou > best[0]:
                        best = (iou, j)
            if best and best[0] >= 0.3:
                used.add(best[1])
                hits += 1
        return hits / len(G)

    per_protein = []
    pooled_scores, pooled_labels = [], []
    for nm in sorted(ext):
        lab = labels[nm][:ridx_ext[nm]]
        sc = clf.predict_proba(ext[nm]["resid_target"].astype(np.float32))[:, 1]
        if len(lab) != len(sc):
            die(f"标签/表示不对齐 {nm}: {len(lab)} vs {len(sc)}")
        if not lab or set(lab) == {0} or set(lab) == {1}:
            continue
        from sklearn.metrics import balanced_accuracy_score, matthews_corrcoef
        pred = (sc >= 0.5).astype(int)
        per_protein.append({
            "name": nm, "n_res": len(lab), "base_rate": round(float(np.mean(lab)), 4),
            "auprc": round(float(average_precision_score(lab, sc)), 4),
            "balanced_acc": round(float(balanced_accuracy_score(lab, pred)), 4),
            "mcc": round(float(matthews_corrcoef(lab, pred)), 4),
            "segment_recall_iou030": (lambda v: None if v is None else round(v, 4))(iou_segments(lab, pred)),
        })
        pooled_scores.append(sc)
        pooled_labels.append(lab)
    Xc = np.concatenate(pooled_scores)
    yc = np.concatenate(pooled_labels)
    pooled = average_precision_score(yc, Xc)
    perm = []
    for i in range(100):
        prng = np.random.RandomState(SEED + i)
        yp = np.concatenate([prng.permutation(v) for v in pooled_labels])
        perm.append(average_precision_score(yp, Xc))
    os.makedirs(RES, exist_ok=True)
    with open(f"{RES}/per_protein.tsv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(per_protein[0].keys()), delimiter="\t", lineterminator="\n")
        w.writeheader()
        w.writerows(per_protein)
    aups = np.array([p["auprc"] for p in per_protein])
    base = np.array([p["base_rate"] for p in per_protein])
    metrics = {
        "n_proteins": len(per_protein),
        "per_protein_auprc_mean": round(float(aups.mean()), 4),
        "per_protein_auprc_median": round(float(np.median(aups)), 4),
        "per_protein_base_rate_mean": round(float(base.mean()), 4),
        "improvement_over_base": round(float((aups - base).mean()), 4),
        "pooled_auprc_secondary": round(float(pooled), 4),
        "pooled_base_rate": round(float(yc.mean()), 4),
        "pooled_permutation": {"mean": round(float(np.mean(perm)), 4),
                               "max": round(float(np.max(perm)), 4)},
        "balanced_acc_mean": round(float(np.mean([p["balanced_acc"] for p in per_protein])), 4),
        "mcc_mean": round(float(np.mean([p["mcc"] for p in per_protein])), 4),
        "segment_recall_mean": round(float(np.mean([p["segment_recall_iou030"] for p in per_protein
                                                    if p["segment_recall_iou030"] is not None])), 4),
        "note": "Supplementary correction; frozen h33/C=.01 devfit replay; no layer/threshold selection",
    }
    json.dump(metrics, open(f"{RES}/pooled_metrics.json", "w"), indent=1, ensure_ascii=False)
    layer_audit["n_dev_manifest_names"] = len(dev_store)
    layer_audit["n_external_manifest_names"] = len(ext)
    json.dump(layer_audit, open(f"{RES}/layer_audit.json", "w"), indent=1)
    print(json.dumps(metrics, ensure_ascii=False, indent=1))


def main():
    kept = step1_exclude()
    step2_embed(kept)
    step3_eval(kept)


if __name__ == "__main__":
    main()
