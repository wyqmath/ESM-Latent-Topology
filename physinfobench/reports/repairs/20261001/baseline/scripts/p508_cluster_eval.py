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
import subprocess
import sys
from collections import defaultdict

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score

ROOT = "/lenovofs1/home/jyma/PLM_benchmark/p303_resplit/proj"
I = f"{ROOT}/data/interim"
RES = f"{ROOT}/results/p508_disorder_contrast"
MMSEQS = "/lenovofs1/home/jyma/PLM_benchmark/tools_linux/mmseqs/bin/mmseqs"
VENV_PY = "/lenovofs1/home/jyma/PLM_benchmark/p303_resplit/venv/bin/python"
SEED = 2026


def die(m):
    print(f"[p508 FATAL] {m}", file=sys.stderr)
    sys.exit(1)


def rd(p):
    with open(p, newline="") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def step1_exclude():
    final = f"{I}/p508_final_set.json"
    if os.path.exists(final):
        return json.load(open(final))["kept"]
    subprocess.run([MMSEQS, "easy-cluster", f"{I}/p508_combined.fa",
                    f"{I}/p508_mmseqs", f"{I}/p508_mm_tmp",
                    "--min-seq-id", "0.3", "-c", "0.7", "--cov-mode", "1"], check=True)
    members = defaultdict(list)
    for line in open(f"{I}/p508_mmseqs_cluster.tsv"):
        rep, m = line.rstrip("\n").split("\t")
        members[rep].append(m)
    p508 = set()
    for line in open(f"{I}/p508_seqs.fa"):
        if line.startswith(">"):
            p508.add(line[1:].strip())
    drop = set()
    for rep, ms in members.items():
        if any(m in p508 for m in ms) and any(m.startswith("disprot:") for m in ms):
            drop |= {m for m in ms if m in p508}
    kept = sorted(p508 - drop)
    json.dump({"kept": kept, "dropped_homology": sorted(drop)},
              open(final, "w"), indent=1)
    print(f"[p508] mmseqs 排除 {len(drop)}，幸存 {len(kept)}")
    return kept


def step2_embed(kept):
    man = f"{I}/p508_emb/extract_manifest.tsv"
    if not os.path.exists(man):
        fa = f"{I}/p508_final.fa"
        seqs = {}
        name = None
        for line in open(f"{I}/p508_seqs.fa"):
            if line.startswith(">"):
                name = line[1:].strip()
            else:
                seqs.setdefault(name, []).append(line.strip())
        with open(fa, "w") as f:
            for k in kept:
                f.write(f">{k}\n{''.join(seqs[k])}\n")
        # resid-index 过滤到幸存链
        rows = [r for r in rd(f"{I}/p508_resid_idx.tsv") if r["name"] in set(kept)]
        with open(f"{I}/p508_final_ridx.tsv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["name", "resid_indices"], delimiter="\t", lineterminator="\n")
            w.writeheader()
            w.writerows(rows)
        subprocess.run([VENV_PY, f"{ROOT}/scripts/extract_representations.py",
                        "--fasta", fa, "--out-dir", f"{I}/p508_emb",
                        "--resid-layers", "33",
                        "--resid-index-file", f"{I}/p508_final_ridx.tsv",
                        "--device", "cuda", "--batch-max-tokens", "49152"],
                       check=True, cwd=ROOT)
    got = {r["name"] for r in rd(man)}
    if got != set(kept):
        die(f"嵌入集合不符 {len(got)} vs {len(kept)}（禁止静默子集）")


def step3_eval(kept):
    labels = {m["name"]: m["labels"] for m in json.load(open(f"{I}/p508_labels.json"))}
    # —— 冻结读出：dev 训练（P5.02 同款）——
    man = rd(f"{ROOT}/data/interim/p303/emb_disorder_dev/extract_manifest.tsv")
    d = f"{ROOT}/data/interim/p303/emb_disorder_dev"
    key2arr = {}
    for row in man:
        if row["key"] not in key2arr:
            with np.load(os.path.join(d, row["key"] + ".npz"), allow_pickle=False) as z:
                key2arr[row["key"]] = {k: z[k] for k in z.files if k != "meta"}
    dev_store = {row["name"]: key2arr[row["key"]] for row in man}
    if len(dev_store) != 2279:
        die(f"dev 宇宙断言失败 {len(dev_store)}")
    ridx_dev = {r["name"]: [int(x) for x in r["resid_indices"].split(",")]
                for r in rd(f"{ROOT}/data/interim/p303/disorder_resid_idx.tsv")}
    state = defaultdict(dict)
    for r in rd(f"{ROOT}/data/curated/disorder_masks.tsv"):
        if r["state"] in ("0", "1") and r["mask"] == "1":
            for p in range(int(r["start"]), int(r["end"]) + 1):
                state[r["disprot_id"]][p] = int(r["state"])
    Xl, yl = [], []
    for nm in sorted(dev_store):
        pos_idx = [i for i, p in enumerate(ridx_dev[nm]) if p in state[nm]]
        Xl.append(dev_store[nm]["resid_layers"][0].astype(np.float32)[np.array(pos_idx)])
        yl.append([state[nm][ridx_dev[nm][i]] for i in pos_idx])
    clf = LogisticRegression(C=0.01, class_weight="balanced", solver="liblinear",
                             max_iter=1000, random_state=SEED)
    clf.fit(np.concatenate(Xl), np.array(sum(yl, [])))
    print("[p508] dev 读出拟合完成", flush=True)

    # —— 外部评估 ——
    man2 = rd(f"{I}/p508_emb/extract_manifest.tsv")
    d2 = f"{I}/p508_emb"
    key2 = {}
    for row in man2:
        if row["key"] not in key2:
            with np.load(os.path.join(d2, row["key"] + ".npz"), allow_pickle=False) as z:
                key2[row["key"]] = {k: z[k] for k in z.files if k != "meta"}
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
        sc = clf.predict_proba(ext[nm]["resid_layers"][0].astype(np.float32))[:, 1]
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
        "note": "补充分析；冻结读出零重拟合；主张级=逐蛋白 AUPRC 相对各自基率",
    }
    json.dump(metrics, open(f"{RES}/pooled_metrics.json", "w"), indent=1, ensure_ascii=False)
    print(json.dumps(metrics, ensure_ascii=False, indent=1))


def main():
    kept = step1_exclude()
    step2_embed(kept)
    step3_eval(kept)


if __name__ == "__main__":
    main()
