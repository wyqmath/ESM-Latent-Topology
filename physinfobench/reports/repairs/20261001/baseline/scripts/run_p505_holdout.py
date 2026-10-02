#!/usr/bin/env python3
"""P5.05 保留集泛化评估（集群侧，final_lock 冻结执行）：lock 校验 → 保留集表示抽取
→ H-KNOT-PRESENCE + H-DISORDER-RES 一次性评估（与确认侧同冻结读出/指标/对照；
读出仅在 development 拟合，不合并确认集）。

门禁：本脚本运行前须满足 final_lock.gate.user_authorization_recorded 非空
（用户对 final_holdout 解锁的明确授权登记，decisions.md 引用）——缺失即 die。
first_read 守卫：results/holdout/<run_id> 已存在=拒绝。
输出：results/holdout/{HOLD-KNOT-PRESENCE-P505-v1, HOLD-DISORDER-RES-P505-v1}/
     + holdout_qc.json + first_read_record.json（本运行器强制写入，P5.02 缺陷不再犯）。
用法：python scripts/run_p505_holdout.py --step all
"""
import argparse
import csv
import hashlib
import json
import os
import subprocess
import sys
from collections import defaultdict
from datetime import datetime, timezone

import numpy as np
import yaml
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run_p502_confirmation_cluster import die, load_emb, log, now_pair, rd, sha256

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RUN_ID_KNOT = "HOLD-KNOT-PRESENCE-P505-v1"
RUN_ID_DIS = "HOLD-DISORDER-RES-P505-v1"
SEED = 2026
BOOT_B = 2000
PERM_N = 100
H33_STORE_IDX = 2
LOCK = "configs/final_lock.yaml"
OUTBASE = os.path.join(ROOT, "results", "holdout")


def verify_final_lock():
    lock = yaml.safe_load(open(os.path.join(ROOT, LOCK)))
    for rel, want in lock["integrity"]["verify_files"].items():
        p = os.path.join(ROOT, rel)
        if not os.path.exists(p):
            die(f"final_lock 校验文件缺失: {rel}")
        if sha256(p) != want:
            die(f"final_lock 校验失败 {rel}")
    gate = lock["gate"]
    if not gate.get("user_authorization_recorded"):
        die("final_holdout 解锁门禁：无用户明确授权记录（final_lock.gate 为空）——禁止运行")
    log(f"final_lock 校验通过（{len(lock['integrity']['verify_files'])} 文件）；"
        f"解锁授权={gate['user_authorization_recorded']}")
    return lock


def first_read_guard(lock, run_id):
    outd = os.path.join(OUTBASE, run_id)
    if os.path.exists(outd):
        if os.listdir(outd):
            die(f"first_read 违规：{run_id} 结果已存在（禁覆盖、禁二次读取）")
        os.rmdir(outd)
    reg = lock["first_read"]["registered_at"]
    now = now_pair()
    if now["utc"] <= reg:
        die(f"first_read 时间语义错误：now {now['utc']} <= registered {reg}")
    os.makedirs(outd, exist_ok=True)
    return outd, now


def pooled_h33(store, name):
    return store[name]["resid_layers"][H33_STORE_IDX].astype(np.float32).mean(axis=0)


def extract(fa, out_dir, args):
    man = os.path.join(out_dir, "extract_manifest.tsv")
    if os.path.exists(man):
        log(f"抽取幂等跳过 {out_dir}")
        return
    os.makedirs(out_dir, exist_ok=True)
    cmd = [sys.executable, os.path.join(ROOT, "scripts", "extract_representations.py"),
           "--fasta", fa, "--out-dir", out_dir, "--device", "cuda",
           "--batch-max-tokens", "49152"] + args
    log("抽取: " + " ".join(cmd))
    subprocess.run(cmd, check=True, cwd=ROOT)


def run_knot(lock, records):
    outd, now = first_read_guard(lock, RUN_ID_KNOT)
    kdir = os.path.join(ROOT, "data", "interim", "p402", "emb_knots_resid")
    kX = load_emb(os.path.join(kdir, "extract_manifest.tsv"))
    kseq = {r["record_id"].lower(): r for r in rd(os.path.join(ROOT, "data/curated/knots_sequences.tsv"))}
    knots = {r["record_id"].lower(): r for r in rd(os.path.join(ROOT, "data/curated/knots.tsv"))}
    man_rows = rd(os.path.join(ROOT, "data/splits/split_manifest.tsv"))
    units = []
    for r in man_rows:
        if r["task_area"] == "knot" and r["split"] == "development":
            rec = r["sample_id"].split(":", 1)[1]
            s = kseq.get(rec.lower())
            if s is None or s["presence_target"] not in ("0", "1"):
                continue
            krec = knots.get(rec.lower())
            if krec is None or krec["presence_mask"] != "1" or krec["presence_target"] != s["presence_target"]:
                die(f"dev 链标签口径漂移: {rec}")
            if r["sample_id"] not in kX:
                die(f"dev 表示缺失: {r['sample_id']}")
            units.append({"sid": r["sample_id"], "len": int(s["len"]), "y": int(s["presence_target"])})
    if len(units) != 750:
        die(f"knots dev 宇宙断言失败 {len(units)} != 750")
    Xdev = np.stack([pooled_h33(kX, u["sid"]) for u in units])
    ydev = np.array([u["y"] for u in units])
    clf = LogisticRegression(C=1, class_weight="balanced", solver="liblinear",
                             max_iter=1000, random_state=SEED)
    clf.fit(Xdev, ydev)
    clf_len = LogisticRegression(C=1, class_weight="balanced", solver="liblinear",
                                 max_iter=1000, random_state=SEED)
    clf_len.fit(np.log1p(np.array([[u["len"]] for u in units], dtype=float)), ydev)

    conf_fa = os.path.join(ROOT, "data/interim/p504/knots_hold.fa")
    hold_names = [line[1:].strip() for line in open(conf_fa) if line.startswith(">")]
    cdir = os.path.join(ROOT, "data", "interim", "p505", "emb_knots_hold")
    cX = load_emb(os.path.join(cdir, "extract_manifest.tsv"))
    if sorted(cX) != sorted(hold_names):
        die(f"保留集表示集合不符 {len(cX)} vs fasta {len(hold_names)}（禁止静默子集）")
    Xc = np.stack([pooled_h33(cX, n) for n in hold_names])
    yc = np.array([int(knots[n.split(":", 1)[1].lower()]["presence_target"]) for n in hold_names])
    if yc.sum() != 25 or (1 - yc).sum() != 83:
        die(f"保留集标签构成断言失败 {int(yc.sum())}/{int((1 - yc).sum())} != 25/83")
    sc = clf.predict_proba(Xc)[:, 1]
    auroc = roc_auc_score(yc, sc)
    y_by_unit = {hold_names[i]: int(yc[i]) for i in range(len(hold_names))}

    def auroc_stat(sample):
        labs = np.array([y_by_unit[u] for u in sample])
        scs = np.array([sample[u][0] for u in sample])
        if len(set(labs.tolist())) < 2:
            return float("nan")
        return roc_auc_score(labs, scs)

    by_unit = {hold_names[i]: [sc[i]] for i in range(len(hold_names))}
    rng = np.random.RandomState(SEED)
    units_n = list(by_unit)
    n = len(units_n)
    draws, degen = [], 0
    for _ in range(BOOT_B):
        pick = rng.randint(0, n, size=n)
        sample = defaultdict(list)
        for i in pick:
            sample[units_n[i]].extend(by_unit[units_n[i]])
        if len(sample) < 2:
            degen += 1
            continue
        draws.append(auroc_stat(sample))
    ok = np.array([d for d in draws if not np.isnan(d)])
    boot = {"point": round(float(auroc), 6),
            "ci95_low": round(float(np.percentile(ok, 2.5)), 6),
            "ci95_high": round(float(np.percentile(ok, 97.5)), 6),
            "valid_draws": int(len(ok)), "degenerate_discarded": int(degen),
            "no_conclusion_flag": bool(len(ok) < 1000)}
    perm = []
    for i in range(PERM_N):
        prng = np.random.RandomState(SEED + i)
        perm.append(float(roc_auc_score(prng.permutation(yc), sc)))
    conf_len = np.log1p(np.array(
        [[int(kseq[nm.split(":", 1)[1].lower()]["len"])] for nm in hold_names], dtype=float))
    len_sc = clf_len.predict_proba(conf_len)[:, 1]
    metrics = {"run_id": RUN_ID_KNOT, "hypothesis": "H-KNOT-PRESENCE", "executed_at": now,
               "reader": {"layer_hidden": 33, "C": 1, "class_weight": "balanced",
                          "solver": "liblinear", "max_iter": 1000, "random_state": SEED},
               "dev_fit": {"n": 750, "pos": int(ydev.sum()), "neg": int((1 - ydev).sum()),
                           "merge_confirmation": "no（final_lock 冻结：仅 dev 拟合）"},
               "holdout": {"n": len(hold_names), "pos": int(yc.sum()), "neg": int((1 - yc).sum()),
                           "no_sequence_dropped": ["knot:3pvm_B"]},
               "auroc": round(float(auroc), 6), "auroc_boot": boot,
               "permutation_null": {"n": PERM_N, "mean": round(float(np.mean(perm)), 6),
                                    "sd": round(float(np.std(perm)), 6),
                                    "max": round(float(np.max(perm)), 6)},
               "length_control_auroc": round(float(roc_auc_score(yc, len_sc)), 6),
               "statistics": {"B": BOOT_B, "seed": SEED, "ci": "percentile95"}}
    with open(os.path.join(outd, "scores.tsv"), "w", newline="") as f:
        w = csv.writer(f, delimiter="\t", lineterminator="\n")
        w.writerow(["sample_id", "label", "score"])
        w.writerows(zip(hold_names, yc.tolist(), [f"{v:.6g}" for v in sc]))
    json.dump(metrics, open(os.path.join(outd, "metrics.json"), "w"), indent=1, sort_keys=True)
    log(f"H-KNOT-PRESENCE(holdout): AUROC={metrics['auroc']} "
        f"CI95=[{boot['ci95_low']},{boot['ci95_high']}]")
    records.append({"run_id": RUN_ID_KNOT, "first_read": now,
                    "content": "knots holdout 108（25 阳/83 阴）presence_target + 指标"})
    return {"run_id": RUN_ID_KNOT, "metrics": metrics}


def run_disorder(lock, records):
    outd, now = first_read_guard(lock, RUN_ID_DIS)
    ddir = os.path.join(ROOT, "data", "interim", "p303", "emb_disorder_dev")
    Xd_store = load_emb(os.path.join(ddir, "extract_manifest.tsv"))
    if len(Xd_store) != 2279:
        die(f"disorder dev 宇宙断言失败 {len(Xd_store)} != 2279")
    ridx_dev = {r["name"]: [int(x) for x in r["resid_indices"].split(",")]
                for r in rd(os.path.join(ROOT, "data/interim/p303/disorder_resid_idx.tsv"))}
    state_by_res = defaultdict(dict)
    for r in rd(os.path.join(ROOT, "data/curated/disorder_masks.tsv")):
        if r["state"] in ("0", "1") and r["mask"] == "1":
            for p in range(int(r["start"]), int(r["end"]) + 1):
                state_by_res[r["disprot_id"]][p] = int(r["state"])
    man_rows = rd(os.path.join(ROOT, "data/splits/split_manifest.tsv"))
    dev_ids = {r["sample_id"] for r in man_rows if r["split"] == "development"}
    Xl, yl = [], []
    for nm in sorted(Xd_store):
        if f"disorder:{nm}" not in dev_ids:
            die(f"非 dev 样本混入 dev 拟合: {nm}")
        pos_idx = [i for i, p in enumerate(ridx_dev[nm]) if p in state_by_res[nm]]
        yv = np.array([state_by_res[nm][ridx_dev[nm][i]] for i in pos_idx], dtype=int)
        Xl.append(Xd_store[nm]["resid_layers"][H33_STORE_IDX].astype(np.float32)[np.array(pos_idx)])
        yl.append(yv)
    Xdev = np.concatenate(Xl)
    ydev = np.concatenate(yl)
    log(f"dev 拟合：n_residues={len(ydev)} pos_rate={ydev.mean():.4f}")
    clf = LogisticRegression(C=0.01, class_weight="balanced", solver="liblinear",
                             max_iter=1000, random_state=SEED)
    clf.fit(Xdev, ydev)

    conf_fa = os.path.join(ROOT, "data/interim/p504/disorder_hold.fa")
    hold_names = [line[1:].strip() for line in open(conf_fa) if line.startswith(">")]
    ridx = {r["name"]: [int(x) for x in r["resid_indices"].split(",")]
            for r in rd(os.path.join(ROOT, "data/interim/p504/disorder_hold_resid_idx.tsv"))}
    cdir = os.path.join(ROOT, "data", "interim", "p505", "emb_disorder_hold")
    cX = load_emb(os.path.join(cdir, "extract_manifest.tsv"))
    if sorted(cX) != sorted(hold_names):
        die(f"保留集表示集合不符 {len(cX)} vs fasta {len(hold_names)}")
    prot, xs, ys = {}, [], []
    for nm in hold_names:
        idx = ridx[nm]
        pos_idx = [i for i, p in enumerate(idx) if p in state_by_res[nm]]
        yv = np.array([state_by_res[nm][idx[i]] for i in pos_idx], dtype=int)
        feats = cX[nm]["resid_layers"][H33_STORE_IDX].astype(np.float32)[np.array(pos_idx)]
        if len(yv) != feats.shape[0] or len(yv) == 0:
            die(f"保留蛋白 {nm} 域对齐失败")
        sc = clf.predict_proba(feats)[:, 1]
        prot[nm] = {"y": yv.tolist(), "score": [round(float(v), 6) for v in sc]}
        xs.append(sc)
        ys.append(yv)
    Xc = np.concatenate(xs)
    yc = np.concatenate(ys)
    pooled = average_precision_score(yc, Xc)
    prev = float(yc.mean())
    names = list(prot)

    def pooled_from(units_sel):
        yy = np.concatenate([np.array(prot[u]["y"]) for u in units_sel])
        ss = np.concatenate([np.array(prot[u]["score"]) for u in units_sel])
        if yy.sum() == 0 or len(yy) < 2:
            return float("nan")
        return average_precision_score(yy, ss)

    rng = np.random.RandomState(SEED)
    draws, degen = [], 0
    nn = len(names)
    for _ in range(BOOT_B):
        pick = rng.randint(0, nn, size=nn)
        units_sel = [names[i] for i in pick]
        if len(set(units_sel)) < 2:
            degen += 1
            continue
        draws.append(pooled_from(units_sel))
    ok = np.array([d for d in draws if not np.isnan(d)])
    boot = {"point": round(float(pooled), 6),
            "ci95_low": round(float(np.percentile(ok, 2.5)), 6),
            "ci95_high": round(float(np.percentile(ok, 97.5)), 6),
            "valid_draws": int(len(ok)), "degenerate_discarded": int(degen),
            "no_conclusion_flag": bool(len(ok) < 1000)}
    perm = []
    for i in range(PERM_N):
        prng = np.random.RandomState(SEED + i)
        yp = np.concatenate([prng.permutation(np.array(prot[u]["y"])) for u in names])
        perm.append(float(average_precision_score(yp, Xc)))
    dual = [u for u in names if len(set(prot[u]["y"])) == 2]
    per_prot = {u: round(float(average_precision_score(np.array(prot[u]["y"]),
                                                       np.array(prot[u]["score"]))), 6)
                for u in dual}
    metrics = {"run_id": RUN_ID_DIS, "hypothesis": "H-DISORDER-RES", "executed_at": now,
               "reader": {"layer_hidden": 33, "C": 0.01, "class_weight": "balanced",
                          "solver": "liblinear", "max_iter": 1000, "random_state": SEED},
               "dev_fit": {"n_proteins": len(Xd_store), "n_residues": int(len(ydev)),
                           "pos_rate": round(float(ydev.mean()), 6),
                           "merge_confirmation": "no（final_lock 冻结：仅 dev 拟合）"},
               "holdout": {"n_proteins": len(hold_names), "n_residues": int(len(yc)),
                           "pos_rate": round(prev, 6), "dual_class_proteins": len(dual)},
               "pooled_auprc": round(float(pooled), 6), "pooled_auprc_boot": boot,
               "base_rate_prevalence": round(prev, 6),
               "permutation_null": {"n": PERM_N, "mean": round(float(np.mean(perm)), 6),
                                    "sd": round(float(np.std(perm)), 6),
                                    "max": round(float(np.max(perm)), 6)},
               "per_protein_auprc_descriptive": per_prot,
               "degeneracy_note": "确认侧已示警：双类蛋白计数为判读前置量，如实报告",
               "statistics": {"B": BOOT_B, "seed": SEED, "ci": "percentile95"}}
    import gzip
    with gzip.open(os.path.join(outd, "residue_scores.tsv.gz"), "wt", newline="") as f:
        w = csv.writer(f, delimiter="\t", lineterminator="\n")
        w.writerow(["protein", "resid_1based", "label", "score"])
        for u in names:
            idx = ridx[u]
            for i, (yv, sv) in enumerate(zip(prot[u]["y"], prot[u]["score"])):
                w.writerow([u, idx[i], yv, f"{sv:.6g}"])
    json.dump(metrics, open(os.path.join(outd, "metrics.json"), "w"), indent=1, sort_keys=True)
    log(f"H-DISORDER-RES(holdout): pooled AUPRC={metrics['pooled_auprc']} prev={prev:.4f} "
        f"dual={len(dual)}")
    records.append({"run_id": RUN_ID_DIS, "first_read": now,
                    "content": "disorder holdout 494 蛋白分母域残基 state + 指标"})
    return {"run_id": RUN_ID_DIS, "metrics": metrics}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--step", default="all", choices=["extract", "evaluate", "all"])
    args = ap.parse_args()
    lock = verify_final_lock()
    records = []
    if args.step in ("extract", "all"):
        extract(os.path.join(ROOT, "data/interim/p504/knots_hold.fa"),
                os.path.join(ROOT, "data/interim/p505/emb_knots_hold"),
                ["--resid-layers", "23,29,33"])
        extract(os.path.join(ROOT, "data/interim/p504/disorder_hold.fa"),
                os.path.join(ROOT, "data/interim/p505/emb_disorder_hold"),
                ["--resid-layers", "11,23,33",
                 "--resid-index-file", os.path.join(ROOT, "data/interim/p504/disorder_hold_resid_idx.tsv")])
        log("保留集抽取完成（knots 108 全链 / disorder 494 蛋白分母域）")
    if args.step in ("evaluate", "all"):
        res_k = run_knot(lock, records)
        res_d = run_disorder(lock, records)
        merged = {"executed_at": now_pair(), "runs": [res_k["run_id"], res_d["run_id"]],
                  "knots": res_k["metrics"], "disorder": res_d["metrics"],
                  "one_shot_note": "全部已注册对比同批一次计算；无中途选择、无赢家重跑"}
        os.makedirs(OUTBASE, exist_ok=True)
        json.dump(merged, open(os.path.join(OUTBASE, "holdout_qc.json"), "w"),
                  ensure_ascii=False, indent=1, sort_keys=True)
        fs_rec = lock["gate"].get("fsl2_holdout_decision")
        json.dump({"first_read_record": records,
                   "registered_at": lock["first_read"]["registered_at"],
                   "gate": lock["gate"],
                   "fsl2_holdout_decision": fs_rec},
                  open(os.path.join(OUTBASE, "first_read_record.json"), "w"),
                  indent=1, ensure_ascii=False, sort_keys=True)
    log("DONE")


if __name__ == "__main__":
    main()
