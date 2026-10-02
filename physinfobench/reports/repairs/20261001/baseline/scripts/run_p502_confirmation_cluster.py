#!/usr/bin/env python3
"""P5.02 确认集复核（集群侧）：lock 校验 → 确认集表示抽取 → H-KNOT-PRESENCE +
H-DISORDER-RES 一次性评估（无中途选择、无重跑赢家）。

纪律（P5.01 confirmation_lock 钉版）：
  - 运行前逐文件校验 lock.integrity.verify_files 的 sha256，不一致即停止；
  - first_read 强制：results/confirmation/<run_id> 已存在=拒绝（禁覆盖、禁二次读取）；
  - 读取器=冻结配置一次性重拟合（仅 development），确认标签只在指标计算时消费；
  - 禁止静默子集：确认集任何样本表示缺失即 die；dev 宇宙计数断言。
输出：results/confirmation/{CONF-KNOT-PRESENCE-P502-v1, CONF-DISORDER-RES-P502-v1}/
     + confirmation_qc.json + first_read_record.json + exposure_rows_pending.tsv
用法：python scripts/run_p502_confirmation_cluster.py --step all  (extract|evaluate|all)
"""
import argparse
import csv
import gzip
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

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RUN_ID_KNOT = "CONF-KNOT-PRESENCE-P502-v1"
RUN_ID_DIS = "CONF-DISORDER-RES-P502-v1"
SEED = 2026
BOOT_B = 2000
PERM_N = 100
RESID_IDX = {"knots": [23, 29, 33], "disorder": [11, 23, 33]}
H33_STORE_IDX = 2  # resid_layers[2] = hidden 33（1-based 第 33 层）


def die(m):
    print(f"[p502 FATAL] {m}", file=sys.stderr)
    sys.exit(1)


def log(m):
    print(f"[p502] {m}", flush=True)


def rd(p, d="\t"):
    with open(p, newline="") as f:
        return list(csv.DictReader(f, delimiter=d))


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def now_pair():
    return {"utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "local": datetime.now().strftime("%Y-%m-%d %H:%M")}


def load_emb(manifest_path):
    """name→arrays（extract_manifest name→key 解析；同 P3.03/P4.02 语义）。"""
    man = rd(manifest_path)
    d = os.path.dirname(manifest_path)
    key2arr = {}
    for row in man:
        key = row["key"]
        if key not in key2arr:
            with np.load(os.path.join(d, key + ".npz"), allow_pickle=False) as z:
                key2arr[key] = {k: z[k] for k in z.files if k != "meta"}
    return {row["name"]: key2arr[row["key"]] for row in man}


def verify_lock():
    lock = yaml.safe_load(open(os.path.join(ROOT, "configs", "confirmation_lock.yaml")))
    if lock["meta"]["status"] != os.environ.get("P502_EXPECT_STATUS", lock["meta"]["status"]):
        die("lock status 与预期不符")
    for rel, want in lock["integrity"]["verify_files"].items():
        p = os.path.join(ROOT, rel)
        if not os.path.exists(p):
            die(f"lock 校验文件缺失: {rel}")
        got = sha256(p)
        if got != want:
            die(f"lock 校验失败 {rel}: {got} != {want}")
    log(f"lock 校验通过（{len(lock['integrity']['verify_files'])} 文件）")
    return lock


def first_read_guard(lock, run_id):
    outd = os.path.join(ROOT, "results", "confirmation", run_id)
    if os.path.exists(outd):
        if os.listdir(outd):
            die(f"first_read 违规：{run_id} 结果已存在（禁覆盖、禁二次读取）")
        os.rmdir(outd)  # 前次中断遗留的空目录：清掉后重建，不毒化重跑
    reg = lock["first_read"]["registered_at"]
    now = now_pair()
    if now["utc"] <= reg:
        die(f"first_read 时间语义错误：now {now['utc']} <= registered {reg}")
    os.makedirs(outd, exist_ok=True)
    return outd, now


def pooled_h33(store, name):
    return store[name]["resid_layers"][H33_STORE_IDX].astype(np.float32).mean(axis=0)


def boot_ci(values_by_unit, stat_fn, rng):
    """protein 单位重采样 bootstrap（B=2000，percentile 95% CI；退化抽取丢弃并计数）。"""
    units = list(values_by_unit)
    n = len(units)
    point = stat_fn(values_by_unit)
    draws, degenerate = [], 0
    for _ in range(BOOT_B):
        pick = rng.randint(0, n, size=n)
        sample = defaultdict(list)
        for i in pick:
            sample[units[i]].extend(values_by_unit[units[i]])
        if len(sample) < 2:
            degenerate += 1
            continue
        draws.append(stat_fn(sample))
    draws = np.array(draws, dtype=float)
    ok = draws[~np.isnan(draws)]
    return {"point": round(float(point), 6),
            "ci95_low": round(float(np.percentile(ok, 2.5)), 6),
            "ci95_high": round(float(np.percentile(ok, 97.5)), 6),
            "valid_draws": int(len(ok)), "degenerate_discarded": int(degenerate),
            "no_conclusion_flag": bool(len(ok) < 1000)}


def extract(step_ok, fa, out_dir, args):
    man = os.path.join(out_dir, "extract_manifest.tsv")
    if os.path.exists(man) and step_ok:
        log(f"抽取幂等跳过 {out_dir}（manifest 已存在）")
        return
    os.makedirs(out_dir, exist_ok=True)
    cmd = [sys.executable, os.path.join(ROOT, "scripts", "extract_representations.py"),
           "--fasta", fa, "--out-dir", out_dir, "--device", "cuda",
           "--batch-max-tokens", "49152"] + args
    log("抽取: " + " ".join(cmd))
    subprocess.run(cmd, check=True, cwd=ROOT)


def run_h_knot(lock, pending):
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
            if krec is None or krec["presence_mask"] != "1":
                die(f"dev 链 presence_mask 口径漂移: {rec}")
            if krec["presence_target"] != s["presence_target"]:
                die(f"标签源冲突 knots.tsv vs knots_sequences: {rec}")
            if r["sample_id"] not in kX:
                die(f"dev 表示缺失（禁止静默子集）: {r['sample_id']}")
            units.append({"sid": r["sample_id"], "len": int(s["len"]),
                          "y": int(s["presence_target"])})
    if len(units) != 750:
        die(f"knots dev 宇宙断言失败 {len(units)} != 750")
    Xdev = np.stack([pooled_h33(kX, u["sid"]) for u in units])
    ydev = np.array([u["y"] for u in units])
    log(f"dev 拟合：n={len(units)} pos={int(ydev.sum())} dim={Xdev.shape[1]}")

    clf = LogisticRegression(C=1, class_weight="balanced", solver="liblinear",
                             max_iter=1000, random_state=SEED)
    clf.fit(Xdev, ydev)
    clf_len = LogisticRegression(C=1, class_weight="balanced", solver="liblinear",
                                 max_iter=1000, random_state=SEED)
    clf_len.fit(np.log1p(np.array([[u["len"]] for u in units], dtype=float)), ydev)

    conf_fa = os.path.join(ROOT, "data/interim/p501/knots_conf.fa")
    conf_names = [line[1:].strip() for line in open(conf_fa) if line.startswith(">")]
    cdir = os.path.join(ROOT, "data", "interim", "p502", "emb_knots_conf")
    cX = load_emb(os.path.join(cdir, "extract_manifest.tsv"))
    if sorted(cX) != sorted(conf_names):
        die(f"确认集表示集合不符 {len(cX)} vs fasta {len(conf_names)}（禁止静默子集）")
    Xc = np.stack([pooled_h33(cX, n) for n in conf_names])
    yc = []
    for n in conf_names:
        rec = n.split(":", 1)[1]
        yc.append(int(knots[rec.lower()]["presence_target"]))
    yc = np.array(yc)
    if yc.sum() != 22 or (1 - yc).sum() != 126:
        die(f"确认集标签构成断言失败 pos={int(yc.sum())} neg={int((1 - yc).sum())} != 22/126")

    sc = clf.predict_proba(Xc)[:, 1]
    auroc = roc_auc_score(yc, sc)
    by_unit = {conf_names[i]: [sc[i]] for i in range(len(conf_names))}
    y_by_unit = {conf_names[i]: int(yc[i]) for i in range(len(conf_names))}

    def auroc_stat(sample):
        labs, scs = [], []
        for u, v in sample.items():
            labs.append(y_by_unit[u])
            scs.append(v[0])
        labs = np.array(labs)
        if len(set(labs.tolist())) < 2:
            return float("nan")
        return roc_auc_score(labs, np.array(scs))

    rng = np.random.RandomState(SEED)
    boot = boot_ci(by_unit, auroc_stat, rng)

    perm_scores = []
    for i in range(PERM_N):
        prng = np.random.RandomState(SEED + i)
        yp = prng.permutation(yc)
        perm_scores.append(float(roc_auc_score(yp, sc)))
    conf_len = np.log1p(np.array(
        [[int(kseq[n.split(":", 1)[1].lower()]["len"])] for n in conf_names], dtype=float))
    len_sc = clf_len.predict_proba(conf_len)[:, 1]
    metrics = {
        "run_id": RUN_ID_KNOT, "hypothesis": "H-KNOT-PRESENCE", "executed_at": now,
        "reader": {"layer_hidden": 33, "C": 1, "class_weight": "balanced",
                   "solver": "liblinear", "max_iter": 1000, "random_state": SEED,
                   "features": "hidden33 全残基均值（knots_resid 同协议；dev/确认同源）"},
        "dev_fit": {"n": 750, "pos": int(ydev.sum()), "neg": int((1 - ydev).sum())},
        "conf": {"n": len(conf_names), "pos": int(yc.sum()), "neg": int((1 - yc).sum()),
                 "no_sequence_dropped": ["knot:1giy_M", "knot:2hfx_A"]},
        "auroc": round(float(auroc), 6),
        "auroc_boot": boot,
        "permutation_null": {"n": PERM_N, "seeds": [SEED, SEED + PERM_N - 1],
                             "mean": round(float(np.mean(perm_scores)), 6),
                             "sd": round(float(np.std(perm_scores)), 6),
                             "min": round(float(np.min(perm_scores)), 6),
                             "max": round(float(np.max(perm_scores)), 6)},
        "length_control_auroc": round(float(roc_auc_score(yc, len_sc)), 6),
        "statistics": {"B": BOOT_B, "seed": SEED, "ci": "percentile95",
                       "units": "chain（蛋白级）"},
    }
    with open(os.path.join(outd, "scores.tsv"), "w", newline="") as f:
        w = csv.writer(f, delimiter="\t", lineterminator="\n")
        w.writerow(["sample_id", "label", "score"])
        w.writerows(zip(conf_names, yc.tolist(), [f"{s:.6g}" for s in sc]))
    json.dump(metrics, open(os.path.join(outd, "metrics.json"), "w"), indent=1, sort_keys=True)
    log(f"H-KNOT-PRESENCE: AUROC={metrics['auroc']} CI95=[{boot['ci95_low']},{boot['ci95_high']}]")
    pending.append(["first_read_result_level", "knots conf 148（22 阳/126 阴）presence_target + 指标",
                    RUN_ID_KNOT])
    return {"run_id": RUN_ID_KNOT, "out": outd, "metrics": metrics}


def run_h_disorder(lock, pending):
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

    Xl, yl, prot_meta = [], [], {}
    for n in sorted(Xd_store):
        if f"disorder:{n}" not in dev_ids:
            die(f"非 dev 样本混入 dev 拟合: {n}")
        pos_idx = [i for i, p in enumerate(ridx_dev[n]) if p in state_by_res[n]]
        yv = np.array([state_by_res[n][ridx_dev[n][i]] for i in pos_idx], dtype=int)
        feats = Xd_store[n]["resid_layers"][H33_STORE_IDX].astype(np.float32)[np.array(pos_idx)]
        Xl.append(feats)
        yl.append(yv)
        prot_meta[n] = {"n": len(pos_idx), "pos_rate": round(float(yv.mean()), 6)}
    Xdev = np.concatenate(Xl)
    ydev = np.concatenate(yl)
    log(f"dev 拟合：n_residues={len(ydev)} pos_rate={ydev.mean():.4f} dim={Xdev.shape[1]}")
    clf = LogisticRegression(C=0.01, class_weight="balanced", solver="liblinear",
                             max_iter=1000, random_state=SEED)
    clf.fit(Xdev, ydev)

    conf_fa = os.path.join(ROOT, "data/interim/p501/disorder_conf.fa")
    conf_names = [line[1:].strip() for line in open(conf_fa) if line.startswith(">")]
    ridx_conf = {r["name"]: [int(x) for x in r["resid_indices"].split(",")]
                 for r in rd(os.path.join(ROOT, "data/interim/p501/disorder_conf_resid_idx.tsv"))}
    cdir = os.path.join(ROOT, "data", "interim", "p502", "emb_disorder_conf")
    cX = load_emb(os.path.join(cdir, "extract_manifest.tsv"))
    if sorted(cX) != sorted(conf_names):
        die(f"确认集表示集合不符 {len(cX)} vs fasta {len(conf_names)}（禁止静默子集）")
    conf_prot, xs, ys = {}, [], []
    for n in conf_names:
        idx = ridx_conf[n]
        pos_idx = [i for i, p in enumerate(idx) if p in state_by_res[n]]
        yv = np.array([state_by_res[n][idx[i]] for i in pos_idx], dtype=int)
        feats = cX[n]["resid_layers"][H33_STORE_IDX].astype(np.float32)[np.array(pos_idx)]
        if len(yv) != feats.shape[0] or len(yv) == 0:
            die(f"确认蛋白 {n} 域对齐失败")
        sc = clf.predict_proba(feats)[:, 1]
        conf_prot[n] = {"y": yv.tolist(), "score": [round(float(v), 6) for v in sc]}
        xs.append(sc)
        ys.append(yv)
    Xc = np.concatenate(xs)
    yc = np.concatenate(ys)
    pooled = average_precision_score(yc, Xc)
    prev = float(yc.mean())

    # protein 单位 bootstrap：按蛋白重采样后重组池化 AUPRC
    names = list(conf_prot)

    def pooled_from(units):
        yy = np.concatenate([np.array(conf_prot[u]["y"]) for u in units])
        ss = np.concatenate([np.array(conf_prot[u]["score"]) for u in units])
        if yy.sum() == 0 or len(yy) < 2:
            return float("nan")
        return average_precision_score(yy, ss)

    rng = np.random.RandomState(SEED)
    draws, degen = [], 0
    nn = len(names)
    for _ in range(BOOT_B):
        pick = rng.randint(0, nn, size=nn)
        units = [names[i] for i in pick]
        if len(set(units)) < 2:
            degen += 1
            continue
        draws.append(pooled_from(units))
    draws = np.array([d for d in draws if not np.isnan(d)])
    boot = {"point": round(float(pooled), 6),
            "ci95_low": round(float(np.percentile(draws, 2.5)), 6),
            "ci95_high": round(float(np.percentile(draws, 97.5)), 6),
            "valid_draws": int(len(draws)), "degenerate_discarded": int(degen),
            "no_conclusion_flag": bool(len(draws) < 1000)}

    perm = []
    for i in range(PERM_N):
        prng = np.random.RandomState(SEED + i)
        yp = np.concatenate([prng.permutation(np.array(conf_prot[u]["y"])) for u in names])
        perm.append(float(average_precision_score(yp, Xc)))
    dual = [u for u in names if len(set(conf_prot[u]["y"])) == 2]
    per_prot = {u: round(float(average_precision_score(np.array(conf_prot[u]["y"]),
                                                       np.array(conf_prot[u]["score"]))), 6)
                for u in dual}
    metrics = {
        "run_id": RUN_ID_DIS, "hypothesis": "H-DISORDER-RES", "executed_at": now,
        "reader": {"layer_hidden": 33, "C": 0.01, "class_weight": "balanced",
                   "solver": "liblinear", "max_iter": 1000, "random_state": SEED,
                   "features": "hidden33 逐残基（分母域，resid-index 协议同 dev 抽取）"},
        "dev_fit": {"n_proteins": len(Xd_store), "n_residues": int(len(ydev)),
                    "pos_rate": round(float(ydev.mean()), 6)},
        "conf": {"n_proteins": len(conf_names), "n_residues": int(len(yc)),
                 "pos_rate": round(prev, 6), "dual_class_proteins": len(dual)},
        "pooled_auprc": round(float(pooled), 6),
        "pooled_auprc_boot": boot,
        "base_rate_prevalence": round(prev, 6),
        "permutation_null": {"n": PERM_N, "seeds": [SEED, SEED + PERM_N - 1],
                             "mean": round(float(np.mean(perm)), 6),
                             "sd": round(float(np.std(perm)), 6),
                             "max": round(float(np.max(perm)), 6)},
        "per_protein_auprc_descriptive": per_prot,
        "statistics": {"B": BOOT_B, "seed": SEED, "ci": "percentile95",
                       "units": "protein（残基池化随蛋白重采样）"},
    }
    with gzip.open(os.path.join(outd, "residue_scores.tsv.gz"), "wt", newline="") as f:
        w = csv.writer(f, delimiter="\t", lineterminator="\n")
        w.writerow(["protein", "resid_1based", "label", "score"])
        for u in names:
            idx = ridx_conf[u]
            for i, (yv, sv) in enumerate(zip(conf_prot[u]["y"], conf_prot[u]["score"])):
                w.writerow([u, idx[i], yv, f"{sv:.6g}"])
    json.dump(metrics, open(os.path.join(outd, "metrics.json"), "w"), indent=1, sort_keys=True)
    log(f"H-DISORDER-RES: pooled AUPRC={metrics['pooled_auprc']} "
        f"CI95=[{boot['ci95_low']},{boot['ci95_high']}] prev={prev:.4f}")
    pending.append(["first_read_result_level", "disorder conf 432 蛋白分母域残基 state/mask + 指标",
                    RUN_ID_DIS])
    return {"run_id": RUN_ID_DIS, "out": outd, "metrics": metrics}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--step", default="all", choices=["extract", "evaluate", "all"])
    args = ap.parse_args()
    lock = verify_lock()
    pending_path = os.path.join(ROOT, "results", "confirmation", "exposure_rows_pending.tsv")
    pending = []
    if os.path.exists(pending_path):
        pending = [r for r in rd(pending_path)]

    if args.step in ("extract", "all"):
        extract(True, os.path.join(ROOT, "data/interim/p501/knots_conf.fa"),
                os.path.join(ROOT, "data/interim/p502/emb_knots_conf"),
                ["--resid-layers", "23,29,33"])
        extract(True, os.path.join(ROOT, "data/interim/p501/disorder_conf.fa"),
                os.path.join(ROOT, "data/interim/p502/emb_disorder_conf"),
                ["--resid-layers", "11,23,33",
                 "--resid-index-file", os.path.join(ROOT, "data/interim/p501/disorder_conf_resid_idx.tsv")])
        extract(True, os.path.join(ROOT, "data/interim/p501/fs_conf_endpoints.fa"),
                os.path.join(ROOT, "data/interim/p502/emb_fs_conf_endpoints"),
                ["--mean-layers", "5,11,17,23,29,33"])
        log("确认集抽取完成（knots 148 全链 / disorder 分母域 / FS 端点 12）")
    if args.step in ("evaluate", "all"):
        res_k = run_h_knot(lock, pending)
        res_d = run_h_disorder(lock, pending)
        merged = {"executed_at": now_pair(), "runs": [res_k["run_id"], res_d["run_id"]],
                  "knots": res_k["metrics"], "disorder": res_d["metrics"],
                  "one_shot_note": "全部已注册对比同批一次计算；无中途选择、无赢家重跑"}
        os.makedirs(os.path.join(ROOT, "results", "confirmation"), exist_ok=True)
        json.dump(merged, open(os.path.join(ROOT, "results/confirmation/confirmation_qc.json"),
                               "w"), ensure_ascii=False, indent=1, sort_keys=True)
    os.makedirs(os.path.dirname(pending_path), exist_ok=True)
    with open(pending_path, "w", newline="") as f:
        w = csv.writer(f, delimiter="\t", lineterminator="\n")
        w.writerow(["action", "content_read", "run_id", "recorded_at_utc"])
        for row in pending:
            w.writerow(list(row) + [now_pair()["utc"]])
    log("DONE")


if __name__ == "__main__":
    main()
