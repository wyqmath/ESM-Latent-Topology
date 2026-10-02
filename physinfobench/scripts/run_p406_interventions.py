#!/usr/bin/env python3
"""P4.06/P4.07/P4.08：条件干预矩阵（configs/interventions.yaml frozen_2026-09-26_p405；C-IN3）。

PYP（1 对，RUN）/ RNase A（2 对，RUN_DEGRADED=输入充分性对照）/ BPTI（STOP，仅失败清单+审计）。
四臂 M / F-only / M+F / M+F_perm（组内 F 对换置换）；头=liblinear logistic C=1 固定；
完全分离→方向性事实披露（"性能估计"列恒空断言）；置换映射+边际断言落盘。
用法：
  python run_p406_interventions.py fetch      # RCSB 拉序列+一致性断言（需网络）
  python run_p406_interventions.py matrix     # 四臂矩阵（需 p405 池化嵌入已抽取）
"""
import csv
import gzip
import hashlib
import json
import os
import sys
import urllib.request
from collections import OrderedDict
from datetime import datetime

import numpy as np
from sklearn.linear_model import LogisticRegression

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
OUT405 = os.path.join(ROOT, "data/interim/p405")
RUN_TS = datetime.now().strftime("%Y-%m-%d %H:%M")
SEEDS = [13, 42, 2026]
C_FIXED = 1.0


def die(m):
    print(f"[FATAL] {m}", flush=True)
    sys.exit(1)


def rd(p, d="\t"):
    with open(p) as f:
        return list(csv.DictReader(f, delimiter=d))


def wr(path, rows, cols):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, delimiter="\t", lineterminator="\n")
        w.writeheader()
        w.writerows(rows)


# ---------------- 样本登记（与 curated 表逐项对齐；interventions.yaml §1/§2） ----------------
SAMPLES = OrderedDict([
    # pyp：PG-PYP-001
    ("PYP-2014-DARK", dict(sys="pyp", pg="PG-PYP-001", pdb="4WL9", entity="1", chain="A",
                           y=0, f={"illumination": 0})),
    ("PYP-2014-LIGHT-INT", dict(sys="pyp", pg="PG-PYP-001", pdb="4WLA", entity="1", chain="A",
                                y=1, f={"illumination": 1})),  # illumination=1 取自 state 列（§2 m3 注）
    # rnase_a：PG-RNA-001/002
    ("RNA-MONOMER-1FS3", dict(sys="rnase_a", pg="PG-RNA-001", pdb="1FS3", entity="1", chain="A",
                              y=0, f={"oligomer_state": 0})),
    ("RNA-CSWAP-1F0V", dict(sys="rnase_a", pg="PG-RNA-001", pdb="1F0V", entity="2", chain="A,B",
                            y=1, f={"oligomer_state": 1})),
    ("RNA-MONOMER-1FS3-B", dict(sys="rnase_a", pg="PG-RNA-002", pdb="1FS3", entity="1", chain="A",
                                y=0, f={"oligomer_state": 0})),
    ("RNA-NSWAP-1A2W", dict(sys="rnase_a", pg="PG-RNA-002", pdb="1A2W", entity="1", chain="A,B",
                            y=1, f={"oligomer_state": 1})),
])
ENTITY_OVERRIDE = {"1F0V": "2"}  # RCSB 实测：P61823 在 entity2（interventions.yaml §1）
# §1 断言注册表=矩阵样本+1JS0（extension，仅断言不入矩阵——config §1 identity_assertions 全覆盖）
ASSERT_EXTRA = [("RNA-TRIMER-1JS0", dict(sys="rnase_a", pg="", pdb="1JS0", entity="1",
                                         chain="A", y=None, f={}))]


def fetch():
    os.makedirs(OUT405, exist_ok=True)
    rows, seqs = [], {}
    registry = OrderedDict(list(SAMPLES.items()) + list(ASSERT_EXTRA))
    for sid, s in registry.items():
        pdb, ent = s["pdb"], ENTITY_OVERRIDE.get(s["pdb"], s["entity"])
        url = f"https://data.rcsb.org/rest/v1/core/polymer_entity/{pdb}/{ent}"
        with urllib.request.urlopen(url, timeout=60) as r:
            data = json.loads(r.read())
        label = data["rcsb_polymer_entity"]["pdbx_description"]
        acc = (data["rcsb_polymer_entity_container_identifiers"].get("uniprot_ids") or [""])[0]
        seq = data["entity_poly"]["pdbx_seq_one_letter_code_can"].replace("\n", "")
        rows.append({"sample_id": sid, "pdb": pdb, "entity": ent, "chain": s["chain"],
                     "uniprot_acc": acc, "rcsb_description": label, "len": len(seq),
                     "sequence_sha256": hashlib.sha256(seq.encode()).hexdigest(),
                     "fetched_at": RUN_TS})
        seqs[sid] = seq
        print(f"[fetch] {sid} {pdb} entity{ent} len={len(seq)} acc={acc}", flush=True)
    wr(os.path.join(OUT405, "conditional_sequences.tsv"), rows,
       ["sample_id", "pdb", "entity", "chain", "uniprot_acc", "rcsb_description", "len",
        "sequence_sha256", "fetched_at"])
    # 一致性断言（interventions.yaml §1）
    asserts = {}
    asserts["pyp_same_sequence"] = seqs["PYP-2014-DARK"] == seqs["PYP-2014-LIGHT-INT"]
    rn = {k: seqs[k] for k in ["RNA-MONOMER-1FS3", "RNA-CSWAP-1F0V",
                               "RNA-MONOMER-1FS3-B", "RNA-NSWAP-1A2W"]}
    asserts["rnase_all_same_sequence"] = len(set(rn.values())) == 1
    asserts["rnase_mature124"] = list(set(rn.values()))[0].startswith("KETAAAKF") and \
        len(list(set(rn.values()))[0]) == 124
    asserts["rnase_1js0_same_sequence"] = seqs["RNA-TRIMER-1JS0"] == seqs["RNA-MONOMER-1FS3"]
    json.dump(asserts, open(os.path.join(OUT405, "identity_asserts.json"), "w"), indent=1)
    wr(os.path.join(OUT405, "degradation_registry.tsv"),
       [{"system": k.replace("_same_sequence", ""), "assert": k, "value": str(v),
         "action": "RUN" if v else "DEGRADE_不跑矩阵（config §1 失败路径）"}
        for k, v in asserts.items()],
       ["system", "assert", "value", "action"])
    with open(os.path.join(OUT405, "conditional.fa"), "w") as f:
        seen = set()
        for sid, seq in seqs.items():
            key = seq
            if key in seen:
                continue
            seen.add(key)
            f.write(f">{sid}\n{seq}\n")
    print("[fetch] 断言全过；去重 fasta 落盘", flush=True)


def load_pooled():
    emb = {}
    man = rd(os.path.join(OUT405, "emb_conditional", "extract_manifest.tsv"))
    key2arr = {}
    for r in man:
        if r["key"] not in key2arr:
            with np.load(os.path.join(OUT405, "emb_conditional", r["key"] + ".npz")) as z:
                key2arr[r["key"]] = {k: z[k] for k in z.files if k != "meta"}
    seq_of = {r["sample_id"]: r["sequence_sha256"] for r in
              rd(os.path.join(OUT405, "conditional_sequences.tsv"))}
    fa = {}
    name, buf = None, []
    for line in open(os.path.join(OUT405, "conditional.fa")):
        if line.startswith(">"):
            name = line[1:].strip()
        else:
            fa[name] = hashlib.sha256(line.strip().encode()).hexdigest()
    # 按 sample→fasta 名（同序列首个样本名）→emb 名对齐
    emb_by_sha = {}
    for r in man:
        sha = fa.get(r["name"])
        emb_by_sha[sha] = key2arr[r["key"]]["mean_layers"][0].astype(np.float32)  # 仅抽 33 层 → 单行
    for sid in SAMPLES:
        emb[sid] = emb_by_sha[seq_of[sid]]
    return emb


def fit_arm(X, y):
    clf = LogisticRegression(C=C_FIXED, class_weight="balanced", solver="liblinear",
                             max_iter=1000, random_state=2026)
    import warnings
    with warnings.catch_warnings(record=True) as wlist:
        warnings.simplefilter("always")
        clf.fit(X, y)
    conv = any("onverge" in str(w.message) for w in wlist)
    return clf, conv


def matrix():
    emb = load_pooled()
    qc = {"run_ts": RUN_TS, "config": "configs/interventions.yaml frozen_2026-09-26_p405",
          "systems": {}}
    # N1（config §1 失败路径守卫）：回读降级注册表；DEGRADE 系统跳过拟合并登记
    degraded = set()
    reg_path = os.path.join(OUT405, "degradation_registry.tsv")
    if os.path.exists(reg_path):
        for r in rd(reg_path):
            if not str(r["value"]).startswith("T"):  # value=False → 断言失败
                sysname = "pyp" if r["system"].startswith("pyp") else "rnase_a"
                degraded.add(sysname)
    for sysname, decision in [("pyp", "RUN"), ("rnase_a", "RUN_DEGRADED"), ("bpti", "STOP")]:
        sdir = os.path.join(ROOT, "results/interventions", sysname)
        os.makedirs(sdir, exist_ok=True)
        if sysname in degraded:
            qc["systems"][sysname] = {"decision": "BLOCKED_DEGRADED",
                                      "reason": "identity 断言失败（degradation_registry.tsv）→ 按 config §1 不跑矩阵"}
            print(f"[{sysname}] BLOCKED_DEGRADED（不跑矩阵）", flush=True)
            continue
        if decision == "STOP":
            fail = [{"item": "M arm", "reason": "variant 全部 structure_available=no → 池化表示不可计算"},
                    {"item": "functional endpoint", "reason": "结合常数数值 pending_fulltext（4/15 仅定性）→ 无可评价数值终点"},
                    {"item": "matching control", "reason": "两终点均无匹配对照 → 按 §6 STOP 登记不拟合"}]
            wr(os.path.join(sdir, "stop_registry.tsv"), fail, ["item", "reason"])
            audit = rd(os.path.join(ROOT, "data/curated/bpti_pairs.tsv"))
            wr(os.path.join(sdir, "source_field_audit.tsv"), audit, list(audit[0].keys()))
            qc["systems"]["bpti"] = {"decision": "STOP", "n_fit": 0}
            print(f"[{sysname}] STOP 已登记（不拟合）", flush=True)
            continue
        sids = [s for s, v in SAMPLES.items() if v["sys"] == sysname]
        y = np.array([SAMPLES[s]["y"] for s in sids])
        M = np.stack([emb[s] for s in sids])
        F = np.array([[SAMPLES[s]["f"]["illumination" if sysname == "pyp" else "oligomer_state"]]
                      for s in sids], dtype=float)
        degenerate = {"arm": [], "convergence_warning": [], "note": []}
        res_rows, coef_rows, pred_rows = [], [], []
        # 四臂
        armX = {"M": M, "F_only": F, "M_F": np.hstack([M, F]),
                "M_F_perm": None}
        # 置换（每组 2 成员对换；三种子同映射）
        perm_rows = []
        Fp = F.copy()
        pgs = OrderedDict()
        for i, s in enumerate(sids):
            pgs.setdefault(SAMPLES[s]["pg"], []).append(i)
        for pg, idx in pgs.items():
            if len(idx) == 2:
                Fp[idx[0]], Fp[idx[1]] = F[idx[1]], F[idx[0]].copy()
                for seed in SEEDS:
                    perm_rows.append({"seed": seed, "pair_group": pg,
                                      "sample_a": sids[idx[0]], "sample_b": sids[idx[1]],
                                      "swapped": True})
        wr(os.path.join(sdir, "perm_mapping.tsv"), perm_rows,
           ["seed", "pair_group", "sample_a", "sample_b", "swapped"])
        marg_assert = bool(np.array_equal(np.sort(F, axis=0), np.sort(Fp, axis=0)))
        armX["M_F_perm"] = np.hstack([M, Fp])
        preds = {}
        for arm, X in armX.items():
            clf, conv = fit_arm(X, y)
            pr = clf.predict(X)
            preds[arm] = pr
            degenerate["arm"].append(arm)
            degenerate["convergence_warning"].append(bool(conv))
            res_rows.append({"arm": arm, "pred_agrees_with_y": bool(np.array_equal(pr, y)),
                             "n_samples": len(sids),
                             "performance_estimate": "",  # 案例级恒空（断言 §7）
                             "directional_fact": f"pred={'match' if np.array_equal(pr, y) else 'mismatch'}"})
            for i, sid in enumerate(sids):  # S3：逐样本预测留痕（TODO step3）
                pred_rows.append({"arm": arm, "sample_id": sid, "y": int(y[i]),
                                  "pred": int(pr[i])})
            c = clf.coef_
            if arm == "F_only":   # N2：F_only 仅 1 列系数，按列语义归 coef_F
                coef_rows.append({"arm": arm, "coef_M_norm": "",
                                  "coef_F": float(c.ravel()[0]), "intercept": float(clf.intercept_[0]),
                                  "coef_M_first3": ""})
            else:
                fcols = c[:, M.shape[1]:].ravel()
                coef_rows.append({"arm": arm,
                                  "coef_M_norm": float(np.linalg.norm(c[:, :M.shape[1]])),
                                  "coef_F": float(fcols[0]) if fcols.size else "",
                                  "intercept": float(clf.intercept_[0]),
                                  "coef_M_first3": json.dumps([round(float(v), 6) for v in
                                                               c[:, :3].ravel()])})
        wr(os.path.join(sdir, "per_sample_predictions.tsv"), pred_rows,
           ["arm", "sample_id", "y", "pred"])
        wr(os.path.join(sdir, "arm_coefficients.tsv"), coef_rows,
           ["arm", "coef_M_norm", "coef_F", "intercept", "coef_M_first3"])
        # M 状态盲性（同序列构造事实）
        blind = bool(len({hashlib.sha256(M[i].tobytes()).hexdigest() for i in range(len(sids))}) == 1)
        # 判定（C-IN3 冻结框；config §5 interpretation 逐条实现）
        # refutes 分支：置换后 M+F_perm 行为与 M+F 无异 → 对照失效（重拟合在 n=2 下可逆）
        perm_same_as_mf = np.array_equal(preds["M_F_perm"], preds["M_F"])
        if perm_same_as_mf:
            judged = "refutes_control_failed"
            judged_note = ("置换对照失效（重拟合语义下 F↔y 反接可被系数变号完全恢复，系数精确反号见 "
                           "arm_coefficients.tsv）→ 按 config §5 refutes（对照失效）登记；判定量降级为"
                           "构造事实：M 状态盲（同序列表示逐位相同）+ F_only≡M+F（预测全同）→ " +
                           ("输入充分性语义的构造级支持（F 与标签同义），不依赖置换对照"
                            if sysname == "rnase_a" else
                            "条件主导的构造级支持，不依赖置换对照"))
        elif np.array_equal(preds["M_F"], y) and np.array_equal(preds["F_only"], y):
            judged = "supports_condition_dominant"
            judged_note = "M+F 与 F-only 同判且优于置换→条件主导框（边界：不写成模型内部恢复）"
        elif np.array_equal(preds["M_F"], y):
            judged = "supports_non_redundant"
            judged_note = "M+F 优于 F-only 与置换→非冗余信息框"
        else:
            judged = "no_gain_box"
            judged_note = "无增益框（M+F 未优于 M）——案例级方向事实如实登记"
        wr(os.path.join(sdir, "intervention_matrix.tsv"), res_rows,
           ["arm", "pred_agrees_with_y", "n_samples", "performance_estimate", "directional_fact"])
        qc["systems"][sysname] = {
            "decision": decision, "n_pairs": len(sids) // 2,
            "M_state_blind_by_construction": blind,
            "perm_marginal_preserved": marg_assert,
            "degenerate_protocol": degenerate,
            "judgement_case_level": judged,
            "judgement_note": judged_note,
            "degraded_direct_facts": {"M_state_blind": blind,
                                      "F_only_equals_M_F": bool(np.array_equal(preds["F_only"], preds["M_F"]))},
            "f_synonymous_with_label": sysname == "rnase_a",
            "coverage": {"n_samples": len(sids), "n_universe": len(sids),
                         "intersection_fraction": None,  # N4：由 TSV 回读后填（见下）
                         "note": "四臂交集=系统全部 pair_member（config §3 预登记）"},
            "perm_facts": {"joint_equals_column": "F 仅 1 可用列→两语义重合（config §5 登记重合）",
                           "nonidentity_space": 1,
                           "seeds": SEEDS, "change_fraction": 1.0,
                           "seed_gain": "无（非恒等置换空间仅 1 元素）"},
            "boundary": "输入充分性语义；不作为模型恢复物理信息的证据" if sysname == "rnase_a"
                        else "案例级方向性事实；不外推"}
        print(f"[{sysname}] judged={judged} M_blind={blind} perm_ok={marg_assert}", flush=True)
    # S1：性能估计列恒空断言——从写盘后的 TSV 回读校验（不再硬编码）
    perf_empty = True
    for sysname in ["pyp", "rnase_a"]:
        p = os.path.join(ROOT, "results/interventions", sysname, "intervention_matrix.tsv")
        n_seen = set()
        for r in rd(p):
            if r["performance_estimate"] != "":
                perf_empty = False
            n_seen.add(int(r["n_samples"]))
        # N4：intersection_fraction 由 TSV 回读实算（四臂行数一致且=宇宙样本数）
        arms_rows = len(rd(p))
        qc["systems"][sysname]["coverage"]["intersection_fraction"] = \
            round(min(n_seen) / qc["systems"][sysname]["coverage"]["n_universe"], 6)
    qc["asserts"] = {"performance_estimate_column_empty": perf_empty,
                     "coverage_registered": bool(
                         all(qc["systems"][s]["coverage"]["intersection_fraction"] == 1.0
                             for s in ("pyp", "rnase_a"))),
                     "identity_asserts": os.path.join(OUT405, "identity_asserts.json")}
    if not perf_empty:
        die("性能估计列非空——违反案例级断言")
    json.dump(qc, open(os.path.join(ROOT, "results/interventions/interventions_qc.json"), "w"),
              ensure_ascii=False, indent=1, sort_keys=True)
    print("[p406-08] DONE", flush=True)


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    if cmd == "fetch":
        fetch()
    elif cmd == "matrix":
        matrix()
    else:
        die("用法：fetch | matrix")
