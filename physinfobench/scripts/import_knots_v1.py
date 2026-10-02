#!/usr/bin/env python3
"""import_knots_v1.py — P1.04 打结拓扑标签导入（复用并核验旧项目产物）

输入（旧项目只读，2026-09-14/15 表族）：
  topology_task_eligibility_2026-09-14.tsv   568 候选链（presence/type/core 分层）
  topology_full674_verdict_2026-09-14.tsv    568 逐链裁定（topoly.alexander 双闭合）
  topology_support_audit_2026-09-14.tsv      568 支持率与来源注记聚合
  negative_controls_verified_2026-09-14.tsv  833 阴性（832 verified + 1 rejected）
  negative_selection_2026-09-14.tsv          876 匹配关系
输出：
  data/curated/knots.tsv（568 候选 + 833 阴性 = 1401 行）
规则：
  - 存在性（presence）与结型（type）目标分列；阴性 y=0 仅当 verified_negative（有双闭合 unknot 证据）；
  - rejected_knotted 记 excluded；隔离/异常保留 anomaly_resolution 原文；
  - 支持率=闭合稳定性语义，随行保留，不作生物学正确率解释。
"""
import csv
from collections import Counter
from datetime import datetime
from pathlib import Path

B = Path("/Users/yuan/Documents/Codex/2026-09-08/jie/benchmark_step1")
OUT = Path("data/curated/knots.tsv")
SOURCE_VERSION = "legacy topology 表族 2026-09-14（eligibility/verdict/support/annotations/negatives/selection）"
CREATED_AT = datetime.now().strftime("%Y-%m-%d %H:%M")

FIELDS = [
    "record_id", "row_kind", "pdb", "chain", "pdb_chain", "kind",
    "presence_target", "presence_mask", "presence_task_tier",
    "type_target", "type_mask", "type_task_tier",
    "verdict", "c2_primary", "c1_primary", "c2_primary_support",
    "c2_knotted_support", "c2_unknot_support", "c1_primary_support",
    "class_c2", "class_c1", "c1_c2_class_agree",
    "core_knot", "flank_knot", "core_best_type", "core_best_support",
    "n_annotations", "source_types", "internal_missing", "n_ca",
    "anomaly_resolution", "support_threshold_used", "type_agrees_source",
    "matched_positives", "n_matched_negatives", "exclusion_reason",
    "source_version", "created_at",
]


def main() -> int:
    te = {r["pdb_chain"]: r for r in csv.DictReader(open(B / "manifests/topology_task_eligibility_2026-09-14.tsv"), delimiter="\t")}
    vd = {f"{r['pdb']}_{r['chain']}": r for r in csv.DictReader(open(B / "manifests/topology_full674_verdict_2026-09-14.tsv"), delimiter="\t")}
    sa = {r["pdb_chain"]: r for r in csv.DictReader(open(B / "manifests/topology_support_audit_2026-09-14.tsv"), delimiter="\t")}
    nc = list(csv.DictReader(open(B / "manifests/negative_controls_verified_2026-09-14.tsv"), delimiter="\t"))
    ns = list(csv.DictReader(open(B / "manifests/negative_selection_2026-09-14.tsv"), delimiter="\t"))

    # 每个阳性链匹配到的阴性数（negative_selection: positive_chain -> 多行阴性 pdb_id）
    neg_count = Counter(r["positive_chain"] for r in ns)
    problems = []

    out = []
    for pcb, r in te.items():
        v = vd.get(pcb)
        s = sa.get(pcb)
        if v is None or s is None:
            problems.append(f"{pcb}: verdict/support 缺行")
            continue
        presence_tier = r["presence_task"]
        type_tier = r["type_task"]
        # 存在性目标：K/S 候选链即打结侧阳性候选（y=1），但仅 eligible 进入评价
        p_target, p_mask = ("1", "1") if presence_tier == "eligible" else ("", "0")
        # 结型目标：type_task eligible 仅 K 链（本数据集 112 全为 K），类型取主协议闭合 c2_primary；
        # S 链的 core 结型（core_best_type）仅 core_type_only 层，不入 eligible 目标
        best_type = s.get("core_best_type", "")
        if type_tier == "eligible" and r["kind"] == "K" and v["p_c2"] and v["p_c2"] != "none":
            t_target, t_mask = v["p_c2"], "1"
        elif type_tier == "eligible":
            t_target, t_mask = "", "0"
        else:
            t_target, t_mask = "", "0"
        pchain_key = f"{v['pdb']}_{v['chain']}"
        out.append({
            "record_id": pcb, "row_kind": "candidate",
            "pdb": v["pdb"], "chain": v["chain"], "pdb_chain": pcb, "kind": r["kind"],
            "presence_target": p_target, "presence_mask": p_mask, "presence_task_tier": presence_tier,
            "type_target": t_target, "type_mask": t_mask, "type_task_tier": type_tier,
            "verdict": v["verdict"], "c2_primary": v["p_c2"], "c1_primary": v["p_c1"],
            "c2_primary_support": s.get("c2_primary_support", ""),
            "c2_knotted_support": s.get("c2_knotted_support", ""),
            "c2_unknot_support": s.get("c2_unknot_support", ""),
            "c1_primary_support": s.get("c1_primary_support", ""),
            "class_c2": s.get("class_c2", ""), "class_c1": s.get("class_c1", ""),
            "c1_c2_class_agree": s.get("c1_c2_class_agree", ""),
            "core_knot": v["core_knot"], "flank_knot": v["flank_knot"],
            "core_best_type": best_type, "core_best_support": s.get("core_best_support", ""),
            "n_annotations": s.get("n_annotations", ""), "source_types": s.get("source_types", ""),
            "internal_missing": s.get("internal_missing", ""), "n_ca": s.get("n_ca", ""),
            "anomaly_resolution": r["anomaly_resolution_2026-09-14"],
            "support_threshold_used": r["support_threshold_used"],
            "type_agrees_source": r["type_agrees_source"],
            "matched_positives": "", "n_matched_negatives": str(neg_count.get(pchain_key, 0)),
            "exclusion_reason": "",
            "source_version": SOURCE_VERSION, "created_at": CREATED_AT,
        })

    for r in nc:
        pcb = f"{r['pdb_id']}{r['chain']}"
        verified = r["verdict"] == "verified_negative"
        out.append({
            "record_id": pcb, "row_kind": "negative",
            "pdb": r["pdb_id"], "chain": r["chain"], "pdb_chain": pcb, "kind": "N",
            "presence_target": ("0" if verified else ""), "presence_mask": ("1" if verified else "0"),
            "presence_task_tier": "verified_negative" if verified else "rejected_knotted",
            "type_target": "", "type_mask": "0", "type_task_tier": "not_applicable",
            "verdict": r["verdict"], "c2_primary": "", "c1_primary": "",
            "c2_primary_support": "", "c2_knotted_support": "",
            "c2_unknot_support": r.get("unknot_support_c2", ""), "c1_primary_support": "",
            "class_c2": r.get("class_c2", ""), "class_c1": r.get("class_c1", ""),
            "c1_c2_class_agree": "",
            "core_knot": "", "flank_knot": "", "core_best_type": "", "core_best_support": "",
            "n_annotations": "", "source_types": "",
            "internal_missing": r.get("internal_missing", ""), "n_ca": r.get("n_ca", ""),
            "anomaly_resolution": "", "support_threshold_used": "",
            "type_agrees_source": "",
            "matched_positives": r.get("matched_positives", ""),
            "n_matched_negatives": "", "exclusion_reason": ("" if verified else "rejected_knotted"),
            "source_version": SOURCE_VERSION, "created_at": CREATED_AT,
        })

    # ---- 断言 ----
    assert len(te) == 568 and len(vd) == 568 and len(sa) == 568
    assert len(out) == 568 + 833 == 1401, len(out)
    cands = [x for x in out if x["row_kind"] == "candidate"]
    negs = [x for x in out if x["row_kind"] == "negative"]
    assert sum(1 for x in cands if x["presence_task_tier"] == "eligible") == 188
    assert sum(1 for x in cands if x["type_task_tier"] == "eligible") == 112
    assert sum(1 for x in negs if x["verdict"] == "verified_negative") == 832
    assert sum(1 for x in negs if x["verdict"] == "rejected_knotted") == 1
    assert all((x["presence_target"] == "1") == (x["presence_mask"] == "1") for x in cands)
    assert not any(x["presence_target"] == "0" and x["row_kind"] == "candidate" for x in out)
    # eligible 结型目标必须非空且非 none
    t_elig = [x for x in cands if x["type_task_tier"] == "eligible"]
    assert len(t_elig) == 112
    assert all(x["type_target"] and x["type_target"] != "none" and x["type_mask"] == "1" for x in t_elig), \
        "eligible 结型目标存在空/none"
    print("eligible 结型分布:", dict(Counter(x["type_target"] for x in t_elig)))
    # 主键唯一
    ids = [x["record_id"] for x in out]
    assert len(ids) == len(set(ids)), "record_id 重复"

    if problems:
        print("PROBLEMS:")
        for p in problems:
            print("  -", p)
        return 1

    with open(OUT, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS, delimiter="\t", lineterminator="\n")
        w.writeheader()
        w.writerows(out)

    print("WROTE", OUT, len(out), "rows x", len(FIELDS), "cols")
    print("presence_task_tier(候选):", dict(Counter(x["presence_task_tier"] for x in cands)))
    print("type_task_tier(候选):", dict(Counter(x["type_task_tier"] for x in cands)))
    print("verdict(候选):", dict(Counter(x["verdict"] for x in cands)))
    print("kind(候选):", dict(Counter(x["kind"] for x in cands)))
    print("阴性: verified=832 rejected=1 | 有匹配关系记录的候选:", sum(1 for x in cands if x["n_matched_negatives"] != "0"))
    print("anomaly 留痕行:", sum(1 for x in cands if x["anomaly_resolution"]))
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
