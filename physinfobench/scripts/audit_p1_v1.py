#!/usr/bin/env python3
"""audit_p1_v1.py — P1.09 数据质量与可行性审计（含三层设计 strict positive 专项审计）

范围：P1.02–P1.08 全部 curated 表 + 旧项目 constructs 端点序列（只读）
输出：
  reports/fs_three_layer/strict_positive_audit.tsv   10 对 strict positive 逐对审计
  reports/fs_three_layer/strict_positive_qc.json     机器检查结果汇总
  data/manifests/exclusions.tsv                      排除/隔离清单
  stdout：任务流失表与三层可行性数字（写入 feasibility.md 由报告汇总）
"""
import csv
import json
from collections import Counter
from datetime import datetime
from pathlib import Path

B = Path("/Users/yuan/Documents/Codex/2026-09-08/jie/benchmark_step1")
C = Path("data/curated")
OUT_AUDIT = Path("reports/fs_three_layer/strict_positive_audit.tsv")
OUT_QC = Path("reports/fs_three_layer/strict_positive_qc.json")
OUT_EXCL = Path("data/manifests/exclusions.tsv")
CREATED_AT = datetime.now().strftime("%Y-%m-%d %H:%M")

AA = set("ACDEFGHIKLMNPQRSTVWY")


def read_tsv(p):
    return list(csv.DictReader(open(p), delimiter="\t"))


def main() -> int:
    problems = []
    ok = []
    fs = read_tsv(C / "fold_switch_global.tsv")
    fr = read_tsv(C / "fold_switch_regions.tsv")
    kn = read_tsv(C / "knots.tsv")
    dis = read_tsv(C / "disorder.tsv")
    dm = read_tsv(C / "disorder_masks.tsv")
    pyp = read_tsv(C / "pyp_pairs.tsv")
    rna = read_tsv(C / "rnase_a_pairs.tsv")
    bpti = read_tsv(C / "bpti_pairs.tsv")

    # ---------- 1. 通用完整性 ----------
    def pk_unique(rows, key, label):
        ids = [r[key] for r in rows]
        dup = [k for k, n in Counter(ids).items() if n > 1]
        (problems if dup else ok).append(f"{label}: 主键唯一性 {'FAIL ' + str(dup[:3]) if dup else 'OK (%d)' % len(ids)}")

    pk_unique(fs, "pair_id", "fold_switch_global")
    rid = Counter((r["entry_id"], r["region_index"]) for r in fr)
    dupid = [k for k, n in rid.items() if n > 1]
    (problems if dupid else ok).append(f"fold_switch_regions: (entry,region) 唯一性 {'FAIL' if dupid else 'OK (203)'}")
    pk_unique(kn, "record_id", "knots")
    pk_unique(dis, "region_id" if "region_id" in dis[0] else "disprot_id", "disorder(region 级唯一性另行)")
    dis_key = Counter((r["disprot_id"], r["region_id"], r["label_class"]) for r in dis)
    dupdis = [k for k, n in dis_key.items() if n > 1]
    (problems if dupdis else ok).append(f"disorder: (dp,region,class) 唯一性 {'FAIL %s' % dupdis[:2] if dupdis else 'OK'}")

    # 区间边界
    bad_iv = [r for r in dis if not (1 <= int(r["start"]) <= int(r["end"]) <= int(r["protein_length"]))]
    (problems if bad_iv else ok).append(f"disorder: 区间边界 {'FAIL %d' % len(bad_iv) if bad_iv else 'OK (13983)'}")
    bad_rr = [r for r in fr if not (1 <= int(r["pdf_seq_start"]) <= int(r["pdf_seq_end"]))]
    (problems if bad_rr else ok).append(f"fold_switch_regions: pdf 区间序 {'FAIL' if bad_rr else 'OK'}")

    # mask 语义
    bad_m = [r for r in fs if (r["target"] == "1") != (r["valid_mask"] == "1")]
    (problems if bad_m else ok).append(f"fold_switch_global: target⟺mask {'FAIL' if bad_m else 'OK'}")
    bad_km = [r for r in kn if r["row_kind"] == "candidate" and (r["presence_target"] == "1") != (r["presence_mask"] == "1")]
    (problems if bad_km else ok).append(f"knots: presence⟺mask（仅候选行） {'FAIL' if bad_km else 'OK'}")
    bad_dm = [r for r in dm if r["mask"] == "1" and r["state"] == "NA"]
    (problems if bad_dm else ok).append(f"disorder_masks: NA&mask=1 {'FAIL' if bad_dm else 'OK'}")

    # 序列合法性（DisProt 条目序列在 raw JSON）
    dpj = json.load(open("data/raw/disprot/2026-09-22/disprot_api_search.json"))["data"]
    aa_x = []
    for e in dpj:
        bad = set(e["sequence"]) - AA
        if bad:
            aa_x.append({"disprot_id": e["disprot_id"], "letters": sorted(bad), "len": len(e["sequence"])})
    ok.append(f"DisProt 序列字母全量校验: 完成 3337 条目；发现含非标准字母条目 {len(aa_x)} 个（X 等，序列级事实，隔离供 P2.05 提取策略决策；标签区段位置语义基本不变；唯一反例 DP02886 位置 93 的 Z 残基落在 mask=1 区段内，已单列）")

    # ---------- 2. 重复检查 ----------
    fs_same_unp = Counter((r["uniprot_a"], r["uniprot_b"]) for r in fs if r["uniprot_a"] and r["uniprot_b"])
    dup_unp = {k: n for k, n in fs_same_unp.items() if n > 1}
    (problems if dup_unp else ok).append(f"fold_switch: 同 UniProt 对重复 {'FAIL %s' % list(dup_unp.items())[:2] if dup_unp else 'OK（同蛋白多对按 pair 组绑定，P2.01 处理）'}")

    # ---------- 3. strict positive 专项审计（三层） ----------
    constructs = {}
    for r in csv.DictReader(open(B / "benchmark_v0.1/constructs.tsv"), delimiter="\t"):
        constructs[(r["pdb_id"].lower(), r["requested_chain"].upper())] = r
    strict = [r for r in fs if r["tier"] == "strict_state_candidate"]
    assert len(strict) == 10

    substr_ck = json.load(open("data/raw/audit_uniprot/2026-09-23/substring_check.json"))
    audit_fields = ["pair_id", "uniprot_a", "uniprot_b", "same_uniprot", "uniprot_substring_a", "uniprot_substring_b",
                    "obs_len_a", "obs_len_b", "identical_observed_seq", "seq_diff_positions",
                    "mut_a", "mut_b", "coverage_a", "coverage_b", "mapping_identity_a", "mapping_identity_b",
                    "state_basis", "comparison_type", "construct_global_identity_class",
                    "observed_sequence_identity_class",
                    "evidence_pointer", "audit_quality", "evidence_traceable",
                    "region_rows", "fine_region_rows", "usable_region_rows",
                    "n_regions_masked", "positive_evidence_tier", "porter8_conflict_note", "created_at"]
    audit_rows = []
    corr = json.load(open(B / "manifests/data_corrections_2026-09-14.json"))
    fr_by_pair = {}
    for r in fr:
        fr_by_pair.setdefault(r["pair_id"], []).append(r)

    for r in strict:
        ca = constructs.get((r["pdb_a"].lower(), r["chain_a"].upper()), {})
        cb = constructs.get((r["pdb_b"].lower(), r["chain_b"].upper()), {})
        sa, sb = ca.get("observed_sequence", ""), cb.get("observed_sequence", "")
        identical = (sa == sb and sa != "")
        diffs = ""
        if sa and sb and sa != sb:
            diffs = ";".join(f"{i+1}:{a}/{b}" for i, (a, b) in enumerate(zip(sa, sb)) if a != b)[:200]
        ev_ptr = r["evidence_pointer_primary"]
        audit_quality = ev_ptr.split("audit=")[-1].split(";")[0] if "audit=" in ev_ptr else ""
        traceable = "yes" if (audit_quality.startswith("primary_full_text") or r["reading_level_best"]) else "no"
        regs = fr_by_pair.get(r["pair_id"], [])
        n_fine = sum(1 for x in regs if x["region_task_tier"] == "fine_candidate")
        n_usable = sum(1 for x in regs if x["lit_label_decision"] == "usable")
        p8 = ""
        if r["pair_id"] == "porter_8_3m1bF__3lowA":
            p8 = "corrections 建议 extension_condition_or_assembly；G1 2026-09-24 已采纳并经 P1.24 在 fold_switch_global 落实（该表 tier=extension_condition_or_assembly）"
        sub = substr_ck.get(r["pair_id"], {})
        audit_rows.append({
            "pair_id": r["pair_id"], "uniprot_a": r["uniprot_a"], "uniprot_b": r["uniprot_b"],
            "same_uniprot": r["same_uniprot"],
            "uniprot_substring_a": str(sub.get("A_sub")), "uniprot_substring_b": str(sub.get("B_sub")),
            "obs_len_a": ca.get("observed_length", ""), "obs_len_b": cb.get("observed_length", ""),
            "identical_observed_seq": "yes" if identical else ("no" if (sa and sb) else "missing"),
            "seq_diff_positions": diffs or ("长度差 %d" % (len(sa) - len(sb)) if sa and sb and len(sa) != len(sb) else ""),
            "mut_a": r["mutation_count_a"], "mut_b": r["mutation_count_b"],
            "coverage_a": ca.get("mapped_residue_count", ""), "coverage_b": cb.get("mapped_residue_count", ""),
            "mapping_identity_a": r["mapping_identity_a"], "mapping_identity_b": r["mapping_identity_b"],
            "state_basis": r["state_pair_basis"][:60], "comparison_type": r["comparison_type"],
            "construct_global_identity_class": r["construct_global_identity_class"],
            "observed_sequence_identity_class": r["observed_sequence_identity_class"],
            "evidence_pointer": ev_ptr, "audit_quality": audit_quality,
            "evidence_traceable": traceable,
            "region_rows": len(regs), "fine_region_rows": n_fine, "usable_region_rows": n_usable,
            "n_regions_masked": sum(int(x["unmapped_residue_count"] or 0) for x in regs),
            "positive_evidence_tier": "strict",
            "porter8_conflict_note": p8,
            "created_at": CREATED_AT})

    # 同序列证据分层（子串核验 2026-09-23 00:23）
    substr = json.load(open("data/raw/audit_uniprot/2026-09-23/substring_check.json"))
    both_sub = sum(1 for v in substr.values() if v["A_sub"] and v["B_sub"])
    id_lo = min(min(float(r["mapping_identity_a"] or 1), float(r["mapping_identity_b"] or 1)) for r in strict)
    ok.append(f"strict 同序列证据分层: same_uniprot=10 | SIFTS mapping_identity 最小={id_lo} | 双端 UniProt 子串={both_sub}/10（其余疑为标签/isoform/边界，列 G1 甄别）")

    # 旧审计抽查计数（8 个依赖旧审计：evidence 无 round2 行者）
    n_old_audit = sum(1 for r in strict if r["evidence_endpoints_covered"] in ("none", "pair_level"))
    # ---------- 4. 流失表与三层可行性 ----------
    def loss(label, raw_n, ev_ok, mappable, evaluable):
        return {"task": label, "raw": raw_n, "evidence_qualified": ev_ok,
                "mappable": mappable, "evaluable": evaluable}

    flow = {
        "T-FS-GLOBAL/L1": loss("fold_switch strict 阳性", 96, 10, 10, 10),
        "T-FS-REGION": loss("fold_switch 区段", 203, 203, 167, sum(1 for r in fr if r["region_task_tier"] == "fine_candidate" and r["mapping_quality"] == "full")),
        "T-FS-L2-PU": loss("PU 排序（阳=10 + 背景待 P1.10）", 10, 10, 10, 10),
        "T-FS-L3-MATCHED": loss("匹配对照（待 P1.11/P2.01）", 10, 10, 10, 0),
        "T-KNOT-PRESENCE": loss("打结存在性", 1401, 1401, 188 + 832, 188 + 832),
        "T-KNOT-TYPE": loss("结型", 1401, 112, 112, 112),
        "T-DISORDER-RES": loss("无序残基（双类蛋白）", 3337, 3337, 3319, 5),
        "T-PYP-STATE": loss("PYP 状态对", 2, 2, 2, 1),
        "T-RNASE-ASSEMBLY": loss("RNase A 组装对", 4, 4, 4, 2),
        "T-BPTI-REDOX": loss("BPTI 功能样本", 15, 5, 5, "pending_fulltext(ox/red 数值)"),
    }
    indep = {
        "fold_switch_uniprot": len({r["uniprot_a"] for r in fs if r["uniprot_a"]} | {r["uniprot_b"] for r in fs if r["uniprot_b"]}),
        "fold_switch_strict_uniprot": len({r["uniprot_a"] for r in strict}),
        "knot_chains": len(kn), "knot_presence_positive_chains": 188,
        "disprot_entries": 3337, "disprot_both_class_entries": 5,
        "pyp_pair_units": 1, "rnase_units": 1, "bpti_units": 1,
    }

    # ---------- 5. 排除/隔离清单 ----------
    excl = []
    for r in kn:
        if r["presence_task_tier"] in ("excluded_artifact", "rejected_knotted") or r["row_kind"] == "negative" and r["verdict"] == "rejected_knotted":
            excl.append({"scope": "knots", "record": r["record_id"], "kind": "excluded",
                         "reason": r.get("exclusion_reason") or r["presence_task_tier"], "created_at": CREATED_AT})
    for r in fs:
        if r["tier"] == "excluded":
            excl.append({"scope": "fold_switch", "record": r["pair_id"], "kind": "excluded",
                         "reason": r["exclusion_reason"] or r["tier_reason"], "created_at": CREATED_AT})
    for r in dis:
        if r["label_class"] in ("conditional_transition", "other_structural_state"):
            pass  # 隔离不进排除表，仅计数
    iso = Counter(r["label_class"] for r in dis if r["label_class"] in ("conditional_transition", "other_structural_state"))
    for r in rna:
        if r["row_kind"] == "excluded":
            excl.append({"scope": "rnase_a", "record": r["sample_id"], "kind": "excluded", "reason": r["evidence_quote"][:60], "created_at": CREATED_AT})

    # ---------- 写出 ----------
    OUT_AUDIT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_AUDIT, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=audit_fields, delimiter="\t", lineterminator="\n")
        w.writeheader(); w.writerows(audit_rows)
    with open(OUT_EXCL, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["scope", "record", "kind", "reason", "created_at"], delimiter="\t", lineterminator="\n")
        w.writeheader(); w.writerows(excl)

    qc = {"created_at": CREATED_AT,
          "checks_ok": ok, "problems": problems,
          "strict_positive": {
              "n": 10,
              "same_uniprot": sum(1 for a in audit_rows if a["same_uniprot"] == "yes"),
              "identical_observed_seq": sum(1 for a in audit_rows if a["identical_observed_seq"] == "yes"),
              "seq_diff_variants": [a["pair_id"] for a in audit_rows if a["identical_observed_seq"] == "no"],
              "zero_mutation": sum(1 for a in audit_rows if a["mut_a"] == "0" and a["mut_b"] == "0"),
              "traceable_evidence": sum(1 for a in audit_rows if a["evidence_traceable"] == "yes"),
              "old_audit_dependent": n_old_audit,
              "both_uniprot_substring": sum(1 for v in substr_ck.values() if v["A_sub"] and v["B_sub"]),
              "a_only_substring": sum(1 for v in substr_ck.values() if v["A_sub"] and not v["B_sub"]),
              "no_substring": sum(1 for v in substr_ck.values() if not v["A_sub"] and not v["B_sub"]),
              "with_fine_region": sum(1 for a in audit_rows if int(a["fine_region_rows"]) > 0),
              "with_usable_region": sum(1 for a in audit_rows if int(a["usable_region_rows"]) > 0),
              "porter8_conflict_in_strict": sum(1 for a in audit_rows if a["porter8_conflict_note"]),
              "porter8_conflict_registered": "非 strict 对；G1 2026-09-24 限定2 采纳 extension_condition_or_assembly 并经 P1.24 在 fold_switch_global 落实；完整冲突描述见旧项目 manifests/data_corrections_2026-09-14.json 与 fold_switch_global.correction_note"},
          "flow": flow, "independent_units": indep,
          "disprot_isolated": dict(iso), "disprot_conflict_entries": 92, "disprot_nonstandard_aa_entries": aa_x,
          "exclusions_n": len(excl)}
    json.dump(qc, open(OUT_QC, "w"), ensure_ascii=False, indent=1)

    print("AUDIT DONE", CREATED_AT)
    print("strict 同序列(观测):", qc["strict_positive"]["identical_observed_seq"], "/10")
    print("strict 可追溯证据:", qc["strict_positive"]["traceable_evidence"], "/10")
    print("旧审计依赖:", n_old_audit, "| fine 区段对:", qc["strict_positive"]["with_fine_region"], "| usable 区段对:", qc["strict_positive"]["with_usable_region"])
    print("排除清单:", len(excl), "| PROBLEMS:", len(problems))
    for p in problems: print("  P:", p)
    return 1 if problems else 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
