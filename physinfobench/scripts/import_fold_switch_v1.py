#!/usr/bin/env python3
"""import_fold_switch_v1.py — P1.02 折叠转换全局标签导入（复用并核验旧项目产物）

输入（旧项目只读）：
  $B/manifests/fold_switching_release_eligibility.tsv      96 对资格表（分层与放行理由）
  $B/execution_round_2026-09-10/evidence/paper_evidence_round2_2026-09-13.tsv  186 行证据记录
  $B/manifests/data_corrections_2026-09-14.json            2 项已知修正
  $B/manifests/mapping_adjudication_2026-09-15.tsv         4 条映射裁决
  $B/benchmark_v0.1/constructs.tsv                         192 端点构建体记录
输出：
  data/curated/fold_switch_global.tsv（96 行，schema 见 configs/sample_schema.yaml 约定）
规则：
  - 仅 strict_state_candidate 且理由为 fulltext_evidence_and_full_construct 的 pair 记 target=1/mask=1；
    其余一律 target 空 + mask=0（未知不进分母）；不放任何阴性行（无可靠阴性来源，OD-NEGATIVE-FS 待讨论）。
  - 修正与裁决以注释列留痕，不静默改写旧分层。
"""
import csv
import json
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

B = Path("/Users/yuan/Documents/Codex/2026-09-08/jie/benchmark_step1")
OUT = Path("data/curated/fold_switch_global.tsv")
SOURCE_VERSION = ("legacy release_eligibility 2026-09-15 + evidence_round2 2026-09-13 + corrections 2026-09-14 "
                  "+ adjudication 2026-09-15 + P1.24 field revision (G1 2026-09-24)")
CREATED_AT = datetime.now().strftime("%Y-%m-%d %H:%M")

# P1.24（G1 2026-09-24 限定 2）：porter_8 表内分层落实——corrections 的 extension 建议
# 经 G1 采纳，tier 由 pending_evidence 改为 extension_condition_or_assembly（分析层=extension
# 敏感性，不入 L1 主分析；target/mask 保持空/0，不新增阳性）。
G1_TIER_OVERRIDE = {"porter_8_3m1bF__3lowA": "extension_condition_or_assembly"}
# P1.24：observed_sequence_identity_class 取值来源=reports/fs_three_layer/strict_evidence_tiers.tsv
# （P1.20；same_observed_sequence 列）。旧列 construct_difference_class 承自旧项目
# "构建体对齐区间全局 identity=1.0"语义，与"观测序列相同"混同——更名保留旧值，不回填未复核对。
TIERS_TSV = "reports/fs_three_layer/strict_evidence_tiers.tsv"

VERDICT_PREFIXES = ("yes", "partial", "unclear", "pending", "single_endpoint",
                    "no_state_change_on_shared_scope", "no")


def verdict_class(text: str) -> str:
    if not text:
        return "no_evidence_row"
    for p in VERDICT_PREFIXES:
        if text.startswith(p):
            return p
    return "other"


def main() -> int:
    elig = list(csv.DictReader(open(B / "manifests/fold_switching_release_eligibility.tsv"), delimiter="\t"))
    ev = list(csv.DictReader(open(B / "execution_round_2026-09-10/evidence/paper_evidence_round2_2026-09-13.tsv"), delimiter="\t"))
    corrections = json.load(open(B / "manifests/data_corrections_2026-09-14.json"))
    adj = list(csv.DictReader(open(B / "manifests/mapping_adjudication_2026-09-15.tsv"), delimiter="\t"))
    constructs = list(csv.DictReader(open(B / "benchmark_v0.1/constructs.tsv"), delimiter="\t"))

    ev_by_pair = {}
    for r in ev:
        ev_by_pair.setdefault(r["pair_id"], []).append(r)

    # constructs 关联键：pdb_id + requested_chain（每对两端点各一行，共 192）
    con_by_key = {}
    for r in constructs:
        con_by_key[(r["pdb_id"].lower(), r["requested_chain"].upper())] = r

    adj_by_pair = {}
    for r in adj:
        adj_by_pair.setdefault(r["pair_id"], []).append(r)

    # P1.24：观测序列同一性类别（仅 strict 10 对填值，源自 P1.20 tiers 表）
    tiers_rows = list(csv.DictReader(open(TIERS_TSV), delimiter="\t"))
    OBS_CLASS = {t["pair_id"]: ("identical_observed" if t["same_observed_sequence"] == "yes"
                                else "different_observed_construct_explained")
                 for t in tiers_rows}

    fields = [
        "pair_id", "pdb_a", "chain_a", "pdb_b", "chain_b",
        "tier", "tier_reason", "target", "valid_mask", "inclusion_status", "exclusion_reason",
        "uniprot_a", "uniprot_b", "mapping_identity_a", "mapping_identity_b",
        "mutation_count_a", "mutation_count_b", "same_uniprot",
        "construct_global_identity_class", "observed_sequence_identity_class", "comparison_type",
        "evidence_pointer_primary", "evidence_n_rows", "evidence_endpoints_covered",
        "reading_level_best",
        "verdict_a", "verdict_b", "verdict_pair_level", "doi_a", "doi_b", "evidence_location_sample",
        "state_pair_basis", "evidence_level", "unresolved_question",
        "correction_note", "adjudication_note",
        "source_version", "created_at",
    ]

    out_rows, problems = [], []
    for r in elig:
        pid = r["pair_id"]
        tier = r["release_state_status"]
        reason = r["release_state_reason"]

        # ---- 标签：保守阳性（仅 strict + fulltext 理由）----
        if tier == "strict_state_candidate" and reason == "fulltext_evidence_and_full_construct":
            target, mask, status = "1", "1", "eligible"
        elif tier == "excluded":
            target, mask, status = "", "0", "excluded"
        else:
            target, mask, status = "", "0", "pending"

        # ---- G1 2026-09-24 限定 2（P1.24）：porter_8 表内分层落实 ----
        g1_note = ""
        if G1_TIER_OVERRIDE.get(pid) and tier == "pending_evidence":
            tier = G1_TIER_OVERRIDE[pid]
            g1_note = (" | G1 2026-09-24 限定2 采纳 extension_condition_or_assembly，"
                       "本表 tier 已落实（P1.24）；分析层=extension 敏感性，不入 L1 主分析；"
                       "target/mask 不变")
        observed_seq_class = ""
        if pid in OBS_CLASS:
            observed_seq_class = OBS_CLASS[pid]

        # ---- 证据连接 ----
        rows_ev = ev_by_pair.get(pid, [])
        eps = sorted({x["endpoint"] for x in rows_ev if x["endpoint"] in ("A", "B")})
        pair_rows = [x for x in rows_ev if not x["endpoint"]]
        reading_best = ""
        order = ["fulltext", "fulltext_pmc_authormanuscript", "fulltext_green_oa",
                 "fulltext_machine_extracted", "full_text_historical_trial10",
                 "fulltext_cross_study_comparison", "metadata_only", "abstract_only",
                 "no_text_available"]
        for lv in order:
            if any(x["reading_level"] == lv for x in rows_ev):
                reading_best = lv
                break

        def end_ev(ep):
            cand = [x for x in rows_ev if x["endpoint"] == ep]
            if not cand:
                return "", "", ""
            # 多行时优先有实际 verdict 的行（避免空 verdict 行掩盖有效判定）
            cand_sorted = sorted(cand, key=lambda x: 0 if x["pair_state_supported"] else 1)
            x = cand_sorted[0]
            loc = "; ".join(filter(None, [x["evidence_location"], x["page_figure"]]))
            return x["doi"], verdict_class(x["pair_state_supported"]), loc

        doi_a, v_a, loc_a = end_ev("A")
        doi_b, v_b, loc_b = end_ev("B")
        v_pair = verdict_class(pair_rows[0]["pair_state_supported"]) if pair_rows else ""

        # ---- 构建体记录（按 pdb+chain）----
        ca = con_by_key.get((r["pdb_a"].lower(), r["chain_a"].upper()), {})
        cb = con_by_key.get((r["pdb_b"].lower(), r["chain_b"].upper()), {})
        unp_a, unp_b = ca.get("primary_uniprot", ""), cb.get("primary_uniprot", "")
        same_uniprot = "yes" if unp_a and unp_a == unp_b else ("no" if unp_a and unp_b else "")

        # ---- 修正与裁决（留痕不静默改写）----
        corr_note = ""
        for key, c in corrections.items():
            if key.startswith(pid):
                corr_note = f"{c['issue']} -> {c['resolution']} | action: {c['action']}"
                if g1_note:
                    corr_note += g1_note
                elif tier == "pending_evidence" and "extension_condition_or_assembly" in c["action"]:
                    corr_note += " | 注意：修正建议的分层未反映在 release 资格表中（仍为 pending_evidence），疑点清单已登记"
        adj_notes = []
        for a in adj_by_pair.get(pid, []):
            adj_notes.append(f"{a['pdb_chain']}: {a['adjudication']} ({a['rationale'][:80]}…)")

        # ---- 一致性检查 ----
        # 阳性的证据可追溯性：round2 文献扫描连接（任一端点/对级，且须存在 yes 前缀判定），
        # 或旧项目对原发论文的全文本审计依据（state_pair_basis + source_evidence_quality）。
        basis = r["state_pair_basis"]
        quality = r["source_evidence_quality"]
        has_round2_yes = any(
            (x["pair_state_supported"] or "").startswith("yes") for x in rows_ev
        )
        has_primary_audit = bool(basis and quality and quality.startswith("primary_full_text"))
        if target == "1" and not (has_round2_yes or has_primary_audit):
            problems.append(f"{pid}: strict 阳性但无任何可追溯证据（round2 与原文审计均缺失）")
        if tier == "strict_state_candidate" and reason != "fulltext_evidence_and_full_construct":
            problems.append(f"{pid}: strict 层但放行理由异常: {reason}")
        evidence_pointer = (f"Porter2018_PNAS_10.1073/pnas.1800168115_TableS1; "
                            f"audit={quality or 'none'}; basis={basis or 'none'}")

        out_rows.append({
            "pair_id": pid, "pdb_a": r["pdb_a"], "chain_a": r["chain_a"],
            "pdb_b": r["pdb_b"], "chain_b": r["chain_b"],
            "tier": tier, "tier_reason": reason,
            "target": target, "valid_mask": mask, "inclusion_status": status,
            "exclusion_reason": "historical_not_usable" if reason == "historical_not_usable" else "",
            "uniprot_a": unp_a, "uniprot_b": unp_b,
            "mapping_identity_a": ca.get("mapping_identity", ""),
            "mapping_identity_b": cb.get("mapping_identity", ""),
            "mutation_count_a": ca.get("mutation_count", ""),
            "mutation_count_b": cb.get("mutation_count", ""),
            "same_uniprot": same_uniprot,
            "construct_global_identity_class": r["construct_difference_class"],
            "observed_sequence_identity_class": observed_seq_class,
            "comparison_type": r["comparison_type"],
            "evidence_pointer_primary": evidence_pointer,
            "evidence_n_rows": len(rows_ev),
            "evidence_endpoints_covered": "+".join(eps) if eps else ("pair_level" if pair_rows else "none"),
            "reading_level_best": reading_best,
            "verdict_a": v_a, "verdict_b": v_b, "verdict_pair_level": v_pair,
            "doi_a": doi_a, "doi_b": doi_b,
            "evidence_location_sample": loc_a or loc_b,
            "state_pair_basis": r["state_pair_basis"],
            "evidence_level": r["evidence_level"],
            "unresolved_question": r["unresolved_question"],
            "correction_note": corr_note,
            "adjudication_note": " | ".join(adj_notes),
            "source_version": SOURCE_VERSION,
            "created_at": CREATED_AT,
        })

    # ---- 断言 ----
    assert len(out_rows) == 96, f"行数 {len(out_rows)} != 96"
    tiers = Counter(x["tier"] for x in out_rows)
    assert tiers["strict_state_candidate"] == 10, tiers
    assert tiers["excluded"] == 2, tiers
    # P1.24 断言：porter_8 分层落实 + 观测序列类别仅 strict 填充
    assert tiers["extension_condition_or_assembly"] == 6 and tiers["pending_evidence"] == 52, tiers
    p8row = next(x for x in out_rows if x["pair_id"] == "porter_8_3m1bF__3lowA")
    assert p8row["tier"] == "extension_condition_or_assembly" and p8row["target"] == "" and p8row["valid_mask"] == "0"
    assert sum(1 for x in out_rows if "G1 2026-09-24 限定2" in x["correction_note"]) == 1
    obs_filled = [x for x in out_rows if x["observed_sequence_identity_class"]]
    assert len(obs_filled) == 10 and all(x["tier"] == "strict_state_candidate" for x in obs_filled)
    assert sum(1 for x in obs_filled if x["observed_sequence_identity_class"] == "identical_observed") == 1
    n_pos = sum(1 for x in out_rows if x["target"] == "1")
    assert n_pos == sum(1 for x in out_rows if x["tier"] == "strict_state_candidate")
    assert all((x["target"] == "1") == (x["valid_mask"] == "1") for x in out_rows)
    assert not any(x["target"] == "0" for x in out_rows), "不得伪造阴性"
    if problems:
        print("PROBLEMS:")
        for p in problems:
            print("  -", p)
        return 1

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields, delimiter="\t", lineterminator="\n")
        w.writeheader()
        w.writerows(out_rows)

    # ---- 统计 ----
    print("WROTE", OUT, len(out_rows), "rows x", len(fields), "cols")
    print("tier 分布:", dict(tiers))
    ev_cov = Counter(x["evidence_endpoints_covered"] for x in out_rows)
    print("证据端点覆盖:", dict(ev_cov))
    strict_rows = [x for x in out_rows if x["tier"] == "strict_state_candidate"]
    print("strict 10 对：both 端点证据", sum(1 for x in strict_rows if x["evidence_endpoints_covered"] == "A+B"),
          "| same_uniprot", Counter(x["same_uniprot"] for x in strict_rows),
          "| 突变计数", Counter(f"{x['mutation_count_a']}/{x['mutation_count_b']}" for x in strict_rows))
    print("strict 阳性 same_uniprot=yes:", sum(1 for x in strict_rows if x["same_uniprot"] == "yes"))
    corr_pairs = [x["pair_id"] for x in out_rows if x["correction_note"]]
    print("修正留痕:", corr_pairs)
    adj_pairs = [x["pair_id"] for x in out_rows if x["adjudication_note"]]
    print("裁决留痕:", adj_pairs)
    return 0


if __name__ == "__main__":
    sys.exit(main())
