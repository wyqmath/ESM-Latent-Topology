#!/usr/bin/env python3
"""import_fold_switch_regions_v1.py — P1.03 折叠转换核心区导入（复用并核验旧项目产物）

输入（旧项目只读）：
  supplement/s2_regions_v3.tsv                    203 行作者标注区段（PDF 补充序列编号）
  mapping/region_mapping_v2.tsv                   203 行区段→结构坐标映射（label_seq/auth）
  mapping/region_task_eligibility_v1.tsv          203 行区域任务分层（fine/coarse/pending）
  mapping/residue_mapping_v2.tsv                  51332 行逐残基映射（用于抽查校验）
  mapping/endpoint_chain_match.tsv                190 行端点链匹配诊断
  manifests/fold_switching_release_eligibility.tsv 96 对（文献区段标签类型/精度/裁决字段）
输出：
  data/curated/fold_switch_regions.tsv（203 行，区间级；逐残基展开沿用 residue_mapping_v2 并登记）
规则：
  - 作者粗体区段≠逐残基实验真值：literature_region_label_type/precision/decision/reason 随行保留；
  - 部分映射（mapping_quality=partial）不伪装成完整标签，缺口计数显式输出；
  - 标准序列（UniProt）坐标的逐残基转换留待序列导入后执行，本表提供 label_seq/auth/pdf 三坐标体系
    并逐行登记映射质量，不伪造标准坐标。
"""
import csv
from collections import Counter
from datetime import datetime
from pathlib import Path

B = Path("/Users/yuan/Documents/Codex/2026-09-08/jie/benchmark_step1")
OUT = Path("data/curated/fold_switch_regions.tsv")
SOURCE_VERSION = "legacy s2_regions_v3 + region_mapping_v2 + region_task_eligibility_v1 + residue_mapping_v2 + endpoint_chain_match + release_eligibility (2026-09-15)"
CREATED_AT = datetime.now().strftime("%Y-%m-%d %H:%M")


def main() -> int:
    s2 = {(r["entry_id"], r["region_index"]): r for r in
          csv.DictReader(open(B / "execution_round_2026-09-10/supplement/s2_regions_v3.tsv"), delimiter="\t")}
    rmap = list(csv.DictReader(open(B / "execution_round_2026-09-10/mapping/region_mapping_v2.tsv"), delimiter="\t"))
    rte = {(r["entry_id"], r["region_index"]): r for r in
           csv.DictReader(open(B / "execution_round_2026-09-10/mapping/region_task_eligibility_v1.tsv"), delimiter="\t")}
    elig = {r["pair_id"]: r for r in
            csv.DictReader(open(B / "manifests/fold_switching_release_eligibility.tsv"), delimiter="\t")}
    ecm = {r["entry_id"]: r for r in
           csv.DictReader(open(B / "execution_round_2026-09-10/mapping/endpoint_chain_match.tsv"), delimiter="\t")}
    res_by_key = {}
    for r in csv.DictReader(open(B / "execution_round_2026-09-10/mapping/residue_mapping_v2.tsv"), delimiter="\t"):
        res_by_key.setdefault((r["entry_id"], r["pair_id"]), []).append(r)

    fields = [
        "pair_id", "entry_id", "region_index",
        "pdf_seq_start", "pdf_seq_end", "pdf_len",
        "mapped_chain", "label_seq_start", "label_seq_end", "mapped_len",
        "label_interval_internal_gap",
        "auth_start", "auth_end", "mapping_quality", "unmapped_residue_count",
        "seq_prefix10", "seq_suffix10", "source_pages", "s2_mask_basis", "s2_review_status",
        "lit_range_status", "lit_ranges_auth", "range_field_consistency",
        "lit_label_type", "lit_label_precision", "lit_label_decision", "lit_label_reason",
        "endpoint_identity", "endpoint_chain_match_diag",
        "release_state_status", "region_task_tier",
        "region_target_semantics", "residue_expansion_source",
        "source_version", "created_at",
    ]

    out, problems = [], []
    for row_i, r in enumerate(rmap):  # 以 region_mapping 为主表（203 行）
        key = (r["entry_id"], r["region_index"])
        s = s2.get(key)
        t = rte.get(key)
        if s is None:
            problems.append(f"{key}: region_mapping 行在 s2_regions_v3 无对应")
            continue
        if t is None:
            problems.append(f"{key}: region_mapping 行在 region_task_eligibility 无对应")
            continue
        e = elig.get(r["pair_id"])
        if e is None:
            problems.append(f"{key}: pair_id 不在资格表")
            continue

        # 坐标一致性
        if s["seq_start"] != r["pdf_seq_start"] or s["seq_end"] != r["pdf_seq_end"]:
            problems.append(f"{key}: s2 与 region_mapping 的 PDF 区间不一致")
        pdf_len = int(r["pdf_len"])
        mapped_len = int(r["mapped_len"])
        interval_len = int(r["label_seq_end"]) - int(r["label_seq_start"]) + 1
        # label_seq 区间可含内部缺口（构建体缺失/编号跳位），区间长度 >= 实际映射数才合法
        if interval_len < mapped_len:
            problems.append(f"{key}: label_seq 区间长度 {interval_len} < mapped_len {mapped_len}")
        internal_gap = interval_len - mapped_len
        if r["mapping_quality"] == "full" and mapped_len != pdf_len:
            problems.append(f"{key}: quality=full 但映射数≠PDF 长度")
        unmapped = pdf_len - mapped_len
        if (r["mapping_quality"] == "partial") != (unmapped > 0):
            problems.append(f"{key}: quality 与缺口数矛盾")

        # 区段内逐残基抽查（每第 17 行抽 1 个，含全部 full 质量 strict 行）
        rel_state = t["release_state_status"]
        strict = rel_state == "strict_state_candidate"
        if strict or (row_i % 17 == 0):  # 行序号取模抽样，结果可复现
            res = res_by_key.get((r["entry_id"], r["pair_id"]), [])
            lo, hi = int(r["pdf_seq_start"]), int(r["pdf_seq_end"])
            in_region = [x for x in res
                         if x["pdf_position"] and lo <= int(x["pdf_position"]) <= hi]
            # 旧逐残基表只为已映射位置建行：命中数应恰为 mapped_len（partial 区段的
            # 未映射 pdf 位置无行），且每行都应有 label_seq_id
            if len(in_region) != mapped_len:
                problems.append(f"{key}: 逐残基抽查 命中{len(in_region)} != mapped_len {mapped_len}")
            else:
                n_label = sum(1 for x in in_region if x["label_seq_id"])
                if n_label != mapped_len:
                    problems.append(f"{key}: 逐残基抽查 label_seq 覆盖 {n_label} != mapped_len {mapped_len}")

        # 链匹配诊断
        m = ecm.get(r["entry_id"], {})
        if m and m.get("pair_id") != r["pair_id"]:
            problems.append(f"{key}: endpoint_chain_match 的 pair_id 不一致")

        tier = t["region_task_tier"]
        semantics = {"fine_candidate": "core_region_candidate",
                     "coarse_candidate": "coarse_region_candidate",
                     "fine_extension": "extension_region_candidate",
                     "pending_review": "unreviewed_pending"}.get(tier, tier)
        if tier == "fine_candidate" and rel_state != "strict_state_candidate":
            problems.append(f"{key}: fine_candidate 但 pair 非 strict")

        out.append({
            "pair_id": r["pair_id"], "entry_id": r["entry_id"], "region_index": r["region_index"],
            "pdf_seq_start": r["pdf_seq_start"], "pdf_seq_end": r["pdf_seq_end"], "pdf_len": r["pdf_len"],
            "mapped_chain": r["mapped_chain"],
            "label_seq_start": r["label_seq_start"], "label_seq_end": r["label_seq_end"],
            "mapped_len": r["mapped_len"], "label_interval_internal_gap": internal_gap,
            "auth_start": r["auth_start"], "auth_end": r["auth_end"],
            "mapping_quality": r["mapping_quality"], "unmapped_residue_count": unmapped,
            "seq_prefix10": s["seq_prefix10"], "seq_suffix10": s["seq_suffix10"],
            "source_pages": s["source_pages"], "s2_mask_basis": s["mask_basis"],
            "s2_review_status": s["review_status"],
            "lit_range_status": e["literature_range_status"],
            "lit_ranges_auth": e["literature_residue_ranges_auth"],
            "range_field_consistency": e["range_field_consistency"],
            "lit_label_type": e["literature_region_label_type"],
            "lit_label_precision": e["literature_region_label_precision"],
            "lit_label_decision": e["literature_region_label_decision"],
            "lit_label_reason": e["literature_region_label_reason"],
            "endpoint_identity": t["endpoint_identity"],
            "endpoint_chain_match_diag": m.get("diagnosis", "no_ecm_row"),
            "release_state_status": rel_state,
            "region_task_tier": tier,
            "region_target_semantics": semantics,
            "residue_expansion_source": "residue_mapping_v2.tsv(逐残基 pdf→label_seq/auth；UniProt 标准坐标待序列导入)",
            "source_version": SOURCE_VERSION,
            "created_at": CREATED_AT,
        })

    # ---- 断言 ----
    assert len(out) == 203, f"行数 {len(out)} != 203"
    tiers = Counter(x["region_task_tier"] for x in out)
    assert tiers == Counter({"pending_review": 96, "fine_extension": 57,
                             "coarse_candidate": 34, "fine_candidate": 16}), tiers
    mq = Counter(x["mapping_quality"] for x in out)
    assert mq == Counter({"full": 167, "partial": 36}), mq
    strict_rows = [x for x in out if x["release_state_status"] == "strict_state_candidate"]
    assert len(strict_rows) == 20, len(strict_rows)
    fine = [x for x in out if x["region_task_tier"] == "fine_candidate"]
    assert {x["pair_id"] for x in fine} <= {x["pair_id"] for x in strict_rows}, "fine_candidate 超出 strict 对集合"
    strict_no_fine = ({x["pair_id"] for x in strict_rows}
                      - {x["pair_id"] for x in fine})
    if problems:
        print("PROBLEMS:")
        for p in problems:
            print("  -", p)
        return 1

    with open(OUT, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields, delimiter="\t", lineterminator="\n")
        w.writeheader()
        w.writerows(out)

    print("WROTE", OUT, len(out), "rows x", len(fields), "cols")
    print("tier 分布:", dict(tiers))
    print("mapping_quality:", dict(mq))
    print("strict 行:", len(strict_rows), "| fine_candidate 对数:", len({x['pair_id'] for x in fine}))
    print("strict 但无 fine 区段的对:", sorted(strict_no_fine))
    print("partial 缺口分布:", dict(Counter(x["unmapped_residue_count"] for x in out if x["mapping_quality"] == "partial")))
    print("lit_label_decision:", dict(Counter(x["lit_label_decision"] for x in out)))
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
