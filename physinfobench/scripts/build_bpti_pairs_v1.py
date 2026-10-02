#!/usr/bin/env python3
"""build_bpti_pairs_v1.py — P1.08 BPTI 共价约束系统构建

输入（本地 raw，2026-09-23 批次）：
  epmc_krokoszynska1998.json   Krokoszynska & Dadlez 1998 JMB 摘要（PMID 9466927；全文付费墙 inEPMC=N）
  search_snapshot.json         RCSB 检索快照（引用 0 命中；P00974 池 128）
  rcsb_wt_refs.json            WT BPTI 参考结构元数据（4PTI/5PTI/6PTI）
  uniprot_P00974.json          BPTI 标准序列（100 aa 前体；成熟 58）
输出：data/curated/bpti_pairs.tsv
口径（依据摘要原文，不虚构）：
  - 15 个单二硫变体（3 native + 12 non-native）已表达；仅 4 个有功能数据（3 native + non-native Cys5-Cys51）
    ——非 native 的其余 11 个组合摘要未列出，不登记具体配对，只登记总数事实；
  - 终点=功能（β-trypsin 结合抑制常数），测量条件 pH 4.0；不得换名为结构终点；
  - 同变体氧化/还原对照存在（摘要："in some of the variants the binding constants were higher for the
    reduced rather than for the oxidized form"）——具体变体清单 pending_fulltext；
  - Krokoszynska 1998 无结构沉积（检索 0 命中）——变体结构证据缺失；WT 参考结构仅登记，不作变体表示。
"""
import csv
import json
from datetime import datetime
from pathlib import Path

RAW = Path("data/raw/bpti/2026-09-23")
OUT = Path("data/curated/bpti_pairs.tsv")
CREATED_AT = datetime.now().strftime("%Y-%m-%d %H:%M")
SOURCE_VERSION = "raw/bpti/2026-09-23（EPMC 摘要 PMID 9466927 + RCSB 检索快照 + WT refs + UniProt REST P00974）"

FIELDS = ["row_kind", "variant_id", "disulfide_pair", "disulfide_class", "cys_to_ala_note",
          "oxidation_state", "endpoint_type", "endpoint_note",
          "condition_type", "condition_value", "condition_unit",
          "redox_pair_semantics", "structure_available", "structure_ref",
          "doi", "evidence_level", "evidence_location", "evidence_quote",
          "source_version", "created_at"]

Q_15 = "All 15 single-disulfide variants of BPTI (three native and 12 non-native combinations) have been expressed in Escherichia coli"
Q_4ACT = "Four of these variants are shown here to inhibit bovine beta-trypsin: three of them contain native and one non-native (Cys5-Cys51) disulfide"
Q_PH = "measurements were performed at pH 4.0, at which trypsin activity is low"
Q_REDOX = "in some of the variants the binding constants were found to be higher for the reduced rather than for the oxidized form"
Q_RED_NATIVE = "Also for the fully reduced native BPTI, determined here"


def main() -> int:
    ep = json.load(open(RAW / "epmc_krokoszynska1998.json"))["resultList"]["result"][0]
    absT = ep["abstractText"]
    snap = json.load(open(RAW / "search_snapshot.json"))
    wt = json.load(open(RAW / "rcsb_wt_refs.json"))
    uni = json.load(open(RAW / "uniprot_P00974.json"))
    for q in [Q_15, Q_4ACT, Q_PH, Q_REDOX, Q_RED_NATIVE]:
        assert q in absT, q[:50]
    assert snap["cite_search_total"] == 0 and snap["p00974_pool_total"] == 128
    assert len(uni["sequence"]["value"]) == 100
    assert wt["5PTI"]["res"] == 1.0 and str(wt["5PTI"]["year"]) == "1984"
    assert ep["inEPMC"] == "N" and str(ep.get("pmid")) == "9466927"

    SV = SOURCE_VERSION
    base = {"doi": "10.1006/jmbi.1997.1460", "evidence_level": "abstract_only",
            "evidence_location": "EuropePMC 摘要（PMID 9466927）", "source_version": SV, "created_at": CREATED_AT}
    rows = [
        # 实验登记
        {**base, "row_kind": "experiment", "variant_id": "", "disulfide_pair": "", "disulfide_class": "",
         "cys_to_ala_note": "", "oxidation_state": "", "endpoint_type": "function_trypsin_binding",
         "endpoint_note": "结合抑制常数（功能终点）；非结构终点，不得换名",
         "condition_type": "pH", "condition_value": "4.0", "condition_unit": "",
         "redox_pair_semantics": "", "structure_available": "", "structure_ref": "",
         "evidence_quote": Q_PH},
        # 15 变体总登记（事实行）
        {**base, "row_kind": "variant_family", "variant_id": "BPTI-SD-ALL15", "disulfide_pair": "3 native + 12 non-native（非 native 组合未逐一列于摘要）",
         "disulfide_class": "single_disulfide", "cys_to_ala_note": "每变体其余 4 个 Cys→Ala",
         "oxidation_state": "ox_and_red", "endpoint_type": "function_trypsin_binding",
         "endpoint_note": "仅 4/15 有功能数据（见下）；其余 11 变体终点未知",
         "condition_type": "pH", "condition_value": "4.0", "condition_unit": "",
         "redox_pair_semantics": "同变体 ox vs red 对照存在（具体变体清单 pending_fulltext）",
         "structure_available": "no", "structure_ref": "",
         "evidence_quote": Q_15},
        # 4 个有功能数据的变体
        {**base, "row_kind": "variant_functional", "variant_id": "BPTI-SD-5-55", "disulfide_pair": "Cys5-Cys55",
         "disulfide_class": "native", "cys_to_ala_note": "C14A/C30A/C38A/C51A",
         "oxidation_state": "ox_and_red", "endpoint_type": "function_trypsin_binding",
         "endpoint_note": "结合常数 ≥2 个数量级低于 native（摘要总述）",
         "condition_type": "pH", "condition_value": "4.0", "condition_unit": "",
         "redox_pair_semantics": "候选（归属 pending_fulltext）", "structure_available": "no", "structure_ref": "",
         "evidence_quote": Q_4ACT},
        {**base, "row_kind": "variant_functional", "variant_id": "BPTI-SD-14-38", "disulfide_pair": "Cys14-Cys38",
         "disulfide_class": "native", "cys_to_ala_note": "C5A/C30A/C51A/C55A",
         "oxidation_state": "ox_and_red", "endpoint_type": "function_trypsin_binding",
         "endpoint_note": "同上", "condition_type": "pH", "condition_value": "4.0", "condition_unit": "",
         "redox_pair_semantics": "候选（归属 pending_fulltext）", "structure_available": "no", "structure_ref": "",
         "evidence_quote": Q_4ACT},
        {**base, "row_kind": "variant_functional", "variant_id": "BPTI-SD-30-51", "disulfide_pair": "Cys30-Cys51",
         "disulfide_class": "native", "cys_to_ala_note": "C5A/C14A/C38A/C55A",
         "oxidation_state": "ox_and_red", "endpoint_type": "function_trypsin_binding",
         "endpoint_note": "同上", "condition_type": "pH", "condition_value": "4.0", "condition_unit": "",
         "redox_pair_semantics": "候选（归属 pending_fulltext）", "structure_available": "no", "structure_ref": "",
         "evidence_quote": Q_4ACT},
        {**base, "row_kind": "variant_functional", "variant_id": "BPTI-SD-5-51", "disulfide_pair": "Cys5-Cys51",
         "disulfide_class": "non_native", "cys_to_ala_note": "C14A/C30A/C38A/C55A",
         "oxidation_state": "ox_and_red", "endpoint_type": "function_trypsin_binding",
         "endpoint_note": "唯一有功能数据的 non-native 单二硫变体",
         "condition_type": "pH", "condition_value": "4.0", "condition_unit": "",
         "redox_pair_semantics": "候选（归属 pending_fulltext）", "structure_available": "no", "structure_ref": "",
         "evidence_quote": Q_4ACT},
        # fully reduced native（有独立测定）
        {**base, "row_kind": "variant_functional", "variant_id": "BPTI-NATIVE-FULLRED", "disulfide_pair": "none（完全还原）",
         "disulfide_class": "reduced_native", "cys_to_ala_note": "无置换（原生序列）",
         "oxidation_state": "reduced", "endpoint_type": "function_trypsin_binding",
         "endpoint_note": "完全还原 native BPTI 的结合常数在本文测定",
         "condition_type": "pH", "condition_value": "4.0", "condition_unit": "",
         "redox_pair_semantics": "与氧化 native 构成跨条件对照（native 无二硫键差异但氧化态不同）",
         "structure_available": "no", "structure_ref": "",
         "evidence_quote": Q_RED_NATIVE},
        # 同变体氧化/还原对照语义行
        {**base, "row_kind": "redox_pair_registry", "variant_id": "REDOX-PAIRS-1998",
         "disulfide_pair": "部分单二硫变体", "disulfide_class": "", "cys_to_ala_note": "",
         "oxidation_state": "ox_vs_red", "endpoint_type": "function_trypsin_binding",
         "endpoint_note": "同变体氧化态 vs 还原态的结合常数对照——存在性由摘要证明；哪些变体/数值 pending_fulltext",
         "condition_type": "pH", "condition_value": "4.0", "condition_unit": "",
         "redox_pair_semantics": "strict_same_variant_ox_red",
         "structure_available": "no", "structure_ref": "",
         "evidence_quote": Q_REDOX},
        # WT 结构参考（database 证据；非变体表示）
        {"row_kind": "wt_structure_reference", "variant_id": "BPTI-WT", "disulfide_pair": "native 三键（5-55/14-38/30-51）",
         "disulfide_class": "native", "cys_to_ala_note": "",
         "oxidation_state": "oxidized", "endpoint_type": "structure_reference_only",
         "endpoint_note": "仅作结构参考登记；变体序列差 4×Ala 置换，WT 结构不得直接充当变体表示（P4.08 决策）",
         "condition_type": "", "condition_value": "", "condition_unit": "",
         "redox_pair_semantics": "", "structure_available": "yes",
         "structure_ref": ";".join(f"{p}({wt[p]['res']}A,{wt[p]['year']})" for p in ["4PTI", "5PTI", "6PTI"]),
         "doi": ";".join((str(wt[p]["doi"]) or "na") for p in ["4PTI", "5PTI", "6PTI"]),
         "evidence_level": "database", "evidence_location": "RCSB 4PTI/5PTI/6PTI",
         "evidence_quote": " | ".join(wt[p]["title"] for p in ["4PTI", "5PTI", "6PTI"]),
         "source_version": SV, "created_at": CREATED_AT},
    ]

    vf = [r for r in rows if r["row_kind"] == "variant_functional"]
    assert len(vf) == 5  # 4 单二硫 + fully reduced native
    assert sum(1 for r in vf if r["disulfide_class"] == "native") == 3
    assert sum(1 for r in vf if r["disulfide_class"] == "non_native") == 1
    assert all(r["endpoint_type"] == "function_trypsin_binding" for r in vf)
    assert not any(r["structure_available"] == "yes" and r["row_kind"] != "wt_structure_reference" for r in rows)
    assert len(rows) == 9

    with open(OUT, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS, delimiter="\t", lineterminator="\n")
        w.writeheader()
        w.writerows(rows)
    print("WROTE", OUT, len(rows), "rows x", len(FIELDS), "cols")
    print("构成: experiment 1 | variant_family 1 | variant_functional 5 | redox_pair_registry 1 | wt_structure_reference 1")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
