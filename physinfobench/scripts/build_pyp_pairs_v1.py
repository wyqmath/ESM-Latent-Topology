#!/usr/bin/env python3
"""build_pyp_pairs_v1.py — P1.06 PYP 配对系统构建（v1.1：审核整改版）

输入（本地 raw，2026-09-22 批次）：
  konold2020_ncomms11598.xml        Konold 2020 全文（EuropePMC PMC7447820，OA）
  tenboer2014_science1259357.xml    Tenboer 2014 全文获取失败文件（EPMC 500 错误页，留作受阻证据）
  uniprot_P16113.json               PYP 标准序列（125 aa）
  rcsb_entry_metadata.json          103 个 P16113 条目引用/方法元数据
  rcsb_entry_full_4WL9.json / rcsb_entry_full_4WLA.json   含 struct_keywords 的完整条目（状态证据 raw 锚）
输出：data/curated/pyp_pairs.tsv

v1.1 整改（审核 2026-09-22 23:48 两项阻塞 + C1/C2/C3）：
  - PMID 更正 25362349→25477465（Tenboer 正确 PMID，EPMC+RCSB 双源核实）
  - 引文章节定位改为程序化解析 XML <sec> 路径（不再手写）；引文归一化与全文一致
  - condition_evidence 补 pH 混杂（PYP S pH8 vs PYP C pH6.5）与 Yeremenko et al. 合并数据出处
  - struct_keywords 状态证据在 raw 完整条目 JSON 上真断言（4WL9 含 Dark structure；4WLA 不含）
  - 条目池断言 103（分页全量）
"""
import csv
import json
import re
from datetime import datetime
from pathlib import Path
import xml.etree.ElementTree as ET

RAW = Path("data/raw/pyp/2026-09-22")
OUT = Path("data/curated/pyp_pairs.tsv")
CREATED_AT = datetime.now().strftime("%Y-%m-%d %H:%M")
SOURCE_VERSION = "raw/pyp/2026-09-22（Konold XML EPMC PMC7447820 + Tenboer 摘要 EPMC PMID 25477465 + RCSB Data API + UniProt REST P16113）"

FIELDS = ["row_kind", "pair_group_id", "experiment_id", "sample_id", "pdb", "chain",
          "state", "state_basis", "condition_type", "condition_value", "condition_unit",
          "environment", "assay_method", "resolution_a",
          "sequence_check", "mutation_status", "construct_note",
          "doi", "evidence_level", "evidence_location", "evidence_quote",
          "source_version", "created_at"]

QUOTES = {
    "q48ms": "with the latest time point at 48 ms (crystal) or 1 s (solution)",
    "qpcps": "for both PYP in the crystalline form (PYP C ) and in solution (PYP S )",
    "qwindow": "were recorded in the 380–570 nm spectral window",
    "qph": "PYP S was measured at pH 8, PYP C at pH 6.5",
    "qyerem": "augmented these data, with data from Yeremenko et al. 19 , collected from microseconds to 1 s",
}


def norm(s: str) -> str:
    return re.sub(r"\s+", " ", s)


def sec_text_map(root):
    """构建 (sec 路径, 元素类型) -> 归一化文本 的映射；归一化方式与全文一致（标签→空格+折叠）"""
    out = []

    def strip_norm(el):
        # 与全文归一化一致：元素间插入空格再折叠
        parts = []
        for piece in el.itertext():
            parts.append(piece)
        return norm(" " + " ".join(parts) + " ")

    def walk(el, path):
        for child in el:
            tag = child.tag.split("}")[-1]
            if tag == "sec":
                t = child.find(".//{*}title")
                tt = norm("".join(t.itertext())) if t is not None else "(untitled)"
                walk(child, path + [tt])
            elif tag in ("p", "caption"):
                kind = "caption" if tag == "caption" else "p"
                out.append((path, kind, strip_norm(child)))
            else:
                walk(child, path)

    walk(root, [])
    return out


def locate(quote: str):
    """在 sec 文本映射中定位引文，返回 (路径串, 元素类型)；必须恰好一处"""
    hits = [( " > ".join(path) if path else "Front matter", kind)
            for path, kind, txt in SEC_TEXTS if quote in txt]
    assert len(hits) == 1, f"引文定位异常（{len(hits)} 处）: {quote[:50]}"
    loc, kind = hits[0]
    if kind == "caption":
        loc += "（图注）"
    return loc


def main() -> int:
    konold_xml = (RAW / "konold2020_ncomms11598.xml").read_text()
    global SEC_TEXTS
    SEC_TEXTS = sec_text_map(ET.fromstring(konold_xml))
    full_txt = norm(re.sub(r"<[^>]+>", " ", konold_xml))
    for k, q in QUOTES.items():
        assert q in full_txt, f"引文缺失: {k}"

    meta = json.load(open(RAW / "rcsb_entry_metadata.json"))
    assert len(meta) == 103, len(meta)  # 分页全量
    uniprot = json.load(open(RAW / "uniprot_P16113.json"))
    assert len(uniprot["sequence"]["value"]) == 125

    # 状态证据真断言（raw 完整条目）
    kw = {}
    for pdb in ["4WL9", "4WLA"]:
        e = json.load(open(RAW / f"rcsb_entry_full_{pdb}.json"))
        kw[pdb] = e.get("struct_keywords", {}).get("text", "")
    assert "Dark structure" in kw["4WL9"] and "Dark structure" not in kw["4WLA"]
    for pdb, doi, meth, res, yr in [("4WL9", "10.1126/science.1259357", "X-ray", 1.6, "2014"),
                                    ("4WLA", "10.1126/science.1259357", "X-ray", 1.6, "2014")]:
        m = meta[pdb]
        assert m["doi"] == doi and m["method"] == meth and float(m["resolution"]) == res and str(m["year"]) == yr
    tb = (RAW / "tenboer2014_science1259357.xml").read_text()
    assert "Internal Server Error" in tb  # 受阻证据

    LOC = {k: locate(q) for k, q in QUOTES.items()}
    print("章节定位:", json.dumps(LOC, ensure_ascii=False, indent=1))

    SV = SOURCE_VERSION
    spec_common = {"condition_type": "environment", "assay_method": "transient absorption spectroscopy",
                   "doi": "10.1038/s41467-020-18065-9", "evidence_level": "fulltext",
                   "source_version": SV, "created_at": CREATED_AT}
    rows = [
        {"row_kind": "experiment", "pair_group_id": "", "experiment_id": "EXP-PYP-2014-SFX",
         "sample_id": "", "pdb": "", "chain": "", "state": "", "state_basis": "TR-SFX @ XFEL, microcrystals, 差异电子密度解析中间态（摘要）",
         "condition_type": "illumination_pump", "condition_value": "pending_fulltext", "condition_unit": "",
         "environment": "microcrystal (serial femtosecond crystallography)", "assay_method": "X-ray free electron laser",
         "resolution_a": "1.6", "sequence_check": "", "mutation_status": "",
         "construct_note": "WT P16113 125aa（两条目 SIFTS 双向全覆盖）",
         "doi": "10.1126/science.1259357", "evidence_level": "abstract_and_database",
         "evidence_location": "EuropePMC 摘要（PMID 25477465）+ RCSB 条目关键词",
         "evidence_quote": "obtained high-resolution, time-resolved difference electron density maps … structures of reaction intermediates to a resolution of 1.6 angstroms",
         "source_version": SV, "created_at": CREATED_AT},
        {"row_kind": "experiment", "pair_group_id": "", "experiment_id": "EXP-PYP-2020-SPEC",
         "sample_id": "", "pdb": "", "chain": "", "state": "", "state_basis": "瞬态吸收光谱（PYP 晶体 vs 溶液全程光循环对比）——非结构实验，作条件语义/混杂证据",
         "condition_type": "illumination", "condition_value": "UV-visible transient 380-570nm", "condition_unit": "nm",
         "environment": "crystal (PYP C) vs solution (PYP S)", "assay_method": "transient absorption spectroscopy",
         "resolution_a": "", "sequence_check": "", "mutation_status": "",
         "construct_note": "同一 WT PYP 两种物理形态；自测数据 100 fs–0.3 ms，微秒至 1 s 段合并自 Yeremenko et al. 2006（ref 19）",
         "doi": "10.1038/s41467-020-18065-9", "evidence_level": "fulltext",
         "evidence_location": f"Konold2020 {LOC['qyerem']}",
         "evidence_quote": QUOTES["qyerem"],
         "source_version": SV, "created_at": CREATED_AT},
        {"row_kind": "pair_member", "pair_group_id": "PG-PYP-001", "experiment_id": "EXP-PYP-2014-SFX",
         "sample_id": "PYP-2014-DARK", "pdb": "4WL9", "chain": "A",
         "state": "dark", "state_basis": "RCSB struct_keywords 含 'Dark structure'（raw rcsb_entry_full_4WL9.json）",
         "condition_type": "illumination", "condition_value": "0", "condition_unit": "boolean",
         "environment": "microcrystal (TR-SFX)", "assay_method": "X-ray free electron laser",
         "resolution_a": "1.6",
         "sequence_check": "uniprot_sifts_full_coverage_bidirectional", "mutation_status": "identity_check_pending",
         "construct_note": "P16113 全长 125aa 单链",
         "doi": "10.1126/science.1259357", "evidence_level": "abstract_and_database",
         "evidence_location": "RCSB 4WL9 struct_keywords.text（raw 完整条目）",
         "evidence_quote": kw["4WL9"],
         "source_version": SV, "created_at": CREATED_AT},
        {"row_kind": "pair_member", "pair_group_id": "PG-PYP-001", "experiment_id": "EXP-PYP-2014-SFX",
         "sample_id": "PYP-2014-LIGHT-INT", "pdb": "4WLA", "chain": "A",
         "state": "light_intermediate", "state_basis": "RCSB struct_keywords（无 dark 标记的 TR-SFX 中间态）；具体时间延迟与泵浦参数 pending_fulltext",
         "condition_type": "illumination_pump;delay", "condition_value": "pending_fulltext", "condition_unit": "",
         "environment": "microcrystal (TR-SFX)", "assay_method": "X-ray free electron laser",
         "resolution_a": "1.6",
         "sequence_check": "uniprot_sifts_full_coverage_bidirectional", "mutation_status": "identity_check_pending",
         "construct_note": "P16113 全长 125aa 单链",
         "doi": "10.1126/science.1259357", "evidence_level": "abstract_and_database",
         "evidence_location": "RCSB 4WLA struct_keywords.text（raw 完整条目）+ Tenboer 摘要",
         "evidence_quote": kw["4WLA"],
         "source_version": SV, "created_at": CREATED_AT},
        {**spec_common, "pair_group_id": "PG-PYP-001",
         "row_kind": "condition_evidence", "experiment_id": "EXP-PYP-2020-SPEC",
         "sample_id": "PYP-2020-CRYSTAL-SPEC", "pdb": "", "chain": "",
         "state": "photocycle_spectroscopy",
         "state_basis": "PYP C（晶体）瞬态吸收；最新时间点 48 ms（含 Yeremenko 2006 合并数据）；测量 pH 6.5",
         "condition_value": "crystal", "condition_unit": "",
         "environment": "crystal (PYP C, pH 6.5)", "resolution_a": "",
         "sequence_check": "", "mutation_status": "", "construct_note": "WT",
         "evidence_location": f"Konold2020 {LOC['q48ms']} / {LOC['qph']}",
         "evidence_quote": QUOTES["q48ms"] + " || " + QUOTES["qph"]},
        {**spec_common, "pair_group_id": "PG-PYP-001",
         "row_kind": "condition_evidence", "experiment_id": "EXP-PYP-2020-SPEC",
         "sample_id": "PYP-2020-SOLUTION-SPEC", "pdb": "", "chain": "",
         "state": "photocycle_spectroscopy",
         "state_basis": "PYP S（溶液）瞬态吸收；最新时间点 1 s（含 Yeremenko 2006 合并数据）；测量 pH 8",
         "condition_value": "solution", "condition_unit": "",
         "environment": "solution (PYP S, pH 8)", "resolution_a": "",
         "sequence_check": "", "mutation_status": "", "construct_note": "WT",
         "evidence_location": f"Konold2020 {LOC['qpcps']} / {LOC['qph']}",
         "evidence_quote": QUOTES["qpcps"] + " || " + QUOTES["qph"]},
    ]

    pair_members = [r for r in rows if r["row_kind"] == "pair_member"]
    assert len(pair_members) == 2 and len({r["pair_group_id"] for r in pair_members}) == 1
    assert {r["state"] for r in pair_members} == {"dark", "light_intermediate"}
    assert len({r["pdb"] for r in pair_members}) == 2
    assert all(r["sequence_check"].startswith("uniprot_sifts") for r in pair_members)
    assert {r["experiment_id"] for r in rows if r["row_kind"] == "experiment"} == {"EXP-PYP-2014-SFX", "EXP-PYP-2020-SPEC"}

    with open(OUT, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS, delimiter="\t", lineterminator="\n")
        w.writeheader()
        w.writerows(rows)
    print("WROTE", OUT, len(rows), "rows x", len(FIELDS), "cols")
    print("构成: experiment 2 | pair_member 2 (PG-PYP-001) | condition_evidence 2")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
