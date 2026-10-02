#!/usr/bin/env python3
"""build_rnase_a_pairs_v1.py — P1.07 RNase A 配对系统构建

输入（本地 raw，2026-09-23 批次）：
  rcsb_entry_full.json       6 条目 entry 级元数据（含 full_text 检索假阳性证据 9R6Q/R/P）
  rcsb_entry_full_1A2W.json  N-swap dimer 完整条目
  rnase_entities.json        4 个关键 RNase A 实体（1F0V/2、1A2W/1、1FS3/1、1JS0/1）
  epmc_liu2001.json          Liu 2001 NSMB 摘要（PMID 11224563；全文付费墙 inEPMC=N）
  epmc_liu1998.json          Liu 1998 PNAS 摘要（PMID 9520384；N-swap 残基 1-15 出处）
  uniprot_P61823.json        RNase A 标准序列（150 aa 含信号肽；成熟蛋白 124）
输出：data/curated/rnase_a_pairs.tsv
口径：单体 vs 结构域交换二聚体主配对×2（C-swap major=1F0V；N-swap minor=1A2W）；
     trimer 1JS0 登记 extension；9R6Q/R/P 排除（full_text 引用列表假阳性）；
     链数与组装状态同义→标注 synonymous_with_chain_count（输入充分性对照边界）。
"""
import csv
import json
from datetime import datetime
from pathlib import Path

RAW = Path("data/raw/rnase_a/2026-09-23")
OUT = Path("data/curated/rnase_a_pairs.tsv")
CREATED_AT = datetime.now().strftime("%Y-%m-%d %H:%M")
SOURCE_VERSION = "raw/rnase_a/2026-09-23（EPMC 摘要×2 + RCSB entry/polymer_entity + UniProt REST P61823）"

FIELDS = ["row_kind", "pair_group_id", "experiment_id", "sample_id", "pdb", "entity", "chains",
          "oligomer_state", "swapped_segment", "segment_residues", "segment_evidence",
          "condition_type", "condition_value", "condition_unit", "environment", "assay_method", "resolution_a",
          "sequence_check", "mutation_status", "synonymous_with_chain_count",
          "doi", "evidence_level", "evidence_location", "evidence_quote",
          "source_version", "created_at"]


def main() -> int:
    ents = json.load(open(RAW / "rnase_entities.json"))
    meta = json.load(open(RAW / "rcsb_entry_full.json"))
    meta["1A2W"] = json.load(open(RAW / "rcsb_entry_full_1A2W.json")).get("meta_stub", {}) or json.load(open(RAW / "rcsb_entry_full_1A2W.json"))
    # 从完整条目提取统一字段
    if "doi" not in meta["1A2W"]:
        e = json.load(open(RAW / "rcsb_entry_full_1A2W.json"))
        cit = e.get("rcsb_primary_citation", {}) or {}
        meta["1A2W"] = {"doi": cit.get("pdbx_database_id_DOI", ""),
                        "title": (e.get("struct", {}).get("title") or "")[:100],
                        "year": cit.get("year", ""), "journal": cit.get("journal_abbrev", ""),
                        "method": (e.get("rcsb_entry_info", {}) or {}).get("experimental_method", ""),
                        "resolution": ((e.get("rcsb_entry_info", {}).get("resolution_combined") or [None])[0])}
    liu01 = json.load(open(RAW / "epmc_liu2001.json"))
    liu98 = json.load(open(RAW / "epmc_liu1998.json"))
    uni = json.load(open(RAW / "uniprot_P61823.json"))
    assert len(uni["sequence"]["value"]) == 150

    Q_MILD = "forms two types of dimers (a major and a minor component) upon concentration in mild acid"
    Q_CSWAP = "the major dimer forms by swapping its C-terminal beta-strand"
    Q_NSWAP = "The dimer is 3D domain-swapped. The N-terminal helix (residues 1-15) of each subunit is swapped"
    assert Q_MILD in liu01["abstractText"] and Q_CSWAP in liu01["abstractText"]
    assert Q_NSWAP in liu98["abstractText"]

    # 实体断言：全部 P61823 全覆盖、成熟 124aa
    for k, v in ents.items():
        assert v["uniprots"] == ["P61823"] and v["coverage"][0][0] == 1.0 and v["seq_len"] == 124, (k, v)
    assert ents["1F0V_2"]["auth_chains"] == ["A", "B", "C", "D"]   # 2 个 dimer 拷贝
    assert ents["1A2W_1"]["auth_chains"] == ["A", "B"]
    assert ents["1FS3_1"]["auth_chains"] == ["A"]
    assert ents["1JS0_1"]["auth_chains"] == ["A", "B", "C"]
    # entry 级断言
    assert meta["1F0V"]["doi"] == "10.1038/84941" and str(meta["1F0V"]["year"]) == "2001"
    assert meta["1A2W"]["doi"] == "10.1073/pnas.95.7.3437" and str(meta["1A2W"]["year"]) == "1998"
    for fp in ["9R6Q", "9R6R", "9R6P"]:
        assert "s41564" in meta[fp]["doi"]  # 假阳性：冠状病毒论文

    SV = SOURCE_VERSION
    def member(pg, sid, pdb, entity, chains, oligo, swapped="", seg="", seg_ev="", res="per_paper",
               env="crystal", res_a="see_raw", syn="", doi="", level="database", loc="", quote=""):
        return {"row_kind": "pair_member", "pair_group_id": pg, "experiment_id": "EXP-RNA-1998-2001",
                "sample_id": sid, "pdb": pdb, "entity": entity, "chains": chains,
                "oligomer_state": oligo, "swapped_segment": swapped, "segment_residues": seg,
                "segment_evidence": seg_ev,
                "condition_type": "concentration_in_mild_acid", "condition_value": "per_paper", "condition_unit": "",
                "environment": env, "assay_method": "X-ray diffraction", "resolution_a": res_a,
                "sequence_check": "uniprot_sifts_full_coverage_mature124", "mutation_status": "identity_check_pending",
                "synonymous_with_chain_count": syn,
                "doi": doi, "evidence_level": level, "evidence_location": loc, "evidence_quote": quote,
                "source_version": SV, "created_at": CREATED_AT}

    rows = [
        # 实验/条件证据行
        {"row_kind": "experiment", "pair_group_id": "", "experiment_id": "EXP-RNA-1998-2001",
         "sample_id": "", "pdb": "", "entity": "", "chains": "",
         "oligomer_state": "", "swapped_segment": "", "segment_residues": "", "segment_evidence": "",
         "condition_type": "concentration_in_mild_acid", "condition_value": "mild acid（浓缩条件；具体 pH/浓度 pending_fulltext）", "condition_unit": "",
         "environment": "solution（结晶前组装）→ crystal", "assay_method": "", "resolution_a": "",
         "sequence_check": "", "mutation_status": "", "synonymous_with_chain_count": "",
         "doi": "10.1038/84941", "evidence_level": "abstract_only",
         "evidence_location": "EuropePMC 摘要（PMID 11224563）",
         "evidence_quote": Q_MILD,
         "source_version": SV, "created_at": CREATED_AT},
        # 主配对 1：单体 vs C-swap major dimer
        member("PG-RNA-001", "RNA-MONOMER-1FS3", "1FS3", "1", "A", "monomer",
               swapped="", seg="", seg_ev="", env="crystal", res_a=str(meta["1FS3"]["resolution"]), syn="",
               doi="10.1110/ps.ps.31102", level="database", loc="RCSB 1FS3 struct/keywords（WT bovine pancreatic RNase A）",
               quote=meta["1FS3"]["title"]),
        member("PG-RNA-001", "RNA-CSWAP-1F0V", "1F0V", "2", "A;B;C;D（2 个 dimer 拷贝）", "dimer_C_swap_major",
               swapped="C-terminal beta-strand", seg="pending_fulltext",
               seg_ev="Liu2001 摘要（定性 C-swap；残基号在付费正文）",
               env="crystal", res_a="见 raw", syn="yes",
               doi="10.1038/84941", level="abstract_only", loc="EuropePMC 摘要（PMID 11224563）+ RCSB 1F0V",
               quote=Q_CSWAP),
        # 主配对 2：单体 vs N-swap minor dimer
        member("PG-RNA-002", "RNA-MONOMER-1FS3-B", "1FS3", "1", "A", "monomer",
               swapped="", seg="", seg_ev="", env="crystal", res_a=str(meta["1FS3"]["resolution"]), syn="",
               doi="10.1110/ps.ps.31102", level="database", loc="RCSB 1FS3",
               quote=meta["1FS3"]["title"]),
        member("PG-RNA-002", "RNA-NSWAP-1A2W", "1A2W", "1", "A;B", "dimer_N_swap_minor",
               swapped="N-terminal alpha-helix", seg="1-15", seg_ev="Liu1998 PNAS 摘要（含残基号）",
               env="crystal", res_a="见 raw", syn="yes",
               doi="10.1073/pnas.95.7.3437", level="abstract_only", loc="EuropePMC 摘要（PMID 9520384）+ RCSB 1A2W",
               quote=Q_NSWAP),
        # 登记行：trimer（extension，不进主配对）
        {"row_kind": "registered_extension", "pair_group_id": "", "experiment_id": "EXP-RNA-1998-2001",
         "sample_id": "RNA-TRIMER-1JS0", "pdb": "1JS0", "entity": "1", "chains": "A;B;C",
         "oligomer_state": "trimer_minor", "swapped_segment": "N-terminal（域交换三聚体）", "segment_residues": "pending_fulltext",
         "segment_evidence": "RCSB 标题（3D domain-swapped RNase A minor trimer）",
         "condition_type": "concentration_in_mild_acid", "condition_value": "per_paper", "condition_unit": "",
         "environment": "crystal", "assay_method": "X-ray diffraction", "resolution_a": "见 raw",
         "sequence_check": "uniprot_sifts_full_coverage_mature124", "mutation_status": "identity_check_pending",
         "synonymous_with_chain_count": "yes",
         "doi": "10.1110/ps.36602", "evidence_level": "database",
         "evidence_location": "RCSB 1JS0", "evidence_quote": meta["1JS0"]["title"],
         "source_version": SV, "created_at": CREATED_AT},
        # 排除行：full_text 检索假阳性
    ] + [{"row_kind": "excluded", "pair_group_id": "", "experiment_id": "",
         "sample_id": f"EXCL-{p}", "pdb": p, "entity": "", "chains": "",
         "oligomer_state": "", "swapped_segment": "", "segment_residues": "", "segment_evidence": "",
         "condition_type": "", "condition_value": "", "condition_unit": "",
         "environment": "", "assay_method": "", "resolution_a": "",
         "sequence_check": "", "mutation_status": "", "synonymous_with_chain_count": "",
         "doi": meta[p]["doi"], "evidence_level": "n/a",
         "evidence_location": "RCSB full_text 检索 10.1038/84941",
         "evidence_quote": f"引用列表命中假阳性：{meta[p]['title'][:60]}（非 Liu2001 结构）",
         "source_version": SV, "created_at": CREATED_AT}
        for p in ["9R6Q", "9R6R", "9R6P"]]

    pairs = [r for r in rows if r["row_kind"] == "pair_member"]
    assert {r["pair_group_id"] for r in pairs} == {"PG-RNA-001", "PG-RNA-002"}
    for pg in ["PG-RNA-001", "PG-RNA-002"]:
        mem = [r for r in pairs if r["pair_group_id"] == pg]
        assert len(mem) == 2 and {m["oligomer_state"] for m in mem} == {"monomer", mem[1]["oligomer_state"]}
        assert all(m["pdb"] for m in mem)
    assert sum(1 for r in rows if r["row_kind"] == "excluded") == 3

    with open(OUT, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS, delimiter="\t", lineterminator="\n")
        w.writeheader()
        w.writerows(rows)
    print("WROTE", OUT, len(rows), "rows x", len(FIELDS), "cols")
    print("构成: experiment 1 | pair 2×2 | extension 1 | excluded 3")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
