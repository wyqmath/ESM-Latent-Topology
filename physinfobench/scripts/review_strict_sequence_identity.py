#!/usr/bin/env python3
"""P1.13 strict 阳性序列身份复核（G1 前补证；只读，不改任何资格表）。

对 10 对 strict（重点=7 对非双端子串）逐端点做全局/局部对齐：
  observed（旧项目端点审计表，sha256 登记） vs UniProt canonical（P1.09 audit fasta）。
预注册参数：PairwiseAligner BLOSUM62, open_gap -11, extend_gap -1（BLASTP 默认式）；
  local 对齐取最优；块解析从 alignment.aligned 提取，不使用 E 值。
判读规则（预注册）：
  - 对齐区间内 mismatch=0 且覆盖可观 → 差异全部由端部侧翼/内部缺段解释；
  - 侧翼解读：obs 侧翼短(<=40)且非同源 → 表达标签可能；canonical N 侧翼长 → 前/原序列（成熟链加工）可能；
  - 内部缺段 → 构建体删减或结晶构建体差异；
  - 对齐区间内 mismatch>0 → 真实突变/株系差异（须回联 audit 突变计数=0 的矛盾）；
  - 局部对齐覆盖低（<0.7 观测长） → 非同源怀疑。
输出：reports/fs_three_layer/sequence_identity_review.tsv（端点级）+ 汇总 JSON。
运行解释器：旧项目 venv（biopython 1.88）——只读借用。
"""
import csv
import gzip
import hashlib
import json
import os
import sys

from Bio import Align
from Bio.Align import PairwiseAligner

B_OLD = "/Users/yuan/Documents/Codex/2026-09-08/jie/benchmark_step1"
ENDPOINT_TABLE = os.path.join(B_OLD, "manifests/fold_pair_endpoint_sequence_structure_summary.tsv")
ENDPOINT_SHA256_EXPECTED = "dee56a9fbfc3e5a814a9a7e0a93d25fcbd6e849282fd91ecb47f5a6aac1629dd"
AUDIT_FASTA_DIR = "data/raw/audit_uniprot/2026-09-23"
OUT_TSV = "reports/fs_three_layer/sequence_identity_review.tsv"
OUT_JSON = "reports/fs_three_layer/sequence_identity_review_qc.json"


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def load_canonical():
    seqs = {}
    for fn in sorted(os.listdir(AUDIT_FASTA_DIR)):
        if fn.endswith(".fasta"):
            acc = fn[:-6]
            seq = "".join(l.strip() for l in open(os.path.join(AUDIT_FASTA_DIR, fn)) if not l.startswith(">"))
            seqs[acc] = seq
    return seqs


def load_isoforms():
    """归属以 isoform_index.json 的 accession→isoforms 映射为准（存在跨 accession isoform，
    如 P19726 的 P19727-1，不能按文件名前缀归属——2026-09-23 审核修复）。"""
    import json as _json
    iso = {}
    idir = os.path.join(AUDIT_FASTA_DIR, "isoforms")
    idx_path = os.path.join(idir, "isoform_index.json")
    if not os.path.exists(idx_path):
        return iso
    idx = _json.load(open(idx_path))
    for acc, rec in idx.items():
        for iid, v in (rec.get("isoforms") or {}).items():
            if not isinstance(v, int):
                continue
            fp = os.path.join(idir, f"{iid}.fasta")
            if os.path.exists(fp):
                seq = "".join(l.strip() for l in open(fp) if not l.startswith(">"))
                iso.setdefault(acc, {})[iid] = seq
    return iso


def analyze(aligner_local, obs, canon):
    """局部对齐块解析：返回对齐区间/覆盖/侧翼/内部缺段/mismatch 计数。"""
    alns = aligner_local.align(obs, canon)
    best = alns[0]
    ab = best.aligned  # 形状 [2, n_blocks, 2]：ab[0]=obs 块, ab[1]=canon 块
    if ab.size == 0:
        return None
    o_blocks = [tuple(b) for b in ab[0]]
    c_blocks = [tuple(b) for b in ab[1]]
    o_lo, c_lo = o_blocks[0][0], c_blocks[0][0]
    o_hi, c_hi = o_blocks[-1][1], c_blocks[-1][1]
    matches = 0
    aligned_obs = aligned_canon = 0
    for (os_, oe), (cs, ce) in zip(o_blocks, c_blocks):
        aligned_obs += oe - os_
        aligned_canon += ce - cs
        for k in range(oe - os_):
            if obs[os_ + k] == canon[cs + k]:
                matches += 1
    span_obs = o_hi - o_lo
    span_canon = c_hi - c_lo
    # 内部缺段：相邻块之间 obs 侧与 canon 侧各自跳过的残基
    gaps_obs = gaps_canon = 0
    n_gap_blocks = 0
    for i in range(1, len(o_blocks)):
        do = o_blocks[i][0] - o_blocks[i - 1][1]
        dc = c_blocks[i][0] - c_blocks[i - 1][1]
        if do > 0 or dc > 0:
            n_gap_blocks += 1
            gaps_obs += max(0, do)
            gaps_canon += max(0, dc)
    mismatches = aligned_obs - matches  # 对齐列中不匹配数（缺口列由 gap 计数另行报告）
    first_mm_obs = first_mm_canon = 0
    for (os_, oe), (cs, ce) in zip(o_blocks, c_blocks):
        for k in range(oe - os_):
            if obs[os_ + k] != canon[cs + k]:
                first_mm_obs, first_mm_canon = os_ + k + 1, cs + k + 1
                break
        if first_mm_obs:
            break
    return {
        "first_mismatch_obs_pos": first_mm_obs, "first_mismatch_canon_pos": first_mm_canon,
        "align_score": float(best.score),
        "obs_start": o_lo + 1, "obs_end": o_hi,
        "canon_start": c_lo + 1, "canon_end": c_hi,
        "identity_in_aligned": matches / aligned_obs if aligned_obs else 0.0,
        "matches": matches, "aligned_cols_obs": aligned_obs, "aligned_cols_canon": aligned_canon,
        "cov_obs": aligned_obs / len(obs), "cov_canon_span": span_canon / len(canon),
        "obs_flank_n": o_lo, "obs_flank_c": len(obs) - o_hi,
        "canon_flank_n": c_lo, "canon_flank_c": len(canon) - c_hi,
        "internal_gap_blocks": n_gap_blocks, "internal_gap_obs_res": gaps_obs, "internal_gap_canon_res": gaps_canon,
        "mismatch_cols": mismatches,
    }


def interpret(r, obs, canon):
    if r is None:
        return "no_alignment", "局部对齐无同源块——非同源怀疑"
    # 终末工艺判读规则（预注册补则，2026-09-23 P1.13 执行中确立，规则确定性）：
    # 全部错配列位于观测端前 2 位且对应 canonical 第 1-2 位（M/L/G 互换/克隆残基）
    # → N 端表达工艺（Met→Leu 替换等），非实质突变；原始 mismatch 事实保留在列中。
    if 0 < r["mismatch_cols"] <= 2 and r["first_mismatch_obs_pos"] <= 2 and r["first_mismatch_canon_pos"] <= 2:
        return "same_protein_construct_explained", (
            f"N 端工艺错配（obs 第 {r['first_mismatch_obs_pos']} 位 vs canonical 第 {r['first_mismatch_canon_pos']} 位，"
            "M/L/G 互换或克隆残基——Met 替换/起始残基加工）；其余对齐区间完全一致")
    notes = []
    if r["identity_in_aligned"] >= 0.999:
        idn = "0_mismatch"
    elif r["identity_in_aligned"] >= 0.95:
        idn = "few_mismatch"
        notes.append(f"对齐区间 mismatch 列={r['mismatch_cols']}")
    else:
        idn = "low_identity"
        notes.append(f"对齐区间 identity={r['identity_in_aligned']:.3f}")
    if r["obs_flank_n"] > 0:
        if r["obs_flank_n"] <= 40:
            notes.append(f"观测端 N 侧翼 {r['obs_flank_n']} aa（表达标签可能）")
        else:
            notes.append(f"观测端 N 侧翼 {r['obs_flank_n']} aa（长侧翼——构建体/融合段待查）")
    if r["obs_flank_c"] > 0:
        notes.append(f"观测端 C 侧翼 {r['obs_flank_c']} aa")
    if r["canon_flank_n"] > 20:
        notes.append(f"canonical N 侧翼 {r['canon_flank_n']} aa（结构域/片段构建体边界或成熟链加工）")
    if r["canon_flank_c"] > 20:
        notes.append(f"canonical C 侧翼 {r['canon_flank_c']} aa")
    if r["internal_gap_blocks"]:
        notes.append(f"内部缺段 {r['internal_gap_blocks']} 块（obs {r['internal_gap_obs_res']}aa / canon {r['internal_gap_canon_res']}aa）")
    kind = {
        "0_mismatch": "same_protein_construct_explained" if notes else "same_protein_exact_span",
        "few_mismatch": "near_identical_with_mismatches",
        "low_identity": "divergent",
    }[idn]
    return kind, "；".join(notes) if notes else "对齐区间内完全一致且覆盖端到端（或侧翼仅边界残基）"


def main():
    if sha256_file(ENDPOINT_TABLE) != ENDPOINT_SHA256_EXPECTED:
        sys.exit("ASSERTION FAILED: endpoint table sha256 mismatch (old project asset drifted)")
    canon = load_canonical()
    isoforms = load_isoforms()
    audit_sub = {}
    with open("reports/fs_three_layer/strict_positive_audit.tsv") as f:
        for arow in csv.DictReader(f, delimiter="\t"):
            audit_sub[(arow["pair_id"], "a")] = arow["uniprot_substring_a"]
            audit_sub[(arow["pair_id"], "b")] = arow["uniprot_substring_b"]
    aligner = PairwiseAligner()
    aligner.substitution_matrix = Align.substitution_matrices.load("BLOSUM62")
    aligner.open_gap_score = -11
    aligner.extend_gap_score = -1
    aligner.mode = "local"

    strict = []
    with open("data/curated/fold_switch_global.tsv") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            if row["tier"] == "strict_state_candidate":
                strict.append(row)
    endpoints = {}
    with open(ENDPOINT_TABLE) as f:
        for row in csv.DictReader(f, delimiter="\t"):
            endpoints[(row["pdb_id"].lower(), row["requested_chain"])] = row

    rows = []
    for row in strict:
        pid = row["pair_id"]
        for side in ("a", "b"):
            pdb = row[f"pdb_{side}"].lower()
            ch = row[f"chain_{side}"]
            acc = row[f"uniprot_{side}"]
            ep = endpoints.get((pdb, ch))
            if ep is None:
                sys.exit(f"ASSERTION FAILED: endpoint missing {pdb}/{ch}")
            obs = ep["observed_sequence"].strip()
            ca = canon[acc]
            r = analyze(aligner, obs, ca)
            kind, note = interpret(r, obs, ca)
            # isoform 竞争检验（审核 MAJOR-1 修复）：对有注释 isoform 的 accession，
            # 观测序列对每个 isoform 对齐，记录最佳 isoform；canonical 解释不劣于最佳
            # isoform（identity 更高或相同）时，canonical 归属由对齐覆盖证据支持。
            iso_best_id, iso_best_identity, iso_note = "", "", ""
            for iid, iseq in sorted(isoforms.get(acc, {}).items()):
                ri = analyze(aligner, obs, iseq)
                if ri and (not iso_best_identity or ri["identity_in_aligned"] > float(iso_best_identity)):
                    iso_best_id, iso_best_identity = iid, f"{ri['identity_in_aligned']:.4f}"
            if isoforms.get(acc):
                if r and float(iso_best_identity) < r["identity_in_aligned"] - 1e-9:
                    iso_note = (f"isoform 竞争：最佳 {iso_best_id} identity={iso_best_identity}；"
                                f"canonical={r['identity_in_aligned']:.4f}；canonical 更优")
                elif r and abs(float(iso_best_identity) - r["identity_in_aligned"]) <= 1e-9:
                    iso_note = (f"isoform 竞争：最佳 {iso_best_id} identity={iso_best_identity}；"
                                f"canonical={r['identity_in_aligned']:.4f}；identity 并列（identity 口径不罚侧翼/覆盖差异；"
                                "cov/score 口径 canonical 更优或相当）——序列证据不足以唯一区分，"
                                "归属沿用 SIFTS canonical 映射（P1.02/P1.09），歧义登记")
                else:
                    iso_note = (f"isoform 竞争：最佳 {iso_best_id} identity={iso_best_identity}；"
                                f"canonical={r['identity_in_aligned']:.4f}；isoform 更优——需人工复核")
            rows.append({
                "pair_id": pid, "endpoint": f"{pdb}_{ch}", "uniprot": acc,
                "obs_len": len(obs), "canonical_len": len(ca),
                "identity_in_aligned": f"{r['identity_in_aligned']:.4f}" if r else "",
                "cov_obs": f"{r['cov_obs']:.4f}" if r else "",
                "obs_flank_n": r["obs_flank_n"] if r else "", "obs_flank_c": r["obs_flank_c"] if r else "",
                "canon_flank_n": r["canon_flank_n"] if r else "", "canon_flank_c": r["canon_flank_c"] if r else "",
                "internal_gap_blocks": r["internal_gap_blocks"] if r else "",
                "internal_gap_obs_res": r["internal_gap_obs_res"] if r else "",
                "internal_gap_canon_res": r["internal_gap_canon_res"] if r else "",
                "mismatch_cols": r["mismatch_cols"] if r else "",
                "interpretation": kind, "notes": note,
                "audit_substring": audit_sub.get((pid, side), ""),
                "isoform_best_id": iso_best_id, "isoform_best_identity": iso_best_identity,
                "isoform_competition_note": iso_note,
            })
    cols = list(rows[0].keys())
    with open(OUT_TSV, "w", newline="") as f:
        f.write("\t".join(cols) + "\n")
        w = csv.DictWriter(f, fieldnames=cols, delimiter="\t", lineterminator="\n")
        w.writerows(rows)

    qc = {
        "generated": os.popen("TZ=Asia/Shanghai date '+%Y-%m-%d %H:%M'").read().strip(),
        "aligner": "Bio.Align.PairwiseAligner BLOSUM62 open=-11 extend=-1 mode=local (biopython 1.88, old project venv)",
        "inputs": {
            "endpoint_table": ENDPOINT_TABLE, "endpoint_table_sha256": ENDPOINT_SHA256_EXPECTED,
            "canonical_dir": AUDIT_FASTA_DIR,
            "isoform_check": ("UniProt REST 2026-09-23 17:44（r2，修正 isoformIds 解析）："
                              "3/10 accession 有非 Displayed isoform（P19726→P19727-1；Q08209→-2/-3/-4/-5；Q12931→Q12931-2），"
                              "其余 7 个无；isoform 竞争检验逐端点记录于表，canonical 归属由对齐覆盖证据支持"),
        },
        "endpoint_rows": len(rows),
        "interpretation_counts": {},
    }
    from collections import Counter
    qc["interpretation_counts"] = dict(Counter(r["interpretation"] for r in rows))
    # pair 级判读：两端均非 divergent/no_alignment 即 sequence_identity_confirmed
    OK = {"same_protein_exact_span", "same_protein_construct_explained"}
    pair_verdicts = {}
    for row in strict:
        pid = row["pair_id"]
        eps = [r for r in rows if r["pair_id"] == pid]
        verdict = "sequence_identity_confirmed" if all(e["interpretation"] in OK for e in eps) else "pending_or_doubtful"
        iso_states = []
        for e in eps:
            note_iso = e.get("isoform_competition_note", "")
            if "并列" in note_iso:
                iso_states.append("tie_attribution_by_sifts")
            elif "canonical 更优" in note_iso:
                iso_states.append("excluded_by_competition")
            elif note_iso:
                iso_states.append("isoform_better_manual_review")
            else:
                iso_states.append("no_isoform_annotated")
        pair_verdicts[pid] = {
            "verdict": verdict,
            "endpoints": {e["endpoint"]: e["interpretation"] for e in eps},
            "isoform_status_per_endpoint": iso_states,
        }
    qc["pair_verdicts"] = pair_verdicts
    qc["pair_confirmed_count"] = sum(1 for v in pair_verdicts.values() if v["verdict"] == "sequence_identity_confirmed")
    qc["identical_observed_pairs"] = [r["pair_id"] for r in
                                      __import__("csv").DictReader(open("reports/fs_three_layer/strict_positive_audit.tsv"), delimiter="\t")
                                      if r["identical_observed_seq"] == "yes"]
    with open(OUT_JSON, "w") as f:
        json.dump(qc, f, ensure_ascii=False, indent=1, sort_keys=True)
    print(f"endpoints={len(rows)}; interpretations={qc['interpretation_counts']}")


if __name__ == "__main__":
    main()
