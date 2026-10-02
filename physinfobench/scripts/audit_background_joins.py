#!/usr/bin/env python3
"""P1.21 背景版本跨源连接核查（G1 材料修订第 3 项）。

对 10 个 strict 阳性 + 25 个 L3 候选（pna_evidence_review 全集，含主选/备选/标记），
核查接受背景版本差异（FASTA 2026_03 vs SIFTS 2026-09-13[UniProt 2026.04] vs SIFTS
家族/PubMed 2026-09-20[PDB 38.26|UniProt 2026.03] vs wwPDB 2026-09-12）所需的六类连接：

  J1 长度：fasta_2026_03(=A/B 覆盖分母与 PN sp_length 源) vs REST 当前(审计 fasta) vs
     SIFTS 09-13 隐含长度（SP_BEG/SP_END 超出 fasta 长度=版本间序列伸长信号）；
  J2 家族：候选/阳性 PDB（uniprot_pdb 09-13）在 09-20 家族文件(pfam/cath/scop2)的存在性；
     反向=09-20 家族知晓但 09-13 映射没有的新条目；
  J3 PubMed：同键在 pdb_pubmed(09-20) 的存在性（n_independent_studies 分子源）；
  J4 wwPDB：同键在 entries.idx/pdb_entry_type(09-12) 的存在性（experimental/分辨率源）；
  J5 PDB 集偏差：上述四个来源的全局 PDB 集两两差（规模级）；
  J6 覆盖复算：候选在 B 表的 mapped/observed coverage max 用同口径（fasta 分母+区间并集）
     复算比对；阳性端点链另记覆盖率（不入 B）。

只读审计；不修改任何背景产物。"""
import csv
import datetime
import gzip
import os
import re
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RAW = os.path.join(ROOT, "data/raw")
SIFTS = os.path.join(RAW, "sifts/2026-09-23")
WW = os.path.join(RAW, "wwpdb/2026-09-23")
OUT = os.path.join(ROOT, "reports/fs_three_layer/background_version_join_audit.tsv")
QC = os.path.join(ROOT, "reports/fs_three_layer/background_version_join_audit_qc.json")
RUN_TS = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def die(m):
    print(f"[P1.21 FATAL] {m}", file=sys.stderr)
    sys.exit(1)


def rd(p):
    with open(p, newline="") as f:
        return list(csv.DictReader(f, delimiter="\t"))


# ---- 键集 ----
g_rows = rd(os.path.join(ROOT, "data/curated/fold_switch_global.tsv"))
positives = {}
for r in g_rows:
    if r["tier"] == "strict_state_candidate":
        positives[r["uniprot_a"]] = r
if len(positives) != 10:
    die(f"strict uniprot_a 数={len(positives)} != 10")
pna = rd(os.path.join(ROOT, "reports/fs_three_layer/pna_evidence_review.tsv"))
cands = sorted({r["uniprot_accession"] for r in pna})
if len(cands) != 25:
    die(f"pna 候选数={len(cands)} != 25")
wanted = set(positives) | set(cands)

# ---- J1a fasta 2026_03 长度 ----
fasta_len = {}
fasta_gz = os.path.join(RAW, "swissprot/2026-09-23/uniprot_sprot.fasta.gz")
acc = None
n = 0
with gzip.open(fasta_gz, "rt") as f:
    for line in f:
        if line.startswith(">"):
            if acc in wanted:
                fasta_len[acc] = n
            m = re.match(r">..\|([^|]+)\|", line)
            acc = m.group(1) if m else None
            n = 0
        elif acc:
            n += len(line.strip())
    if acc in wanted:
        fasta_len[acc] = n
missing_fasta = wanted - set(fasta_len)

# ---- J1b REST 当前（审计 fasta，阳性 10 个） ----
rest_len = {}
for u in positives:
    fp = os.path.join(RAW, f"audit_uniprot/2026-09-23/{u}.fasta")
    if not os.path.exists(fp):
        die(f"缺审计 fasta: {fp}")
    seqs = []
    name = None
    for line in open(fp):
        if line.startswith(">"):
            if name is not None:
                seqs.append((name, len(s)))
            name, s = line.strip(), ""
        else:
            s += line.strip()
    if name is not None:
        seqs.append((name, len(s)))
    can = [(k, v) for k, v in seqs if "|sp|" in k or u in k]
    if len(can) != 1:
        die(f"{u} 审计 fasta canonical 唯一性失败: {seqs}")
    rest_len[u] = can[0][1]

# ---- SIFTS 09-13：映射段（按需键）+ uniprot_pdb ----
map_segs = {}   # (pdb,chain,sp) -> [(b,e)]
obs_segs = {}
sp_end_max = {}
with gzip.open(os.path.join(SIFTS, "pdb_chain_uniprot.csv.gz"), "rt") as f:
    for row in csv.DictReader(l for l in f if not l.startswith("#")):
        sp = row["SP_PRIMARY"]
        if sp in wanted:
            key = (row["PDB"].lower(), row["CHAIN"], sp)
            try:
                b, e = int(row["SP_BEG"]), int(row["SP_END"])
            except ValueError:
                continue
            if b <= e:
                map_segs.setdefault(key, []).append((b, e))
                sp_end_max[sp] = max(sp_end_max.get(sp, 0), e, b)
pdbs_0913 = {}
with gzip.open(os.path.join(SIFTS, "uniprot_pdb.csv.gz"), "rt") as f:
    for row in csv.DictReader(l for l in f if not l.startswith("#")):
        sp = row["SP_PRIMARY"]
        if sp in wanted and row["PDB"]:
            pdbs_0913.setdefault(sp, set()).update(x.lower() for x in row["PDB"].split(";"))
with gzip.open(os.path.join(SIFTS, "uniprot_segments_observed.csv.gz"), "rt") as f:
    for row in csv.DictReader(l for l in f if not l.startswith("#")):
        sp = row["SP_PRIMARY"]
        if sp in wanted:
            key = (row["PDB"].lower(), row["CHAIN"], sp)
            try:
                b, e = int(row["SP_BEG"]), int(row["SP_END"])
            except ValueError:
                continue
            if b <= e:
                obs_segs.setdefault(key, []).append((b, e))


def union_len(segs):
    iv = sorted(segs)
    tot = 0
    cb, ce = None, None
    for b, e in iv:
        if cb is None:
            cb, ce = b, e
        elif b <= ce + 1:
            ce = max(ce, e)
        else:
            tot += ce - cb + 1
            cb, ce = b, e
    if cb is not None:
        tot += ce - cb + 1
    return tot


# ---- J2/J3：家族三系统 + PubMed（全局 PDB 集 + 按需键 acc→pdb） ----
fam_sets = {}
fam_acc_pdbs = {}
for name, fn in [("pfam", "pdb_chain_pfam.csv.gz"), ("cath", "pdb_chain_cath_uniprot.csv.gz"),
                 ("scop2", "pdb_chain_scop2_uniprot.csv.gz")]:
    s, ap = set(), {}
    with gzip.open(os.path.join(SIFTS, fn), "rt") as f:
        for row in csv.DictReader(l for l in f if not l.startswith("#")):
            pdb = row["PDB"].lower()
            s.add(pdb)
            sp = row.get("SP_PRIMARY")
            if sp in wanted:
                ap.setdefault(sp, set()).add(pdb)
    fam_sets[name] = s
    fam_acc_pdbs[name] = ap
fam_all = fam_sets["pfam"] | fam_sets["cath"] | fam_sets["scop2"]
pubmed_set = set()
with gzip.open(os.path.join(SIFTS, "pdb_pubmed.csv.gz"), "rt") as f:
    for row in csv.DictReader(l for l in f if not l.startswith("#")):
        pubmed_set.add(row["PDB"].lower())

# ---- J4：wwPDB 09-12 集 ----
idx_set = set()
for line in open(os.path.join(WW, "entries.idx")):
    t = line.split("\t", 1)[0].strip().lower()
    if re.fullmatch(r"[0-9a-z]{4}", t):
        idx_set.add(t)
etype_set, etype_comp = set(), set()
for line in open(os.path.join(WW, "pdb_entry_type.txt")):
    parts = line.split()
    if parts and re.fullmatch(r"[0-9a-z]{4}", parts[0].lower()):
        etype_set.add(parts[0].lower())
        if len(parts) >= 3 and parts[2] == "computational":
            etype_comp.add(parts[0].lower())

# ---- J6：候选在 B 表的存储覆盖（对比用） ----
btab = {}
with gzip.open(os.path.join(ROOT, "data/curated/fold_switch_unlabeled_structure.tsv.gz"), "rt") as f:
    for row in csv.DictReader(f, delimiter="\t"):
        if row["uniprot_accession"] in wanted:
            btab[row["uniprot_accession"]] = row

# ---- 逐键输出 ----
rows = []
for acc in sorted(wanted):
    fl = fasta_len.get(acc)
    L = fl
    mc, oc = None, None
    if L:
        mcovs = [min(union_len(v) / L, 1.0) for k, v in map_segs.items() if k[2] == acc]
        ocovs = [min(union_len(v) / L, 1.0) for k, v in obs_segs.items() if k[2] == acc]
        mc = max(mcovs) if mcovs else None
        oc = max(ocovs) if ocovs else None
    p13 = pdbs_0913.get(acc, set())
    fam_miss = sorted(p13 - fam_all)
    pub_miss = sorted(p13 - pubmed_set)
    idx_miss = sorted(p13 - idx_set)
    et_miss = sorted(p13 - etype_set)
    newer_in_fam = sorted((fam_acc_pdbs["pfam"].get(acc, set())
                           | fam_acc_pdbs["cath"].get(acc, set())
                           | fam_acc_pdbs["scop2"].get(acc, set())) - p13)
    skew = sp_end_max.get(acc)
    len_mismatch = ""
    if fl is None:
        len_status = "MISSING_IN_FASTA_2026_03(TrEMBL-only)"
    else:
        if acc in rest_len and rest_len[acc] != fl:
            len_mismatch = f"fasta={fl} vs REST={rest_len[acc]}"
        if skew and skew > fl:
            len_mismatch += f";SIFTS_SP_MAX={skew}>fasta({fl})"
        len_status = "consistent" if not len_mismatch else f"MISMATCH:{len_mismatch}"
    b = btab.get(acc)
    cov_check = ""
    if b:
        sm, so = b["mapped_coverage_max"], b["observed_coverage_max"]
        rm = f"{mc:.4f}" if mc is not None else ""
        ro = f"{oc:.4f}" if oc is not None else ""
        if sm != rm or so != ro:
            cov_check = f"stored={sm}/{so} vs recomputed={rm}/{ro}"
        else:
            cov_check = "match"
    rows.append({
        "accession": acc, "role": "positive" if acc in positives else "pn_candidate",
        "fasta_2026_03_len": fl if fl else "MISSING",
        "rest_current_len": rest_len.get(acc, ""),
        "sifts_sp_max_pos": skew if skew else "",
        "length_join_status": len_status,
        "n_pdb_0913": len(p13),
        "family_missing_pdbs": ";".join(fam_miss) or "-",
        "pubmed_missing_pdbs": ";".join(pub_miss) or "-",
        "wwpdb_idx_missing": ";".join(idx_miss) or "-",
        "entrytype_missing": ";".join(et_miss) or "-",
        "newer_pdbs_in_0920_family_only": ";".join(newer_in_fam) or "-",
        "coverage_recheck_vs_B": cov_check or ("n/a(positive,不在B)" if acc in positives else "n/a"),
        "recomputed_mapped_cov_max": f"{mc:.4f}" if mc is not None else "",
        "recomputed_observed_cov_max": f"{oc:.4f}" if oc is not None else "",
    })

# ---- 全局集偏差（J5） ----
set_stats = {
    "pfam_0920": len(fam_sets["pfam"]), "cath_0920": len(fam_sets["cath"]),
    "scop2_0920": len(fam_sets["scop2"]), "pubmed_0920": len(pubmed_set),
    "wwpdb_idx_0912": len(idx_set), "entrytype_0912": len(etype_set),
    "fam0920_minus_idx0912": len(fam_all - idx_set),
    "pubmed0920_minus_idx0912": len(pubmed_set - idx_set),
    "computational_in_entrytype": len(etype_comp),
}

# ---- 断言与汇总 ----
missing_fasta = [r["accession"] for r in rows if r["fasta_2026_03_len"] == "MISSING"]
cand_not_in_b = [a for a in cands if a not in btab]
if cand_not_in_b:
    die(f"PN 候选不在背景 B 表（候选应来自 B）: {cand_not_in_b}")
pos_len_mis = [r["accession"] for r in rows
               if r["role"] == "positive" and r["length_join_status"] != "consistent"]
cand_len_mis = [r["accession"] for r in rows
                if r["role"] == "pn_candidate" and r["length_join_status"] != "consistent"]
cov_mis = [r["accession"] for r in rows if r["coverage_recheck_vs_B"] not in ("match", "n/a", "n/a(positive,不在B)")]
qc = {
    "run_ts": RUN_TS, "wanted": len(wanted), "positives": 10, "pn_candidates": len(cands),
    "missing_in_fasta_2026_03": missing_fasta,
    "positive_length_mismatch": pos_len_mis, "candidate_length_mismatch": cand_len_mis,
    "candidate_sifts_sp_exceeds_fasta": [r["accession"] for r in rows
                                         if "SIFTS_SP_MAX" in r["length_join_status"]],
    "coverage_recheck_mismatch": cov_mis,
    "family_missing_pdbs_total": sum(len(r["family_missing_pdbs"].split(";"))
                                     for r in rows if r["family_missing_pdbs"] != "-"),
    "pubmed_missing_pdbs_total": sum(len(r["pubmed_missing_pdbs"].split(";"))
                                     for r in rows if r["pubmed_missing_pdbs"] != "-"),
    "wwpdb_missing_pdbs_total": sum(len(r["wwpdb_idx_missing"].split(";"))
                                    for r in rows if r["wwpdb_idx_missing"] != "-"),
    "newer_in_0920_family_only_total": sum(len(r["newer_pdbs_in_0920_family_only"].split(";"))
                                           for r in rows if r["newer_pdbs_in_0920_family_only"] != "-"),
    "set_stats": set_stats,
}
cols = list(rows[0])
with open(OUT, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=cols, delimiter="\t", lineterminator="\n")
    w.writeheader()
    w.writerows(rows)
with open(QC, "w") as f:
    import json
    json.dump(qc, f, ensure_ascii=False, indent=1, sort_keys=True)
print(f"[P1.21] 35 键核查完成 len_mis(pos/cand)={len(pos_len_mis)}/{len(cand_len_mis)} "
      f"cov_mis={len(cov_mis)} fam_miss={qc['family_missing_pdbs_total']} "
      f"pub_miss={qc['pubmed_missing_pdbs_total']} wwpdb_miss={qc['wwpdb_missing_pdbs_total']} "
      f"newer0920={qc['newer_in_0920_family_only_total']}")
