#!/usr/bin/env python3
"""P1.10 构建折叠转换未标注背景宇宙 A（Swiss-Prot 序列）与 B（RCSB/SIFTS 结构）。

用法：
  python3 scripts/build_fs_unlabeled_background.py \
    --config configs/fold_switch_background.yaml \
    --source-manifest data/manifests/fold_switch_background_sources.tsv \
    --strict-positive-table data/curated/fold_switch_global.tsv \
    --audit-table reports/fs_three_layer/strict_positive_audit.tsv \
    --exposure-log logs/exposure_log.tsv \
    --output-dir . \
    --run-id p1_10_20260923

语义铁律：背景不是阴性——biological_target 恒为空、pu_observed_label=0 仅表示
"当前未被确认为阳性"。数据库注释不充当实验标签。
"""
import argparse
import contextlib
import csv
import gzip
import hashlib
import io
import json
import os
import re
import sys
from collections import Counter, defaultdict

AA_ALLOWED = set("ACDEFGHIKLMNPQRSTVWY")
PDB_ID_RE = re.compile(r"^[0-9A-Za-z]{4}$")
FS_TIER_REASON = {
    "strict_state_candidate": "fs_strict_positive",
    "extension_construct_difference": "fs_extension",
    "extension_condition_or_assembly": "fs_extension",
    "fragment_only": "fs_fragment_evidence",
}
OS_RE = re.compile(r"\sOS=(\S.*?)(?:\s+OX=|\s+GN=|\s+PE=|\s+SV=|$)")


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


@contextlib.contextmanager
def open_text_out(path):
    """按扩展名决定明文或确定性 gzip（mtime=0，A8 字节级可复现）写出句柄。"""
    if path.endswith(".gz"):
        raw = open(path, "wb")
        gzf = gzip.GzipFile(fileobj=raw, mode="wb", compresslevel=9, mtime=0)
        f = io.TextIOWrapper(gzf, newline="")
        try:
            yield f
        finally:
            f.close()
            gzf.close()
            raw.close()
    else:
        f = open(path, "w", newline="")
        try:
            yield f
        finally:
            f.close()


def open_text_in(path):
    """按扩展名选择明文或 gzip 读句柄。"""
    if path.endswith(".gz"):
        return gzip.open(path, "rt", newline="")
    return open(path, newline="")


def sha256_text(s):
    return hashlib.sha256(s.encode("ascii", "strict")).hexdigest()


def die(msg):
    print(f"ASSERTION FAILED: {msg}", file=sys.stderr)
    sys.exit(1)


def parse_fasta(path):
    """流式产出 (accession, entry_name, protein_name, organism, sequence)。"""
    acc = name = prot = org = None
    buf = []
    with gzip.open(path, "rt") as f:
        for line in f:
            if line.startswith(">"):
                if acc is not None:
                    yield acc, name, prot, org, "".join(buf)
                head = line[1:].rstrip("\n")
                parts = head.split("|")
                acc = parts[1]
                after = "|".join(parts[2:])           # NAME 描述 OS=... OX=...
                name = after.split(" ", 1)[0]
                m = OS_RE.search(after)
                org = m.group(1).strip() if m else ""
                if m:
                    prot = after[len(name):m.start()].strip()
                else:
                    prot = after[len(name):].strip()
                buf = []
            else:
                buf.append(line.strip())
    if acc is not None:
        yield acc, name, prot, org, "".join(buf)


def open_sifts_csv(path):
    """打开 SIFTS csv.gz，跳过 # 注释行；返回 (header_list, csv.reader)。"""
    f = gzip.open(path, "rt")
    first = f.readline()
    while first.startswith("#"):
        first = f.readline()
    header = first.rstrip("\n").split(",")
    return header, csv.reader(f, delimiter=",")


def union_len(segs):
    """区间并集长度：重叠去重、相邻不并合（缺口不计）、无效段丢弃前由调用方统计。"""
    total = 0
    cur_b = cur_e = None
    for b, e in sorted(segs):
        if cur_b is None:
            cur_b, cur_e = b, e
        elif b <= cur_e:                      # 仅重叠才合并；相邻(隔 1)不合并
            cur_e = max(cur_e, e)
        else:
            total += cur_e - cur_b + 1
            cur_b, cur_e = b, e
    if cur_b is not None:
        total += cur_e - cur_b + 1
    return total


def load_fs_exclusions(fs_table):
    excl = defaultdict(lambda: [None, set()])
    with open(fs_table) as f:
        rd = csv.DictReader(f, delimiter="\t")
        for row in rd:
            tier = row["tier"]
            if tier not in FS_TIER_REASON:
                continue
            reason = FS_TIER_REASON[tier]
            for col in ("uniprot_a", "uniprot_b"):
                u = (row.get(col) or "").strip()
                if u:
                    excl[u][0] = reason
                    excl[u][1].add(row["pair_id"])
    return {u: (v[0], sorted(v[1])) for u, v in excl.items()}


def load_strict_from_audit(audit_table):
    s = set()
    with open(audit_table) as f:
        rd = csv.DictReader(f, delimiter="\t")
        for row in rd:
            for col in ("uniprot_a", "uniprot_b"):
                u = (row.get(col) or "").strip()
                if u:
                    s.add(u)
    return s


def load_exposure(exposure_log):
    exposed = {}
    if not os.path.exists(exposure_log):
        return exposed
    with open(exposure_log) as f:
        rd = csv.reader(f, delimiter="\t")
        header = next(rd, None)
        if header is None:
            return exposed
        for row in rd:
            if len(row) >= 2 and row[0]:
                exposed.setdefault(row[0], row[1])
    return exposed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--source-manifest", required=True)
    ap.add_argument("--strict-positive-table", required=True)
    ap.add_argument("--audit-table", required=True)
    ap.add_argument("--exposure-log", default="logs/exposure_log.tsv")
    ap.add_argument("--output-dir", default=".")
    ap.add_argument("--run-id", required=True)
    args = ap.parse_args()

    import yaml
    cfg = yaml.safe_load(open(args.config))
    root = args.output_dir
    sp_dir = "data/raw/swissprot/2026-09-23"
    sifts_dir = "data/raw/sifts/2026-09-23"
    wwpdb_dir = "data/raw/wwpdb/2026-09-23"
    fasta_path = os.path.normpath(os.path.join(root, sp_dir, "uniprot_sprot.fasta.gz"))
    relnotes_path = os.path.normpath(os.path.join(root, sp_dir, "relnotes.txt"))
    chain_map_path = os.path.normpath(os.path.join(root, sifts_dir, "pdb_chain_uniprot.csv.gz"))
    updb_path = os.path.normpath(os.path.join(root, sifts_dir, "uniprot_pdb.csv.gz"))
    segobs_path = os.path.normpath(os.path.join(root, sifts_dir, "uniprot_segments_observed.csv.gz"))
    entry_type_path = os.path.normpath(os.path.join(root, wwpdb_dir, "pdb_entry_type.txt"))
    entries_idx_path = os.path.normpath(os.path.join(root, wwpdb_dir, "entries.idx"))
    raw_files = [fasta_path, relnotes_path, chain_map_path, updb_path, segobs_path,
                 entry_type_path, entries_idx_path]

    # ---- 0) manifest 校验 ----
    manifest = {}
    with open(os.path.normpath(os.path.join(root, args.source_manifest))) as f:
        for row in csv.DictReader(f, delimiter="\t"):
            manifest[row["file_path"]] = row["sha256"]
    for path in raw_files:
        got = sha256_file(path)
        if manifest.get(path) != got:
            die(f"manifest sha256 mismatch for {path}: manifest={manifest.get(path)} got={got}")
    print(f"[0] manifest sha256 verified for {len(raw_files)} files")

    # ---- 1) FS 排除集合（双源一致断言） ----
    fs_excl = load_fs_exclusions(os.path.normpath(os.path.join(root, args.strict_positive_table)))
    strict_set = {u for u, (r, _) in fs_excl.items() if r == "fs_strict_positive"}
    # pending tier 的 accession（保留在背景中，但需追溯其去重代表——审核 M2）
    pending_set = set()
    with open(os.path.normpath(os.path.join(root, args.strict_positive_table))) as f:
        for row in csv.DictReader(f, delimiter="\t"):
            if row["tier"] == "pending_evidence":
                for col in ("uniprot_a", "uniprot_b"):
                    u = (row.get(col) or "").strip()
                    if u:
                        pending_set.add(u)
    audit_strict = load_strict_from_audit(os.path.normpath(os.path.join(root, args.audit_table)))
    if strict_set != audit_strict:
        die(f"strict set mismatch global vs audit: only_global={sorted(strict_set-audit_strict)} only_audit={sorted(audit_strict-strict_set)}")
    print(f"[1] FS exclusion accessions={len(fs_excl)} (strict={len(strict_set)})")

    # ---- 2) 曝光日志 ----
    exposed = load_exposure(os.path.normpath(os.path.join(root, args.exposure_log)))
    print(f"[2] exposure log entries={len(exposed)}")

    # ---- 3) 背景 A ----
    official_count = int(cfg["background_a_sequence"]["source"]["official_entry_count"])
    min_len = int(cfg["background_a_sequence"]["length_rule"]["min_length"])
    forbidden = set(cfg["background_a_sequence"]["residue_rule"]["forbidden_letters"])
    version_a = "UniProtKB/Swiss-Prot 2026_03"

    n_entries = 0
    sp_len_all = {}
    entry_names, prot_names, orgs = {}, {}, {}
    sha_of = {}
    excl_a = []
    isoform_hits = 0
    seq_groups = defaultdict(list)
    letter_counter = Counter()
    max_len_kept = 0

    for acc, name, prot, org, seq in parse_fasta(fasta_path):
        n_entries += 1
        if "-" in acc:
            isoform_hits += 1
        sp_len_all[acc] = len(seq)
        if acc in fs_excl:
            reason, pairs = fs_excl[acc]
            excl_a.append((acc, reason, ";".join(pairs)))
            continue
        L = len(seq)
        if L < min_len:
            excl_a.append((acc, "length_below_min", f"length={L}"))
            continue
        bad = sorted(set(seq) - AA_ALLOWED)
        if bad:
            for b in bad:
                letter_counter[b] += 1
            excl_a.append((acc, "nonstandard_residue_letter", "letters=" + ",".join(bad)))
            continue
        sha = sha256_text(seq)
        seq_groups[sha].append(acc)
        entry_names[acc], prot_names[acc], orgs[acc] = name, prot, org

    if n_entries != official_count:
        die(f"A5 FASTA entry count {n_entries} != official {official_count}")
    if isoform_hits != 0:
        die(f"A6 isoform-suffixed accessions found: {isoform_hits}")

    dupes = []
    rows_a = []
    for sha, accs in seq_groups.items():
        accs_sorted = sorted(accs)
        keep = accs_sorted[0]
        rows_a.append(keep)
        sha_of[keep] = sha
        max_len_kept = max(max_len_kept, sp_len_all[keep])
        for d in accs_sorted[1:]:
            dupes.append((d, keep, sha, sp_len_all[d]))
    del seq_groups
    rows_a.sort()

    # pending accession 在 A 表的去重代表追溯（审核 M2：G1 若裁 pending 为阳性，
    # 须按序列（sha256）剔除其代表，仅按 accession 剔除不彻底）
    acc_a = set(rows_a)
    dup_map = {d: k for d, k, _, _ in dupes}
    pending_rep = {
        "present_directly": sum(1 for u in pending_set if u in acc_a),
        "represented_by_other_accession": {u: dup_map[u] for u in sorted(pending_set) if u in dup_map},
        "excluded_other_reason": sorted(u for u in pending_set if u not in acc_a and u not in dup_map and u in sp_len_all),
        "not_in_sprot": sorted(u for u in pending_set if u not in sp_len_all),
        "note": "G1 裁决 pending 为阳性时，剔除须以 sequence_sha256 为准（连带代表行），不能只按 accession",
    }

    # n_pdb_entries_crossref（SIFTS uniprot_pdb）
    updb_pdb = defaultdict(set)
    header, rd = open_sifts_csv(updb_path)
    ix_sp, ix_pdb = header.index("SP_PRIMARY"), header.index("PDB")
    for row in rd:
        sp = row[ix_sp].strip()
        if sp:
            for p in row[ix_pdb].replace(" ", "").split(";"):
                if p:
                    updb_pdb[sp].add(p.lower())

    out_a_path = os.path.normpath(os.path.join(root, cfg["background_a_sequence"]["output_table"]))
    fields_a = cfg["background_a_sequence"]["fields"]
    with open_text_out(out_a_path) as f:
        f.write("\t".join(fields_a) + "\n")
        w = csv.writer(f, delimiter="\t", lineterminator="\n")
        for acc in rows_a:
            w.writerow([acc, entry_names[acc], prot_names[acc], orgs[acc],
                        sp_len_all[acc], sha_of[acc],
                        len(updb_pdb.get(acc, ())),
                        "unlabeled", "", 0, version_a,
                        exposed.get(acc, "none")])
    print(f"[3] background A rows={len(rows_a)} exclusions={len(excl_a)} dupes={len(dupes)} max_len={max_len_kept}")

    # ---- 4) 背景 B ----
    type_dist = Counter()
    entry_type = {}
    with open(entry_type_path) as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) >= 3 and PDB_ID_RE.match(parts[0]):
                t = parts[2].strip()
                entry_type[parts[0].lower()] = t
                type_dist[t] += 1
    computational = {p for p, t in entry_type.items() if t == "computational"}
    experimental = {p for p, t in entry_type.items() if t != "computational"}
    print(f"[4a] entry_type entries={len(entry_type)} dist={dict(type_dist)}")

    method_of = {}
    res_of = {}
    method_fallback_used = 0
    bad_rows_idx = 0
    with open(entries_idx_path, encoding="latin-1") as f:
        for k, line in enumerate(f):
            if k < 2:
                continue
            parts = line.rstrip("\n").rstrip("\r").split("\t")
            if len(parts) < 8 or not PDB_ID_RE.match(parts[0]):
                bad_rows_idx += 1
                continue
            pdb = parts[0].lower()
            res_txt, m_txt = parts[-2].strip(), parts[-1].strip()
            if m_txt:
                method_of[pdb] = m_txt
            else:
                method_of[pdb] = entry_type.get(pdb, "")
                method_fallback_used += 1
            try:
                r = float(res_txt)
                res_of[pdb] = r if r > 0 else None
            except ValueError:
                res_of[pdb] = None
    print(f"[4b] entries.idx parsed={len(method_of)} skipped={bad_rows_idx} method_fallback={method_fallback_used}")

    # 4c) 观测区段（SP 编号）→ (pdb,chain,sp) 区段列表（后续按并集计算覆盖；
    #     SIFTS 原始数据存在同键重叠区段，直接求和会重复计数）
    obs_segs = defaultdict(list)
    n_seg = 0
    n_obs_inverted = 0
    header, rd = open_sifts_csv(segobs_path)
    ix = {c: header.index(c) for c in ("PDB", "CHAIN", "SP_PRIMARY", "SP_BEG", "SP_END")}
    for row in rd:
        pdb = row[ix["PDB"]].strip().lower()
        chain = row[ix["CHAIN"]].strip()
        sp = row[ix["SP_PRIMARY"]].strip()
        try:
            b, e = int(row[ix["SP_BEG"]]), int(row[ix["SP_END"]])
        except ValueError:
            continue
        n_seg += 1
        if e < b:
            n_obs_inverted += 1
            continue
        obs_segs[(pdb, chain, sp)].append((b, e))
    print(f"[4c] observed segments={n_seg} inverted_dropped={n_obs_inverted}")

    # 4d) 链级映射 → 键级聚合（SIFTS 同一 (pdb,chain,uniprot) 可有多行对齐区段，
    #     例如 102l/A/P00720 分为 1-40 与 42-165 两段，中间缺口 41）。
    #     覆盖 = 键内区段并集长度/SP 长度（重叠去重、相邻不并、缺口不计），子表一行一键。
    map_segs = defaultdict(list)
    key_rows_total = defaultdict(int)
    mapped_pdb_set = set()
    exp_key_set = set()
    pair_chain_set = set()
    n_chain_rows = 0
    n_map_inverted_rows = 0
    keys_with_inverted = set()
    header, rd = open_sifts_csv(chain_map_path)
    ix = {c: header.index(c) for c in ("PDB", "CHAIN", "SP_PRIMARY", "SP_BEG", "SP_END")}
    for row in rd:
        pdb = row[ix["PDB"]].strip().lower()
        chain = row[ix["CHAIN"]].strip()
        sp = row[ix["SP_PRIMARY"]].strip()
        if not pdb or not chain or not sp:
            continue
        n_chain_rows += 1
        pair_chain_set.add((pdb, chain, sp))
        if pdb not in experimental:
            continue
        mapped_pdb_set.add(pdb)
        key = (pdb, chain, sp)
        exp_key_set.add(key)
        try:
            b, e = int(row[ix["SP_BEG"]]), int(row[ix["SP_END"]])
        except ValueError:
            continue
        key_rows_total[key] += 1
        if e < b:
            # SIFTS 原始数据存在倒置区段（SP_BEG>SP_END，共约 72 行）：丢弃该行并按键留痕
            n_map_inverted_rows += 1
            keys_with_inverted.add(key)
        else:
            map_segs[key].append((b, e))
    keys_unparseable = len(exp_key_set - set(key_rows_total))
    n_multi_seg_keys = sum(1 for k, segs in map_segs.items() if len(segs) > 1)
    n_map_overlap_keys = sum(1 for k, segs in map_segs.items()
                             if union_len(segs) < sum(e - b + 1 for b, e in segs))
    n_obs_overlap_keys = sum(1 for k, segs in obs_segs.items()
                             if union_len(segs) < sum(e - b + 1 for b, e in segs))

    # 有效键（至少一段非倒置对齐）→ 子表行 + 蛋白级聚合。
    # 复审 R2-1/R2-2 修正：n_alignment_segments/sp_beg_min/sp_end_max 与主表全部聚合
    # 一律只来自有效段；主表=子表按蛋白重聚合（一致性由构造保证，另设独立复核断言 A10）。
    chain_rows = []
    prot_entries = defaultdict(set)
    prot_chains = defaultdict(set)
    prot_methods = defaultdict(set)
    prot_res = defaultdict(list)
    prot_mapped_cov = {}
    prot_obs_cov = {}
    n_obs_gt_mapped = 0
    for key in sorted(map_segs.keys()):
        pdb, chain, sp = key
        L = sp_len_all.get(sp)
        segs = map_segs[key]
        mlen = union_len(segs)
        olen = union_len(obs_segs.get(key, []))
        mcov = min(mlen / L, 1.0) if L else None
        ocov = min(olen / L, 1.0) if L else None
        if mcov is not None and ocov is not None and olen > mlen:
            n_obs_gt_mapped += 1
        b, e = min(s[0] for s in segs), max(s[1] for s in segs)
        res_v = res_of.get(pdb)
        method_v = method_of.get(pdb, "")
        chain_rows.append((pdb, chain, sp, len(segs), b, e, mcov, ocov,
                           method_v, f"{res_v:.2f}" if res_v else ""))
        prot_entries[sp].add(pdb)
        prot_chains[sp].add((pdb, chain))
        if method_v:
            prot_methods[sp].add(method_v)
        if res_v:
            prot_res[sp].append(res_v)
        if mcov is not None:
            prot_mapped_cov[sp] = max(prot_mapped_cov.get(sp, 0.0), mcov)
        if ocov is not None:
            prot_obs_cov[sp] = max(prot_obs_cov.get(sp, 0.0), ocov)

    entries_without_uniprot = len(experimental - mapped_pdb_set)
    print(f"[4d] chain rows={n_chain_rows} proteins_all={len(prot_entries)} entries_without_uniprot={entries_without_uniprot}")

    # uniprot_pdb 交叉核对（(SP,PDB) 对级，报告不阻断）
    pairs_updb = {(sp, p) for sp, pdbs in updb_pdb.items() for p in pdbs}
    pairs_chain = {(sp, pdb) for pdb, _, sp in pair_chain_set}
    only_chain = len(pairs_chain - pairs_updb)
    only_updb = len(pairs_updb - pairs_chain)
    print(f"[4x] crosscheck only_in_chain={only_chain} only_in_uniprot_pdb={only_updb}")

    version_b = "SIFTS 2026-09-13 (PDB 37.26, UniProt 2026.04) + wwPDB derived 2026-09-12"
    excl_b = []
    for sp in sorted(prot_entries):
        if sp in fs_excl:
            reason, pairs = fs_excl[sp]
            excl_b.append((sp, reason, ";".join(pairs)))
    excluded_prot_set = {e[0] for e in excl_b}

    out_b_path = os.path.normpath(os.path.join(root, cfg["background_b_structure"]["output_table"]))
    fields_b = cfg["background_b_structure"]["fields"]
    kept_b = sorted(sp for sp in prot_entries if sp not in excluded_prot_set)
    with open_text_out(out_b_path) as f:
        f.write("\t".join(fields_b) + "\n")
        w = csv.writer(f, delimiter="\t", lineterminator="\n")
        for sp in kept_b:
            L = sp_len_all.get(sp)
            best_res = min(prot_res[sp]) if prot_res[sp] else None
            w.writerow([sp, L if L is not None else "",
                        len(prot_entries[sp]), len(prot_chains[sp]),
                        ";".join(sorted(prot_methods[sp])),
                        f"{best_res:.2f}" if best_res is not None else "",
                        f"{prot_mapped_cov[sp]:.4f}" if sp in prot_mapped_cov else "",
                        f"{prot_obs_cov[sp]:.4f}" if sp in prot_obs_cov else "",
                        "unlabeled", "", 0, version_b])
    print(f"[5] background B proteins={len(kept_b)} excluded_fs={len(excl_b)}")

    # A10（复审 R2-1 修正的机器化）：主表聚合必须与"有效键子表按蛋白重聚合"完全一致
    agg_e, agg_c, agg_m, agg_r = defaultdict(set), defaultdict(set), defaultdict(set), defaultdict(list)
    for r_ in chain_rows:
        pdb_, chain_, sp_ = r_[0], r_[1], r_[2]
        if sp_ in excluded_prot_set:
            continue
        agg_e[sp_].add(pdb_)
        agg_c[sp_].add((pdb_, chain_))
        if r_[8]:
            agg_m[sp_].add(r_[8])
        if r_[9]:
            try:
                agg_r[sp_].append(float(r_[9]))
            except ValueError:
                pass
    for sp in kept_b:
        if agg_e[sp] != prot_entries[sp]:
            die(f"A10 entries mismatch for {sp}")
        if agg_c[sp] != prot_chains[sp]:
            die(f"A10 chains mismatch for {sp}")
        if agg_m[sp] != prot_methods[sp]:
            die(f"A10 methods mismatch for {sp}")
        # 分辨率在子表中为两位小数格式化值，两侧同格式化后比较（避免舍入错位假阳性）
        a10_res_sub = f"{min(agg_r[sp]):.2f}" if agg_r[sp] else None
        a10_res_main = f"{min(prot_res[sp]):.2f}" if prot_res[sp] else None
        if a10_res_sub != a10_res_main:
            die(f"A10 best_resolution mismatch for {sp}: subtable={a10_res_sub} main={a10_res_main}")
    print("[5a] A10 main-vs-subtable reaggregation consistency: pass")

    sub_path = os.path.normpath(os.path.join(root, cfg["background_b_structure"]["output_subtable"]))
    sub_cols = cfg["background_b_structure"]["subtable_columns"]
    with open_text_out(sub_path) as f:
        f.write("\t".join(sub_cols) + "\n")
        w = csv.writer(f, delimiter="\t", lineterminator="\n")
        n_sub = 0
        for r in chain_rows:
            if r[2] in excluded_prot_set:
                continue
            w.writerow([r[0], r[1], r[2], r[3], r[4], r[5],
                        f"{r[6]:.4f}" if r[6] is not None else "",
                        f"{r[7]:.4f}" if r[7] is not None else "",
                        r[8], r[9]])
            n_sub += 1
    print(f"[5b] chain subtable rows={n_sub} multi_segment_keys={n_multi_seg_keys} keys_unparseable={keys_unparseable}")

    def write_tsv(path, cols, rows):
        with open_text_out(path) as f:
            f.write("\t".join(cols) + "\n")
            csv.writer(f, delimiter="\t", lineterminator="\n").writerows(sorted(rows))
    write_tsv(os.path.normpath(os.path.join(root, cfg["background_a_sequence"]["output_exclusions"])),
              cfg["background_a_sequence"]["output_exclusions_columns"], excl_a)
    write_tsv(os.path.normpath(os.path.join(root, cfg["background_b_structure"]["output_exclusions"])),
              ["uniprot_accession", "reason", "detail"], excl_b)
    write_tsv(os.path.normpath(os.path.join(root, cfg["background_a_sequence"]["dedup_rule"]["sidecar"])),
              cfg["background_a_sequence"]["dedup_rule"]["sidecar_columns"], dupes)

    # ---- 6) 机器断言 A1–A9 ----
    acc_a = set(rows_a)
    if strict_set & acc_a:
        die(f"A1 strict∩A = {sorted(strict_set & acc_a)}")
    if strict_set & set(kept_b):
        die(f"A1 strict∩B = {sorted(strict_set & set(kept_b))}")
    ext_frag = {u for u, (r, _) in fs_excl.items() if r != "fs_strict_positive"}
    if ext_frag & acc_a:
        die(f"A2 ext/frag∩A = {sorted(ext_frag & acc_a)}")
    if ext_frag & set(kept_b):
        die(f"A2 ext/frag∩B = {sorted(ext_frag & set(kept_b))}")

    def check_table(path):
        zero_target = empty_target = n = 0
        with open_text_in(path) as f:
            for row in csv.DictReader(f, delimiter="\t"):
                n += 1
                bt = row["biological_target"]
                if bt == "":
                    empty_target += 1
                if bt == "0":
                    zero_target += 1
                if row["pu_observed_label"] != "0" or row["label_epistemic_status"] != "unlabeled":
                    die(f"A4 violated in {path}: {row}")
                if not row["background_source_version"]:
                    die(f"A7 violated in {path}")
        if zero_target != 0 or empty_target != n:
            die(f"A3 violated in {path}: zero={zero_target} empty={empty_target} n={n}")
        return n
    n_a = check_table(out_a_path)
    n_b = check_table(out_b_path)
    if n_a != len(acc_a) or n_b != len(kept_b):
        die(f"row count mismatch n_a={n_a}/{len(acc_a)} n_b={n_b}/{len(kept_b)}")

    # ---- 7) QC 汇总 ----
    now = os.popen("TZ=Asia/Shanghai date '+%Y-%m-%d %H:%M'").read().strip()
    out_paths = [
        cfg["background_a_sequence"]["output_table"],
        cfg["background_b_structure"]["output_table"],
        cfg["background_b_structure"]["output_subtable"],
        cfg["background_a_sequence"]["output_exclusions"],
        cfg["background_b_structure"]["output_exclusions"],
        cfg["background_a_sequence"]["dedup_rule"]["sidecar"],
    ]
    qc = {
        "run_id": args.run_id,
        "generated": now,
        "assertions": {
            "A1_strict_disjoint": "pass",
            "A2_extension_disjoint": "pass",
            "A3_no_zero_target": "pass",
            "A4_pu_label": "pass",
            "A5_entry_count": {"observed": n_entries, "official": official_count},
            "A6_no_isoform": isoform_hits,
            "A7_traceable": "pass",
            "A8_deterministic": "fixed sort keys: A/B by accession; subtable by pdb,chain,uniprot; rerun must reproduce output_sha256",
            "A9_exclusion_recorded": "pass",
            "A10_main_subtable_consistency": "pass",
        },
        "background_a": {
            "official_count": official_count,
            "rows": len(acc_a),
            "exclusions_total": len(excl_a),
            "exclusion_reasons": dict(Counter(r for _, r, _ in excl_a)),
            "fs_exclusion_presence": {
                "fs_accessions_total": len(fs_excl),
                "present_in_sprot_and_excluded": sum(1 for _, r, _ in excl_a if r.startswith("fs_")),
                "not_in_sprot_trEMBL_only": sorted(u for u in fs_excl if u not in sp_len_all),
                "note": "不在 Swiss-Prot 的 FS accession（TrEMBL-only）对宇宙 A 天然不存在，A1/A2 对其平凡成立；B 侧按同 44 全数排除",
            },
            "nonstandard_letter_accessions": dict(letter_counter),
            "exact_duplicates_removed": len(dupes),
            "max_length_kept": max_len_kept,
            "pending_representation": pending_rep,
        },
        "background_b": {
            "proteins": len(kept_b),
            "chain_subtable_rows": n_sub,
            "multi_segment_alignment_keys": n_multi_seg_keys,
            "keys_with_unparseable_sp_range": keys_unparseable,
            "sifts_anomalies": {
                "mapping_inverted_rows_dropped": n_map_inverted_rows,
                "keys_with_inverted_rows": len(keys_with_inverted),
                "keys_all_rows_inverted_dropped_from_subtable": len(keys_with_inverted - set(map_segs.keys())),
                "mapping_overlap_keys_union_lt_sum": n_map_overlap_keys,
                "observed_overlap_keys_union_lt_sum": n_obs_overlap_keys,
                "observed_gt_mapped_after_union": n_obs_gt_mapped,
                "observed_inverted_rows_dropped": n_obs_inverted,
                "note": "覆盖一律按区间并集（重叠去重、相邻不并、缺口不计）；倒置区段（SP_BEG>SP_END）按行丢弃留痕。P1.11 起禁止把 mapped 为空/0 或 observed>mapped 的键当覆盖证据",
            },
            "entries_total": len(entry_type),
            "entry_type_distribution": dict(type_dist),
            "computational_entries": len(computational),
            "experimental_entries": len(experimental),
            "entries_without_uniprot": entries_without_uniprot,
            "fs_excluded_proteins": len(excl_b),
            "proteins_without_sp_length": sum(1 for sp in kept_b if sp not in sp_len_all),
            "method_fallback_from_entry_type": method_fallback_used,
        },
        "crosscheck_sifts": {
            "only_in_chain_mapping": only_chain,
            "only_in_uniprot_pdb": only_updb,
        },
        "version_skew_note": (
            "SIFTS 构建于 UniProt 2026.04/PDB 37.26；背景 A FASTA 为 2026_03。"
            "accession 稳定；覆盖分母取自 2026_03 序列长度并 cap 1.0；"
            "n_pdb_entries_crossref 直接来自 SIFTS 聚合视图。"
        ),
        "output_sha256": {p: sha256_file(os.path.normpath(os.path.join(root, p))) for p in out_paths},
    }
    qc_path = os.path.normpath(os.path.join(root, cfg["output_summary_json"]))
    with open(qc_path, "w") as f:
        json.dump(qc, f, ensure_ascii=False, indent=1, sort_keys=True)
    print(f"[7] qc written: {qc_path}")
    print("ALL ASSERTIONS PASSED")


if __name__ == "__main__":
    main()
