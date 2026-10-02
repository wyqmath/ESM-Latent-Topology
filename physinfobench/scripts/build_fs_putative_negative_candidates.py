#!/usr/bin/env python3
"""P1.11 构建折叠转换推定阴性（PN）候选池。

两阶段：
  阶段一（离线）：从 P1.10 背景 B + SIFTS 家族文件生成候选（家族/长度/结构数排序，每 strict positive top-30）。
  阶段二（检索）：Europe PMC 系统检索（ACC 自由文本 OR 蛋白名）AND 术语集；Crossref 仅作失败替补；
                  响应存档 data/raw/europepmc/2026-09-23/，--search-mode=cached 时从存档重放（可复现）。

语义铁律：PN 等级=证据覆盖程度而非阴性证明；候选不写 biological_target；
"无检索命中"与"完成全文审查"分列；命中 fold_switching/domain_swapping 者退出并进复核队列。
"""
import argparse
import csv
import gzip
import hashlib
import io
import json
import math
import os
import re
import sys
import time
import urllib.parse
import urllib.request
from collections import defaultdict

EXIT_CATEGORIES = {"fold_switching", "domain_swapping"}
TERM_SET = '("fold switching" OR "fold-switching" OR metamorphic OR "alternative fold" OR "conformational switch" OR "domain swapping")'
CLASS_RULES = [
    ("fold_switching", [r"fold[- ]switch", r"metamorphic protein", r"alternative fold", r"dual fold", r"fold plasticity"]),
    ("domain_swapping", [r"domain[- ]swapp", r"swapped dimer", r"swapped oligomer", r"3d domain exchange"]),
    ("ligand_induced", [r"ligand[- ]induced conformational", r"ligand binding (?:induces|causes|triggers)", r"allosteric"]),
    ("local_conformational_change", [r"conformational (?:change|rearrangement|transition)", r"conformational switch"]),
]
CLASS_RES = [(name, [re.compile(p, re.IGNORECASE) for p in pats]) for name, pats in CLASS_RULES]


def die(msg):
    print(f"ASSERTION FAILED: {msg}", file=sys.stderr)
    sys.exit(1)


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def open_sifts_csv(path):
    f = gzip.open(path, "rt")
    first = f.readline()
    while first.startswith("#"):
        first = f.readline()
    header = first.rstrip("\n").split(",")
    return header, csv.reader(f, delimiter=",")


def classify_text(title, abstract):
    text = (title or "") + " . " + (abstract or "")
    for name, regs in CLASS_RES:
        hit_kws = [m.pattern for m in regs if m.search(text)]
        if hit_kws:
            return name, ";".join(hit_kws)
    return "other_or_unclear", ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--source-manifest", required=True)
    ap.add_argument("--strict-positive-table", required=True)
    ap.add_argument("--background-b", required=True)
    ap.add_argument("--background-a", default="data/curated/fold_switch_unlabeled_sequence.tsv.gz")
    ap.add_argument("--chains-subtable", required=True)
    ap.add_argument("--p10-qc", default="reports/fs_three_layer/background_qc.json")
    ap.add_argument("--evidence-dir", default="data/raw/europepmc/2026-09-23")
    ap.add_argument("--search-mode", choices=["live", "cached"], default="cached")
    ap.add_argument("--reuse-archive", action="store_true",
                    help="live 模式下已存在归档的候选直接重放（仅新候选联网获取），降低快照漂移")
    ap.add_argument("--run-id", required=True)
    args = ap.parse_args()

    import yaml
    cfg = yaml.safe_load(open(args.config))
    root = "."
    sifts = "data/raw/sifts/2026-09-23"
    fam_files = {
        "pfam": os.path.join(sifts, "pdb_chain_pfam.csv.gz"),
        "cath": os.path.join(sifts, "pdb_chain_cath_uniprot.csv.gz"),
        "scop2fa": os.path.join(sifts, "pdb_chain_scop2_uniprot.csv.gz"),
    }
    pubmed_file = os.path.join(sifts, "pdb_pubmed.csv.gz")
    os.makedirs(args.evidence_dir, exist_ok=True)

    # ---- 0) 输入完整性 ----
    manifest = {}
    with open(args.source_manifest) as f:
        for row in csv.DictReader(f, delimiter="\t"):
            manifest[row["file_path"]] = row["sha256"]
    for p in list(fam_files.values()) + [pubmed_file]:
        if manifest.get(p) != sha256_file(p):
            die(f"manifest sha256 mismatch: {p}")
    p10_qc = json.load(open(args.p10_qc))
    for p in [args.background_b, args.chains_subtable]:
        key = os.path.relpath(p, ".")
        if p10_qc["output_sha256"].get(key) != sha256_file(p):
            die(f"P1.10 qc sha256 mismatch: {p}")
    print(f"[0] inputs verified (4 raw family/pubmed + background B + subtable)")

    # ---- 1) 背景 B + FS 排除集合 ----
    bg = {}
    with gzip.open(args.background_b, "rt") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            bg[row["uniprot_accession"]] = row
    fs_excluded = set()
    with open("data/curated/fold_switch_unlabeled_structure_exclusions.tsv") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            fs_excluded.add(row["uniprot_accession"])
    print(f"[1] background B proteins={len(bg)} fs_excluded={len(fs_excluded)}")

    # strict positives（双源核对沿用 P1.09 审计）
    strict_pairs = defaultdict(list)
    strict_set = set()
    # B1 修复（2026-09-23 审核）：阳性长度与序列 sha 取自 P1.09 audit fasta
    # （背景 B 已剔除阳性，从背景查阳性长度恒空——原实现的窗口失效根因）
    audit_dir = "data/raw/audit_uniprot/2026-09-23"
    raw_checksums = {}
    with open("data/manifests/checksums_raw.tsv") as f:
        for line in f:
            parts = line.rstrip("\n").split(None, 1)
            if len(parts) == 2:
                raw_checksums[parts[1]] = parts[0]
    with open(args.strict_positive_table) as f:
        for row in csv.DictReader(f, delimiter="\t"):
            if row["tier"] == "strict_state_candidate":
                for c in ("uniprot_a", "uniprot_b"):
                    u = (row.get(c) or "").strip()
                    if u:
                        strict_set.add(u)
                        strict_pairs[u].append(row["pair_id"])
    audit_set = set()
    with open("reports/fs_three_layer/strict_positive_audit.tsv") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            for c in ("uniprot_a", "uniprot_b"):
                u = (row.get(c) or "").strip()
                if u:
                    audit_set.add(u)
    if strict_set != audit_set:
        die(f"strict set mismatch vs audit: {sorted(strict_set ^ audit_set)}")
    pos_len, pos_sha = {}, {}
    for u in sorted(strict_set):
        fp = os.path.join(audit_dir, f"{u}.fasta")
        if not os.path.exists(fp):
            die(f"B1 audit fasta missing: {fp}")
        if raw_checksums.get(fp) != sha256_file(fp):
            die(f"B1 audit fasta checksum mismatch: {fp}")
        seq = "".join(l.strip() for l in open(fp) if not l.startswith(">"))
        if not seq:
            die(f"B1 empty audit fasta: {fp}")
        pos_len[u] = len(seq)
        pos_sha[u] = hashlib.sha256(seq.encode("ascii", "strict")).hexdigest()
    if len(pos_len) != len(strict_set):
        die("B1 incomplete positive lengths")
    print(f"[1b] strict positives={len(strict_set)} lengths={pos_len}")

    # ---- 2) 家族映射（三系统，按 accession 聚合；对 strict positives 与背景同源取） ----
    fam = defaultdict(lambda: defaultdict(set))
    for system, path in fam_files.items():
        col = {"pfam": "PFAM_ID", "cath": "CATH_ID", "scop2fa": "FA_DOMID"}[system]
        header, rd = open_sifts_csv(path)
        ix = {"PDB": header.index("PDB"), "CHAIN": header.index("CHAIN"),
              "SP": header.index("SP_PRIMARY"), "FAM": header.index(col)}
        for row in rd:
            try:
                sp = row[ix["SP"]].strip()
                v = row[ix["FAM"]].strip()
            except IndexError:
                continue
            if sp and v:
                fam[sp][system].add(v)
    print(f"[2] family map accessions={len(fam)}")

    pos_fam = {u: fam.get(u, {}) for u in strict_set}
    pos_with_family = [u for u in strict_set if sum(len(s) for s in pos_fam[u].values()) > 0]
    if len(pos_with_family) < len(strict_set):
        print(f"[2b] WARNING strict positives without any family mapping: "
              f"{sorted(strict_set - set(pos_with_family))}")

    # ---- 3) 候选生成 ----
    top_n = int(cfg["matching_rules"]["top_n_per_positive"])
    def parse_len(s):
        try:
            return int(s)
        except (TypeError, ValueError):
            return None
    bg_fam_shared = {}
    for cand, row in bg.items():
        cf = fam.get(cand)
        if not cf:
            continue
        for pos in pos_with_family:
            pf = pos_fam[pos]
            shared = {sysname: cf[sysname] & pf.get(sysname, set()) for sysname in ("pfam", "cath", "scop2fa")}
            n_shared = sum(1 for v in shared.values() if v)
            if n_shared >= 1:
                bg_fam_shared[(cand, pos)] = shared
    print(f"[3] (candidate,positive) family-sharing pairs={len(bg_fam_shared)}")

    cand_rows = []
    n_length_filtered = 0
    for pos in sorted(pos_with_family):
        pool = []
        for (cand, p), shared in bg_fam_shared.items():
            if p != pos:
                continue
            brows = bg[cand]
            lc, lp = parse_len(brows.get("sp_length_max")), pos_len[pos]
            if lc is not None and lp is not None:
                ratio = lc / lp
                if not (0.5 <= ratio <= 2.0):
                    n_length_filtered += 1
                    continue
                lenkey = abs(math.log2(ratio))
                ratio_str = f"{ratio:.4f}"
            else:
                lenkey = math.inf
                ratio_str = ""
            pool.append((-sum(1 for v in shared.values() if v), lenkey,
                         -int(brows.get("n_pdb_entries") or 0), cand, shared, ratio_str))
        pool.sort(key=lambda t: (t[0], t[1], t[2], t[3]))
        for rank, (_, lenkey, _, cand, shared, ratio_str) in enumerate(pool[:top_n], 1):
            cand_rows.append((cand, pos, shared, ratio_str, rank))
    unique_candidates = sorted({c for c, *_ in cand_rows})
    if len(unique_candidates) > int(cfg["matching_rules"]["max_total_candidates"]):
        die(f"unique candidates {len(unique_candidates)} > max {cfg['matching_rules']['max_total_candidates']}")
    print(f"[3b] candidate rows={len(cand_rows)} unique candidates={len(unique_candidates)} length_filtered={n_length_filtered}")

    # ---- 4) 每蛋白 PDB 集（子表）与独立研究数（去重 PMID） ----
    prot_pdbs = defaultdict(set)
    with gzip.open(args.chains_subtable, "rt") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            prot_pdbs[row["uniprot_accession"]].add(row["pdb_id"])
    pdb_pmids = defaultdict(set)
    header, rd = open_sifts_csv(pubmed_file)
    ix_p, ix_m = header.index("PDB"), header.index("PUBMED_ID")
    for row in rd:
        try:
            pdb_pmids[row[ix_p].strip().lower()].add(row[ix_m].strip())
        except IndexError:
            continue
    cand_pmids = {}
    for acc in unique_candidates:
        pm = set()
        for pdb in prot_pdbs.get(acc, ()):
            pm |= pdb_pmids.get(pdb, set())
        cand_pmids[acc] = pm

    def n_independent(acc):
        return len(cand_pmids[acc])

    # ---- 5) SP 候选的蛋白名与序列 sha（背景 A） ----
    names, cand_sha = {}, {}
    with gzip.open(args.background_a, "rt") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            names[row["uniprot_accession"]] = row["protein_name"]
            cand_sha[row["uniprot_accession"]] = row["sequence_sha256"]

    # ---- 6) 检索（live 或 cached 重放） ----
    queries_out = []
    hits_out = []
    search_results = {}
    meta_path = os.path.join(args.evidence_dir, "search_meta.json")
    live_statuses = {}
    if args.search_mode == "live":
        q_date = os.popen("TZ=Asia/Shanghai date '+%Y-%m-%d %H:%M'").read().strip()
    else:
        if not os.path.exists(meta_path):
            die("cached mode requires search_meta.json from a prior live run")
        meta = json.load(open(meta_path))
        q_date = meta["query_date"]
        live_statuses = meta.get("http_status", {})
    for i, acc in enumerate(unique_candidates, 1):
        name = names.get(acc, "")
        if name:
            ident = f'("{acc}" OR "{name}")'
        else:
            ident = f'("{acc}")'
        query = f"{ident} AND {TERM_SET}"
        url = ("https://www.ebi.ac.uk/europepmc/webservices/rest/search?"
               + urllib.parse.urlencode({"query": query, "format": "json",
                                         "resultType": "core", "hitsPerPage": "25"}))
        cache_path = os.path.join(args.evidence_dir, f"{acc}.json")
        cf_cache_path = os.path.join(args.evidence_dir, f"{acc}.crossref.json")
        channel, status = "europepmc_primary", None
        page_blobs = []
        if args.search_mode == "cached":
            if os.path.exists(cache_path):
                status = live_statuses.get(acc, "200")  # 归档重放按 live 记录的原始状态
                page_blobs.append(open(cache_path, "rb").read())
                k = 2
                while True:
                    pf = os.path.join(args.evidence_dir, f"{acc}.page{k}.json")
                    if os.path.exists(pf):
                        page_blobs.append(open(pf, "rb").read())
                        k += 1
                    else:
                        break
            elif os.path.exists(cf_cache_path):
                channel, status = "crossref_fallback", live_statuses.get(acc, "200")
                page_blobs.append(open(cf_cache_path, "rb").read())
            else:
                die(f"cached mode but missing archive {cache_path}; run with --search-mode=live first")
        elif args.reuse_archive and os.path.exists(cache_path):
            status = live_statuses.get(acc, "200")
            page_blobs.append(open(cache_path, "rb").read())
            k = 2
            while True:
                pf = os.path.join(args.evidence_dir, f"{acc}.page{k}.json")
                if os.path.exists(pf):
                    page_blobs.append(open(pf, "rb").read())
                    k += 1
                else:
                    break
        else:
            data = None
            for attempt in range(3):
                try:
                    req = urllib.request.Request(url, headers={"User-Agent": "PhysInfoBench-P1.11/1.0"})
                    with urllib.request.urlopen(req, timeout=60) as r:
                        status, data = str(r.status), r.read()
                    break
                except Exception as e:
                    status, data = f"error:{type(e).__name__}", None
                    time.sleep(2 + 3 * attempt)
            if data is None:
                # Crossref 替补（仅 Europe PMC 失败时）
                cf_url = ("https://api.crossref.org/works?"
                          + urllib.parse.urlencode({"query.bibliographic": f"{acc} {name}".strip(),
                                                    "rows": "25"}))
                channel = "crossref_fallback"
                try:
                    req = urllib.request.Request(cf_url, headers={"User-Agent": "PhysInfoBench-P1.11/1.0"})
                    with urllib.request.urlopen(req, timeout=60) as r:
                        status, data = str(r.status), r.read()
                except Exception as e:
                    status = f"error:{type(e).__name__}"
            if data is not None:
                page_blobs.append(data)
                if channel == "europepmc_primary":
                    # 分页抓全（cursorMark；上限 500 条，超出标记截断并压至 PN-C）
                    page1 = json.loads(data)
                    total = int(page1.get("hitCount") or 0)
                    fetched = len((page1.get("resultList") or {}).get("result") or [])
                    next_url = page1.get("nextPageUrl")
                    while next_url and fetched < min(total, 500):
                        time.sleep(0.4)
                        try:
                            # M1 修复（2026-09-23 审核）：Europe PMC 的 nextPageUrl 含未编码
                            # 空格/引号（urllib 报 InvalidURL）——提取 cursorMark 后重编码重建
                            mcm = re.search(r"[?&]cursorMark=([^&]+)", next_url)
                            if not mcm:
                                break
                            fetch_url = url + "&cursorMark=" + urllib.parse.quote(mcm.group(1), safe="")
                            req = urllib.request.Request(fetch_url, headers={"User-Agent": "PhysInfoBench-P1.11/1.0"})
                            with urllib.request.urlopen(req, timeout=60) as r:
                                pdata = r.read()
                            pd = json.loads(pdata)
                            res = (pd.get("resultList") or {}).get("result") or []
                            if not res:
                                break
                            fetched += len(res)
                            page_blobs.append(pdata)
                            next_url = pd.get("nextPageUrl")
                        except Exception:
                            next_url = None
                            status = status + "_partial_pages"
                # 存档：page1 -> <acc>.json，后续页 -> <acc>.pageN.json；crossref -> <acc>.crossref.json
                if channel == "europepmc_primary":
                    for k_, blob in enumerate(page_blobs, 1):
                        suffix = ".json" if k_ == 1 else f".page{k_}.json"
                        with open(os.path.join(args.evidence_dir, acc + suffix), "wb") as f:
                            f.write(blob)
                else:
                    with open(cf_cache_path, "wb") as f:
                        f.write(data)
            time.sleep(0.4)  # 每候选请求间隔（限速纪律）
        data = page_blobs[0] if page_blobs else None
        sha16 = hashlib.sha256(data).hexdigest()[:16] if data else ""
        hit_count, hit_items, truncated = 0, [], False
        if data:
            if channel == "europepmc_primary":
                d0 = json.loads(data)
                hit_count = int(d0.get("hitCount") or 0)
                seen = set()
                for blob in page_blobs:
                    d = json.loads(blob)
                    for item in (d.get("resultList") or {}).get("result") or []:
                        key = (item.get("pmid") or "") + "|" + (item.get("doi") or "") + "|" + (item.get("title") or "")[:60]
                        if key in seen:
                            continue
                        seen.add(key)
                        title = item.get("title") or ""
                        abstract = item.get("abstractText") or ""
                        cat, kws = classify_text(title, abstract)
                        hit_items.append({
                            "pmid": item.get("pmid") or "", "doi": item.get("doi") or "",
                            "title": title, "pub_year": item.get("pubYear") or "",
                            "category": cat, "keywords": kws,
                        })
                truncated = len(hit_items) < hit_count
            else:
                d = json.loads(data)
                for item in (d.get("message") or {}).get("items") or []:
                    title = (item.get("title") or [""])[0] if isinstance(item.get("title"), list) else (item.get("title") or "")
                    cat, kws = classify_text(title, "")
                    hit_items.append({
                        "pmid": "", "doi": item.get("DOI") or "",
                        "title": title, "pub_year": (item.get("issued", {}).get("date-parts", [[None]])[0][0]) or "",
                        "category": cat, "keywords": kws,
                    })
                hit_count = len(hit_items)
        queries_out.append([acc, channel, url, q_date, status, hit_count,
                            "false" if truncated else "true", sha16])
        if args.search_mode == "live":
            live_statuses[acc] = status
        search_results[acc] = {"channel": channel, "status": status, "hit_count": hit_count,
                               "fetched": len(hit_items), "truncated": truncated,
                               "hits": hit_items}
        if args.search_mode == "live" and i % 25 == 0:
            print(f"[6] searched {i}/{len(unique_candidates)}")

    if args.search_mode == "live":
        with open(meta_path, "w") as f:
            json.dump({"query_date": q_date, "run_id": args.run_id,
                       "http_status": live_statuses}, f, ensure_ascii=False, indent=1, sort_keys=True)

    # ---- 7) 组装候选表 + PN 层 ----
    version_str = "P1.10 background B (SIFTS 2026-09-13 + wwPDB 2026-09-12) + SIFTS families 2026-09-20"
    rows = []
    for cand, pos, shared, ratio_str, rank in cand_rows:
        b = bg[cand]
        sr = search_results[cand]
        exit_cats = sorted({h["category"] for h in sr["hits"] if h["category"] in EXIT_CATEGORIES})
        exited = "true" if exit_cats else "false"
        review_ptr = ";".join(f"{h['pmid'] or h['doi']}:{h['category']}" for h in sr["hits"]
                              if h["category"] in EXIT_CATEGORIES)
        ok_status = sr["status"].startswith("200")
        search_ok = ok_status and not sr["truncated"]
        n_ind = n_independent(cand)
        mcov = float(b["mapped_coverage_max"]) if b.get("mapped_coverage_max") else None
        if exited == "true":
            tier = "EXITED"
        elif search_ok and n_ind >= 2 and (mcov is not None and mcov >= 0.70):
            tier = "PN-A"
        elif search_ok and int(b["n_pdb_entries"]) >= 2:
            tier = "PN-B"
        else:
            tier = "PN-C"
        rows.append({
            "candidate_id": f"PNC-{cand}", "uniprot_accession": cand,
            "matched_strict_positive": pos, "match_rank": rank,
            "shared_family_systems": sum(1 for v in shared.values() if v),
            "shared_pfam": ";".join(sorted(shared["pfam"])),
            "shared_cath": ";".join(sorted(shared["cath"])),
            "shared_scop2fa": ";".join(sorted(shared["scop2fa"])),
            "sp_length": b.get("sp_length_max") or "",
            "length_ratio_to_positive": ratio_str,
            "n_pdb_entries": b["n_pdb_entries"],
            "n_mapped_chains": b["n_mapped_chains"],
            "mapped_coverage_max": b.get("mapped_coverage_max") or "",
            "observed_coverage_max": b.get("observed_coverage_max") or "",
            "experiment_methods": b["experiment_methods"],
            "best_resolution": b.get("best_resolution") or "",
            "n_independent_studies": n_ind,
            "search_status": ("completed_truncated" if (ok_status and sr["truncated"]) else ("completed" if search_ok else sr["status"])),
            "search_channel": sr["channel"],
            "hit_count": sr["hit_count"],
            "hit_exit_categories": ";".join(exit_cats),
            "exited_to_review_queue": exited,
            "review_queue_pointer": review_ptr,
            "manual_review_status": "pending_fulltext_review",
            "pn_tier": tier,
            "biological_target": "",
            "background_source_version": version_str,
        })
    rows.sort(key=lambda r: (r["uniprot_accession"], r["matched_strict_positive"]))

    out_c = cfg["output_tables"]["candidates"]
    cols = list(rows[0].keys())
    with open(out_c, "w", newline="") as f:
        f.write("\t".join(cols) + "\n")
        w = csv.DictWriter(f, fieldnames=cols, delimiter="\t", lineterminator="\n")
        w.writerows(rows)

    with open(cfg["output_tables"]["literature_queries"], "w", newline="") as f:
        f.write("candidate_accession\tchannel\tquery_url\tquery_date_asia_shanghai\thttp_status\thit_count\tfetched_all_hits\tresponse_sha256_prefix\n")
        csv.writer(f, delimiter="\t", lineterminator="\n").writerows(queries_out)

    with open(cfg["output_tables"]["literature_hits"], "w", newline="") as f:
        f.write("candidate_accession\tchannel\tpmid\tdoi\ttitle\tpub_year\tclassified_category\tmatched_keywords\tis_pubmed_of_mapped_pdb\n")
        hw = csv.writer(f, delimiter="\t", lineterminator="\n")
        for acc in unique_candidates:
            for h in search_results[acc]["hits"]:
                hw.writerow([acc, search_results[acc]["channel"], h["pmid"], h["doi"],
                             h["title"], h["pub_year"], h["category"], h["keywords"],
                             "true" if (h["pmid"] and h["pmid"] in cand_pmids.get(acc, set())) else "false"])

    # ---- 8) 机器断言 ----
    accs = {r["uniprot_accession"] for r in rows}
    if accs & fs_excluded:
        die(f"B1 FS intersection non-empty: {sorted(accs & fs_excluded)[:5]}")
    if not all(a in bg for a in accs):
        die("B1 candidate not in background B")
    if len(queries_out) != len(unique_candidates):
        die(f"B3 queries {len(queries_out)} != candidates {len(unique_candidates)}")
    for q in queries_out:
        if not q[3]:
            die("B3 empty query date")
    for r in rows:
        if r["biological_target"] != "":
            die("B2 biological_target not empty")
        if r["exited_to_review_queue"] == "true" and not r["review_queue_pointer"]:
            die("B5 exit without pointer")
        if r["exited_to_review_queue"] == "false" and r["hit_exit_categories"]:
            die("B5 exit categories without flag")
    # B3 补强：queries 表 sha 前缀与归档 page1 逐行复核
    for q in queries_out:
        acc = q[0]
        arc = os.path.join(args.evidence_dir, f"{acc}.json")
        if os.path.exists(arc):
            if hashlib.sha256(open(arc, "rb").read()).hexdigest()[:16] != q[7]:
                die(f"B3 archive sha prefix mismatch for {acc}")
        elif not os.path.exists(os.path.join(args.evidence_dir, f"{acc}.crossref.json")):
            die(f"B3 missing archive for {acc}")
    # B6：5 个蛋白独立研究数从 raw 独立重解析（重开 pubmed 文件，不复用内存结构）
    import random
    rng = random.Random(13)
    for acc in rng.sample(sorted(accs), min(5, len(accs))):
        pm = set()
        with gzip.open(pubmed_file, "rt") as pf:
            first = pf.readline()
            while first.startswith("#"):
                first = pf.readline()
            rd2 = csv.reader(pf, delimiter=",")
            for row in rd2:
                try:
                    if row[0].strip().lower() in prot_pdbs.get(acc, set()):
                        pm.add(row[2].strip())
                except IndexError:
                    continue
        if len(pm) != n_independent(acc):
            die(f"B6 recount mismatch {acc}: {len(pm)} vs {n_independent(acc)}")
    # same_sequence 检查（可核子集）：候选序列 sha 不得与任何 strict positive 的 sha 相同
    strict_sha_values = set(pos_sha.values())
    seq_collisions = sorted(a for a in accs if cand_sha.get(a) and cand_sha[a] in strict_sha_values)
    if seq_collisions:
        die(f"same_sequence collision: {seq_collisions}")
    n_not_checkable_seq = sum(1 for a in accs if not cand_sha.get(a))

    tier_counts = defaultdict(int)
    for r in rows:
        tier_counts[r["pn_tier"]] += 1
    print(f"[8] assertions passed; tiers={dict(tier_counts)}")

    qc = {
        "run_id": args.run_id,
        "generated": q_date,
        "strict_positives": sorted(strict_set),
        "positives_without_family": sorted(strict_set - set(pos_with_family)),
        "family_pairs_before_ranking": len(bg_fam_shared),
        "length_filtered_pairs": n_length_filtered,
        "candidate_rows": len(rows),
        "unique_candidates": len(unique_candidates),
        "top_n_per_positive": top_n,
        "search_mode": args.search_mode,
        "tier_counts_rows": dict(tier_counts),
        "tier_counts_unique_candidates": {
            t: len({r["uniprot_accession"] for r in rows if r["pn_tier"] == t}) for t in tier_counts
        },
        "search_channel_counts": dict(
            __import__("collections").Counter(q[1] for q in queries_out)),
        "no_hit_candidates": sum(1 for q in queries_out if q[5] == 0),
        "truncated_candidates": sum(1 for q in queries_out if q[6] == "false"),
        "hit_count_sum": sum(q[5] for q in queries_out),
        "fetched_classified_rows": sum(len(search_results[a]["hits"]) for a in unique_candidates),
        "paginated_candidates_with_pages": sum(
            1 for a in unique_candidates
            if os.path.exists(os.path.join(args.evidence_dir, f"{a}.page2.json"))),
        "same_sequence_checkable": len(accs) - n_not_checkable_seq,
        "same_sequence_not_checkable_trembl": n_not_checkable_seq,
        "same_sequence_collisions": 0,
        "manual_review_pending_all": True,
        "version_skew_note": cfg["meta"]["version_skew_note"],
        "candidates_sha256": sha256_file(out_c),
        "queries_sha256": sha256_file(cfg["output_tables"]["literature_queries"]),
        "hits_sha256": sha256_file(cfg["output_tables"]["literature_hits"]),
    }
    with open(cfg["output_tables"]["qc"], "w") as f:
        json.dump(qc, f, ensure_ascii=False, indent=1, sort_keys=True)
    print(f"[9] qc written: {cfg['output_tables']['qc']}")
    print("ALL ASSERTIONS PASSED")


if __name__ == "__main__":
    main()
