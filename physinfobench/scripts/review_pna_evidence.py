#!/usr/bin/env python3
"""P1.16 L3 对照证据与检索快照（G1 前补证；只读 P1.11 产物，不生成匹配集）。

A. 候选优先级清单：每个 strict positive 的 PN-A 候选按 P1.11 match_rank 取前 3 主选+次 2 备选。
B. 逐候选证据核查：其映射 PDB 的主引文（RCSB GraphQL 批查）→ OA 全文（PMC fullTextXML）
   → 程序化探针：accession/蛋白名/PDB 码/状态关键词/转换类关键词（fold switch|domain swap）。
   语义：检索无命中≠可靠阴性；转换类证据若在全文出现→标记（仅登记，不改 PN 层——层变更属审核流程）。
C. 截断回补：7 个 fetched_all_hits=false 候选在独立目录全量补抓（不触碰 P1.11 冻结归档）。
D. 汇总：每阳性"核查后仍可用"的 PN-A 主选/备选计数。
"""
import csv
import gzip
import hashlib
import json
import os
import re
import time
import urllib.parse
import urllib.request
from collections import defaultdict

UA = "PhysInfoBench-P1.16/1.0"
OUT_DIR = "data/raw/pna_fulltext/2026-09-23"
BACKFILL_DIR = "data/raw/europepmc_p116_backfill/2026-09-23"
OUT_TSV = "reports/fs_three_layer/pna_evidence_review.tsv"
OUT_JSON = "reports/fs_three_layer/pna_evidence_review_qc.json"
TRANSFORM_KW = ["fold switch", "fold-switch", "metamorphic", "alternative fold",
                "domain swap", "domain-swapp", "swapped dimer"]
STATE_KW = ["conformational change", "conformational transition", "open and closed",
            "active and inactive", "allosteric", "two conformations"]


def fetch(url, timeout=60, retries=3):
    last = None
    for a in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": UA})
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return r.read()
        except Exception as e:
            last = e
            time.sleep(2 + 3 * a)
    raise last


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(BACKFILL_DIR, exist_ok=True)

    # ---- A. 候选优先级 ----
    rows = list(csv.DictReader(open("data/curated/fold_switch_putative_negative_candidates.tsv"), delimiter="\t"))
    per_pos = defaultdict(list)
    for r in rows:
        if r["pn_tier"] == "PN-A":
            per_pos[r["matched_strict_positive"]].append((int(r["match_rank"]), r["uniprot_accession"]))
    selected = {}   # acc -> {"primary_for": [...], "backup_for": [...]}
    for pos, lst in sorted(per_pos.items()):
        lst.sort()
        for rank, acc in lst[:3]:
            selected.setdefault(acc, {"primary_for": [], "backup_for": []})["primary_for"].append(pos)
        for rank, acc in lst[3:5]:
            selected.setdefault(acc, {"primary_for": [], "backup_for": []})["backup_for"].append(pos)
    print(f"[A] unique selected candidates={len(selected)} "
          f"(primary slots={sum(len(v['primary_for']) for v in selected.values())}, "
          f"backup slots={sum(len(v['backup_for']) for v in selected.values())})")

    # ---- B. 逐候选证据 ----
    # B1) 候选映射 PDB（链级子表，确定性取法：按 pdb 升序全量）
    prot_pdbs = defaultdict(set)
    with gzip.open("data/curated/fold_switch_unlabeled_structure_chains.tsv.gz", "rt") as f:
        for r in csv.DictReader(f, delimiter="\t"):
            prot_pdbs[r["uniprot_accession"]].add(r["pdb_id"])
    need_pdbs = sorted({p for acc in selected for p in prot_pdbs.get(acc, ())})
    print(f"[B] pdbs to cite-check={len(need_pdbs)}")
    # B2) 主引文批量（GraphQL，200/批，带重试——大批量响应曾出现 IncompleteRead）
    cite = {}
    cit_cache = os.path.join(OUT_DIR, "rcsb_primary_citations.json")
    if os.path.exists(cit_cache):
        cite = json.load(open(cit_cache))  # 快照优先
    todo = [p for p in need_pdbs if p not in cite]
    for i in range(0, len(todo), 200):
        chunk = todo[i:i + 200]
        payload = {"query": "query($ids:[String!]!){ entries(entry_ids:$ids){ rcsb_id "
                            "rcsb_primary_citation{ pdbx_database_id_DOI pdbx_database_id_PubMed title } } }",
                   "variables": {"ids": [p.upper() for p in chunk]}}
        body = None
        for a in range(4):
            try:
                req = urllib.request.Request("https://data.rcsb.org/graphql",
                                             data=json.dumps(payload).encode(),
                                             headers={"Content-Type": "application/json", "User-Agent": UA})
                with urllib.request.urlopen(req, timeout=180) as r:
                    body = r.read()
                break
            except Exception as e:
                print(f"  graphql batch {i//200} attempt {a+1}: {type(e).__name__}")
                time.sleep(3 + 5 * a)
        if body is None:
            raise SystemExit(f"graphql batch {i//200} failed after retries")
        for e in json.loads(body)["data"]["entries"]:
            c = e.get("rcsb_primary_citation") or {}
            cite[e["rcsb_id"].lower()] = {"doi": (c.get("pdbx_database_id_DOI") or "").strip(),
                                          "pmid": str(c.get("pdbx_database_id_PubMed") or "").strip(),
                                          "title": (c.get("title") or "").replace("\t", " ")[:160]}
        time.sleep(0.5)
    with open(os.path.join(OUT_DIR, "rcsb_primary_citations.json"), "w") as f:
        json.dump(cite, f, indent=1, sort_keys=True)
    # 候选名（背景 A）
    names = {}
    with gzip.open("data/curated/fold_switch_unlabeled_sequence.tsv.gz", "rt") as f:
        for r in csv.DictReader(f, delimiter="\t"):
            if r["uniprot_accession"] in selected:
                names[r["uniprot_accession"]] = r["protein_name"]
    # B3) 唯一 DOI 解析（记忆化）
    dois = sorted({c["doi"] for acc in selected for p in prot_pdbs.get(acc, ())
                   for c in [cite.get(p, {})] if c.get("doi")})
    doi_raw = os.path.join(OUT_DIR, "epmc_doi_resolution.json")
    doi_info = {}
    if os.path.exists(doi_raw):
        doi_info = json.load(open(doi_raw))  # 快照优先
    for doi in dois:
        if doi in doi_info:
            continue
        rec = {"is_oa": False, "pmcid": "", "status": "not_looked_up"}
        for a in range(3):
            try:
                b = fetch("https://www.ebi.ac.uk/europepmc/webservices/rest/search?"
                          + urllib.parse.urlencode({"query": f'DOI:"{doi}"', "format": "json",
                                                    "resultType": "core"}))
                d = json.loads(b)
                item = (d.get("resultList") or {}).get("result") or []
                rec = ({"is_oa": item[0].get("isOpenAccess") == "Y",
                        "pmcid": item[0].get("pmcid") or "", "status": "resolved"} if item
                       else {"is_oa": False, "pmcid": "", "status": "no_europepmc_record"})
                break
            except Exception:
                rec = {"is_oa": False, "pmcid": "", "status": "lookup_error"}
                time.sleep(2 + 3 * a)
        doi_info[doi] = rec
        time.sleep(0.4)
    with open(doi_raw, "w") as f:
        json.dump(doi_info, f, indent=1, sort_keys=True)
    paper_cache_dir = os.path.join(OUT_DIR, "papers")
    os.makedirs(paper_cache_dir, exist_ok=True)
    def paper_probe(acc, doi, pdbs, name):
        """(候选,DOI) 粒度快照：首次求值存档 JSON，其后一律重放——保证重跑确定。"""
        key = f"{acc}__{doi.replace('/', '_')}.json"
        fp = os.path.join(paper_cache_dir, key)
        if os.path.exists(fp):
            return json.load(open(fp))
        info = doi_info.get(doi, {})
        text, basis = "", "none"
        if info.get("is_oa") and info.get("pmcid"):
            text = get_fulltext(info["pmcid"])
            if text:
                basis = "fulltext"
        if basis == "none":
            pmid = next((cite[p]["pmid"] for p in pdbs if cite.get(p, {}).get("doi") == doi
                         and cite[p].get("pmid")), "")
            if pmid:
                try:
                    b = fetch("https://www.ebi.ac.uk/europepmc/webservices/rest/search?"
                              + urllib.parse.urlencode({"query": f"EXT_ID:{pmid} AND SRC:MED",
                                                        "format": "json", "resultType": "core"}))
                    with open(os.path.join(OUT_DIR, f"MED_{pmid}.json"), "wb") as f:
                        f.write(b)
                    item = (json.loads(b).get("resultList") or {}).get("result") or [{}]
                    text = item[0].get("abstractText") or ""
                    if text:
                        basis = "abstract"
                    time.sleep(0.4)
                except Exception:
                    text = ""
        tl = text.lower()
        rec = {
            "basis": basis, "text_len": len(text),
            "pdb_cited": any(p.upper() in text.upper() for p in pdbs) if text else False,
            "acc_cited": acc in text if text else False,
            "name_cited": bool(name) and name.lower()[:30] in tl,
            "transform_kw": [k for k in TRANSFORM_KW if k in tl] if text else [],
            "state_kw": [k for k in STATE_KW if k in tl] if text else [],
        }
        with open(fp, "w") as f:
            json.dump(rec, f, ensure_ascii=False, sort_keys=True)
        return rec

    ft_cache = {}
    def get_fulltext(pmcid):
        if pmcid in ft_cache:
            return ft_cache[pmcid]
        fp = os.path.join(OUT_DIR, f"{pmcid}.xml")
        if os.path.exists(fp):
            with open(fp, "rb") as f:
                xml = f.read()
            text = re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", xml.decode("utf-8", "ignore")))
            ft_cache[pmcid] = text
            return text
        try:
            xml = fetch(f"https://www.ebi.ac.uk/europepmc/webservices/rest/{pmcid}/fullTextXML")
            text = re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", xml.decode("utf-8", "ignore")))
            with open(os.path.join(OUT_DIR, f"{pmcid}.xml"), "wb") as f:
                f.write(xml)
        except Exception:
            text = ""
        ft_cache[pmcid] = text
        return text
    # B4) 逐候选核查
    out_rows = []
    ft_n_total = ab_n_total = 0  # qc 全局累计（papers_basis）
    for acc in sorted(selected):
        ft_n = ab_n = 0  # 行内口径：本候选的全文/摘要篇数（第一轮冻结语义）
        pdbs = sorted(prot_pdbs.get(acc, ()))
        name = names.get(acc, "")
        seen_dois = []
        for p in pdbs:
            d = cite.get(p, {}).get("doi")
            if d and d not in seen_dois:
                seen_dois.append(d)
        checked = transform_found = strong = 0
        ev_detail = []
        for doi in seen_dois[:4]:   # 每候选至多核查 4 篇主引文（预注册上限）
            rec = paper_probe(acc, doi, pdbs, name)
            checked += 1
            basis = rec["basis"]
            if basis == "fulltext":
                ft_n += 1
                ft_n_total += 1
            elif basis == "abstract":
                ab_n += 1
                ab_n_total += 1
            if rec["pdb_cited"] or rec["acc_cited"]:
                strong += 1
                ident = "strong_pdb_or_acc_cited"
            elif rec["name_cited"]:
                ident = "moderate_name_cited"
            else:
                ident = "weak_no_identity_probe"
            tkw = rec["transform_kw"]
            skw = rec["state_kw"]
            if tkw:
                transform_found += 1
                ev_detail.append(f"{doi}: transform_kw={tkw[:3]}")
            elif skw:
                ev_detail.append(f"{doi}: state_kw={skw[:2]},{ident}")
            elif basis != "none":
                ev_detail.append(f"{doi}: {ident}")
            else:
                ev_detail.append(f"{doi}: no_text,{ident}")
        out_rows.append({
            "uniprot_accession": acc,
            "primary_for": ";".join(selected[acc]["primary_for"]),
            "backup_for": ";".join(selected[acc]["backup_for"]),
            "protein_name": name, "n_pdb_mapped": len(pdbs),
            "n_papers_checked": checked, "n_fulltext": ft_n, "n_abstract": ab_n,
            "textual_corroboration_papers": strong,
            "textual_corroboration_level": ("strong" if strong else
                                            ("name_only" if any("moderate_name" in d for d in ev_detail) else "none")),
            "identity_basis": "structural: SIFTS UniProt→PDB→primary citation（构造性身份链，2026-09-23 设计修正——"
                              "文字探针仅作附加佐证，不作身份依据）",
            "transform_kw_papers": transform_found,
            "detail": " | ".join(ev_detail)[:400],
            "post_check_usable": ("flag_transform_kw" if transform_found else
                                  ("yes" if checked >= 1 else "no_papers")),
        })
    cols = list(out_rows[0].keys())
    with open(OUT_TSV, "w", newline="") as f:
        f.write("\t".join(cols) + "\n")
        w = csv.DictWriter(f, fieldnames=cols, delimiter="\t", lineterminator="\n")
        w.writerows(out_rows)
    # D. 每阳性核查后可用计数
    per_pos_avail = {}
    for pos in sorted(per_pos):
        prim = [r for r in out_rows if pos in r["primary_for"].split(";")]
        back = [r for r in out_rows if pos in r["backup_for"].split(";")]
        per_pos_avail[pos] = {
            "pn_a_total": len(per_pos[pos]),
            "primary_slots": len(prim), "primary_usable": sum(1 for r in prim if r["post_check_usable"] == "yes"),
            "primary_flagged": sum(1 for r in prim if r["post_check_usable"] == "flag_transform_kw"),
            "primary_weak": sum(1 for r in prim if r["post_check_usable"] == "weak_identity"),
            "backup_slots": len(back), "backup_usable": sum(1 for r in back if r["post_check_usable"] == "yes"),
        }

    # ---- C. 截断回补（独立目录，7 候选）----
    # 冻结纪律（2026-09-23 第三轮审核后确立）：默认只重放已存档页（活动靶根因）；
    # 只有显式设 P116_BACKFILL_LIVE=1 才联网推进，且推进后须重生成 manifest 并同步文档终态。
    backfill_live = os.environ.get("P116_BACKFILL_LIVE") == "1"
    qrows = list(csv.DictReader(open("data/evidence/fold_switch_negative_literature_queries.tsv"), delimiter="\t"))
    truncated = [q["candidate_accession"] for q in qrows if q["fetched_all_hits"] == "false"]
    backfill_summary = {}
    meta = json.load(open("data/raw/europepmc/2026-09-23/search_meta.json"))
    qdate = meta["query_date"]
    for acc in truncated:
        arc1 = f"data/raw/europepmc/2026-09-23/{acc}.json"
        page1 = json.load(open(arc1))
        total = int(page1.get("hitCount") or 0)
        blobs = []
        # 以现存连续页为准（快照不回退）：找到最高连续 pageN
        have = 1
        while os.path.exists(os.path.join(BACKFILL_DIR, f"{acc}.page{have+1}.json")):
            have += 1
        fetched = 0
        for k in range(1, have + 1):
            fpk = f"data/raw/europepmc/2026-09-23/{acc}.json" if k == 1 else os.path.join(BACKFILL_DIR, f"{acc}.page{k}.json")
            d_k = json.load(open(fpk))
            fetched += len((d_k.get("resultList") or {}).get("result") or [])
        nxt = page1.get("nextPageUrl")
        # 跳到 have+1 页（cursor 需逐页取，无法跳读——若 have>1 则从最后现存页的 nextPageUrl 继续）
        if have > 1:
            dk = json.load(open(os.path.join(BACKFILL_DIR, f"{acc}.page{have}.json")))
            nxt = dk.get("nextPageUrl")
        pages = 0
        if not nxt or not backfill_live:
            status = "no_next_page" if not nxt else ("frozen_replay" if fetched >= min(total, 800) else "frozen_partial")
            backfill_summary[acc] = {"hit_count": total, "fetched": fetched, "status": status}
            continue
        with open(arc1) as f1:
            page1_meta = json.load(f1)
        base = ("https://www.ebi.ac.uk/europepmc/webservices/rest/search?"
                + urllib.parse.urlencode({"query": page1_meta.get("request", {}).get("queryString", ""),
                                          "format": "json", "resultType": "core", "hitsPerPage": "25"}))
        cap = 800
        pages = 0
        while nxt and fetched < min(total, cap):
            time.sleep(0.4)
            mcm = re.search(r"[?&]cursorMark=([^&]+)", nxt)
            if not mcm:
                break
            try:
                b = fetch(base + "&cursorMark=" + urllib.parse.quote(mcm.group(1), safe=""))
                pd = json.loads(b)
                res = (pd.get("resultList") or {}).get("result") or []
                if not res:
                    break
                fetched += len(res)
                pages += 1
                have += 1
                with open(os.path.join(BACKFILL_DIR, f"{acc}.page{have}.json"), "wb") as f:
                    f.write(b)
                nxt = pd.get("nextPageUrl")
            except Exception:
                nxt = None
        backfill_summary[acc] = {"hit_count": total, "fetched_after_backfill": fetched,
                                 "pages_total_backfilled": have - 1, "pages_this_run": pages,
                                 "status": "complete" if fetched >= min(total, cap) else
                                           ("capped_800" if total > cap and fetched >= cap else "partial")}

    # 全部命中中转换类关键词复扫（回补后）
    exit_scan = {}
    for acc in truncated:
        files = [f"data/raw/europepmc/2026-09-23/{acc}.json"] + \
                sorted(os.path.join(BACKFILL_DIR, f) for f in os.listdir(BACKFILL_DIR)
                       if f.startswith(acc + ".page"))
        n_exit_kw = 0
        for fp in files:
            if not os.path.exists(fp):
                continue
            d = json.load(open(fp))
            for item in (d.get("resultList") or {}).get("result") or []:
                tl = ((item.get("title") or "") + " " + (item.get("abstractText") or "")).lower()
                if any(k in tl for k in TRANSFORM_KW):
                    n_exit_kw += 1
        exit_scan[acc] = n_exit_kw

    qc = {
        "generated": os.popen("TZ=Asia/Shanghai date '+%Y-%m-%d %H:%M'").read().strip(),
        "selection_rule": "每阳性 PN-A 按 match_rank 前 3 主选+次 2 备选；每候选核查其映射 PDB 主引文至多 4 篇（预注册）",
        "unique_candidates": len(selected),
        "papers_basis": {"fulltext_endpoints": ft_n_total, "abstract_endpoints": ab_n_total,
                         "note": "全候选累计；TSV 行内 n_fulltext/n_abstract 为每候选口径"},
        "per_positivity_availability": per_pos_avail,
        "truncated_backfill": backfill_summary,
        "exit_keyword_scan_after_backfill": exit_scan,
        "semantics": "检索无命中≠可靠阴性；transform_kw 命中仅登记复核，不自动改 PN 层；回补不触碰 P1.11 冻结归档",
    }
    with open(OUT_JSON, "w") as f:
        json.dump(qc, f, ensure_ascii=False, indent=1, sort_keys=True)
    # 回补 manifest 与 qc 同步生成（第三轮审核根因处置：文档/清单不再手工分头维护）
    bf_files = sorted(os.listdir(BACKFILL_DIR)) if os.path.isdir(BACKFILL_DIR) else []
    with open("data/manifests/epmc_p116_backfill_checksums.tsv", "w") as mf:
        for fn in bf_files:
            fp = os.path.join(BACKFILL_DIR, fn)
            h = hashlib.sha256(open(fp, "rb").read()).hexdigest()
            mf.write(f"{h}  {fp}\n")
    print("[D] per-positive availability:")
    for pos, d in per_pos_avail.items():
        print(f"  {pos}: PN-A {d['pn_a_total']} | 主选可用 {d['primary_usable']}/{d['primary_slots']}"
              f" (flag {d['primary_flagged']}, weak {d['primary_weak']}) | 备选可用 {d['backup_usable']}/{d['backup_slots']}")
    print("[C] backfill:", json.dumps(backfill_summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
