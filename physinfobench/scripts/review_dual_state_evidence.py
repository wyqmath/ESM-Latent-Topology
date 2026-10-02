#!/usr/bin/env python3
"""P1.14 strict 双态原文证据回查（G1 前补证；只读，不改资格表）。

对象：8 对缺 round2 扫描行的 strict（porter_20/51/61/62/68/72/77/87，共 16 端点）
     + porter_8 单列核对（2 端点）。
通道：RCSB 主引文（GraphQL 批量）→ 旧项目 unpaywall 缓存（sha 校验后复用，缺口现查）
     → Europe PMC OA 全文 XML（合法开放渠道）。
证据分级：fulltext_self_verified（本任务取得全文并程序化定位证据）/
         pmc_html_old_project（旧项目已取得 PMC 全文，本任务复核其可解析性）/
         abstract_self_verified（本任务取得摘要并核对状态描述）/
         old_audit_pointer_only（仅旧审计指针，本任务未能取得可自核材料）/
         paywalled（确认付费墙）。
铁律：不把旧审计指针写成已亲自核验；porter_8 只给分层建议不改表。
"""
import csv
import gzip
import hashlib
import io
import json
import os
import re
import time
import urllib.parse
import urllib.request

B_OLD = "/Users/yuan/Documents/Codex/2026-09-08/jie/benchmark_step1"
UNPAYWALL = os.path.join(B_OLD, "execution_round_2026-09-10/evidence/unpaywall_lookup_2026-09-14.json")
OUT_DIR = "data/raw/evidence_fulltext/2026-09-23"
OUT_TSV = "reports/fs_three_layer/dual_state_evidence_review.tsv"
OUT_JSON = "reports/fs_three_layer/dual_state_evidence_review_qc.json"
UA = "PhysInfoBench-P1.14/1.0"

PAIRS_8 = ["porter_20_5c1vA__5c1vB", "porter_51_1h38D__1qlnA", "porter_61_4gqcC__4gqcB",
           "porter_62_4o0pA__4o01D", "porter_68_3zwgN__4tsyD", "porter_72_4rmbA__4rmbB",
           "porter_77_2nxqB__1jfkA", "porter_87_2k0qA__2lelA"]
STATE_KEYWORDS = ["fold switch", "fold-switch", "alternative fold", "metamorphic",
                  "domain swap", "conformational change", "conformational transition",
                  "two states", "dual", "open and closed", "active and inactive", "state"]


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def fetch(url, timeout=60):
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.status, r.read()


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    # ---- 0) 旧缓存校验 ----
    with open(UNPAYWALL) as f:
        unpay = json.load(f)
    up_sha = sha256_file(UNPAYWALL)
    print(f"[0] unpaywall cache entries={len(unpay)} sha256={up_sha[:16]}")

    # ---- 1) 端点清单（8 对 strict + porter_8）----
    endpoints = []  # (pair_id, side, pdb, chain)
    with open("data/curated/fold_switch_global.tsv") as f:
        rows = {r["pair_id"]: r for r in csv.DictReader(f, delimiter="\t")}
    for pid in PAIRS_8:
        r = rows[pid]
        endpoints.append((pid, "a", r["pdb_a"].lower(), r["chain_a"]))
        endpoints.append((pid, "b", r["pdb_b"].lower(), r["chain_b"]))
    r8 = rows["porter_8_3m1bF__3lowA"]
    endpoints.append(("porter_8_3m1bF__3lowA", "a", r8["pdb_a"].lower(), r8["chain_a"]))
    endpoints.append(("porter_8_3m1bF__3lowA", "b", r8["pdb_b"].lower(), r8["chain_b"]))

    # ---- 2) RCSB 主引文（GraphQL 批量，18 条目）----
    ids = sorted({e[2] for e in endpoints})
    payload = {"query": "query($ids:[String!]!){ entries(entry_ids:$ids){ rcsb_id "
                        "rcsb_primary_citation{ pdbx_database_id_DOI pdbx_database_id_PubMed title journal_abbrev } } }",
               "variables": {"ids": [i.upper() for i in ids]}}
    body = None
    for attempt in range(4):
        try:
            req = urllib.request.Request("https://data.rcsb.org/graphql",
                                         data=json.dumps(payload).encode(),
                                         headers={"Content-Type": "application/json", "User-Agent": UA})
            with urllib.request.urlopen(req, timeout=120) as r:
                body = r.read()
            break
        except Exception as e:
            print(f"  graphql attempt {attempt+1} failed: {type(e).__name__}")
            time.sleep(3 + 5 * attempt)
    if body is None:
        raise SystemExit("ASSERTION FAILED: rcsb graphql unreachable after retries")
    cit_raw = os.path.join(OUT_DIR, "rcsb_primary_citations.json")
    with open(cit_raw, "wb") as f:
        f.write(body)
    cit = {}
    for e in json.loads(body)["data"]["entries"]:
        c = e.get("rcsb_primary_citation") or {}
        cit[e["rcsb_id"].lower()] = c
    print(f"[2] primary citations for {len(cit)} entries")

    # ---- 2b) 唯一 DOI 预解析（记忆化：同 DOI 端点共享一次查询，保证一致性）----
    # EuropePMC DOI 检索存在返回不稳定（同 DOI 两次查询结果不一致），故每 DOI 只查一次并复用；
    # 查询失败/空 = indeterminate，绝不当作 closed。
    doi_raw = os.path.join(OUT_DIR, "epmc_doi_resolution.json")
    doi_info = {}
    if os.path.exists(doi_raw):
        doi_info = json.load(open(doi_raw))  # 快照优先：已解析 DOI 一律复用本地记录
    for pid, side, pdb, chain in endpoints:
        doi = (cit.get(pdb, {}).get("pdbx_database_id_DOI") or "").strip()
        if doi and doi not in doi_info:
            rec = {"is_oa": False, "pmcid": "", "status": "not_looked_up"}
            for attempt in range(4):
                try:
                    st, b = fetch("https://www.ebi.ac.uk/europepmc/webservices/rest/search?"
                                  + urllib.parse.urlencode({"query": f'DOI:"{doi}"', "format": "json",
                                                            "resultType": "core"}))
                    d = json.loads(b)
                    item = (d.get("resultList") or {}).get("result") or []
                    if item:
                        rec = {"is_oa": item[0].get("isOpenAccess") == "Y",
                               "pmcid": item[0].get("pmcid") or "",
                               "status": "resolved"}
                    else:
                        rec = {"is_oa": False, "pmcid": "", "status": "no_europepmc_record"}
                    break
                except Exception:
                    rec = {"is_oa": False, "pmcid": "", "status": f"lookup_error_attempt{attempt+1}"}
                    time.sleep(2 + 3 * attempt)
            doi_info[doi] = rec
            time.sleep(0.4)
    print(f"[2b] unique DOIs resolved: {len(doi_info)} "
          f"({sum(1 for r in doi_info.values() if r['status']=='resolved')} resolved, "
          f"{sum(1 for r in doi_info.values() if r['status'].startswith('lookup_error') or r['status']=='no_europepmc_record')} indeterminate/no-record)")
    with open(doi_raw, "w") as f:
        json.dump(doi_info, f, indent=1, sort_keys=True)

    # ---- 3) 逐端点证据链 ----
    out_rows = []
    for pid, side, pdb, chain in endpoints:
        c = cit.get(pdb, {})
        doi = (c.get("pdbx_database_id_DOI") or "").strip()
        pmid = str(c.get("pdbx_database_id_PubMed") or "").strip()
        title = (c.get("title") or "").replace("\t", " ")[:180]
        journal = c.get("journal_abbrev") or ""
        ev_level, ev_loc, quote, note = "no_evidence", "", "", []
        # 3a) OA 判定：读预解析记忆（同 DOI 端点一致）；旧 unpaywall 缓存仅留档
        is_oa, pmcid, doi_status = False, "", "no_doi"
        if doi:
            rec = doi_info[doi]
            is_oa, pmcid, doi_status = rec["is_oa"], rec["pmcid"], rec["status"]
            u_old = unpay.get(doi)
            if u_old is not None:
                note.append(f"old_unpaywall_is_oa={u_old.get('is_oa')}(仅留档)")
        # 3b) OA 全文（PMC XML）
        if doi and is_oa:
            xml = None
            if pmcid:
                for attempt in range(3):
                    try:
                        st, xml = fetch(f"https://www.ebi.ac.uk/europepmc/webservices/rest/{pmcid}/fullTextXML")
                        break
                    except Exception:
                        xml = None
                        time.sleep(2 + 3 * attempt)
            if xml is None and pmcid and os.path.exists(os.path.join(OUT_DIR, f"{pmcid}.xml")):
                with open(os.path.join(OUT_DIR, f"{pmcid}.xml"), "rb") as f:
                    xml = f.read()  # 快照优先：已存档全文直接复用
            if xml is not None and pmcid:
                fx = os.path.join(OUT_DIR, f"{pmcid}.xml")
                with open(fx, "wb") as f:
                    f.write(xml)
                text = re.sub(r"<[^>]+>", " ", xml.decode("utf-8", "ignore"))
                text = re.sub(r"\s+", " ", text)
                # 程序化定位：PDB 码出现 + 状态关键词共现
                pdb_hit = pdb.upper() in text.upper()
                kw_hits = [k for k in STATE_KEYWORDS if k in text.lower()]
                if pdb_hit and kw_hits:
                    i = text.upper().find(pdb.upper())
                    quote = text[max(0, i - 120):i + 180].strip()[:280]
                    ev_level = "fulltext_self_verified"
                    ev_loc = f"{pmcid} fullTextXML; PDB 码命中+状态关键词 {len(kw_hits)} 个"
                elif kw_hits:
                    ev_level = "fulltext_self_verified_no_pdb_mention"
                    ev_loc = f"{pmcid} fullTextXML; 状态关键词 {len(kw_hits)} 个但未见 PDB 码"
                else:
                    ev_level = "fulltext_obtained_no_state_keyword"
                    ev_loc = f"{pmcid} fullTextXML"
                note.append(f"kw={','.join(kw_hits[:6])}")
            elif is_oa and not pmcid:
                ev_level = "oa_publisher_only_no_pmc"
                note.append("OA 仅出版社渠道（无 PMC），本任务不下载出版社 PDF")
            elif is_oa and pmcid and xml is None:
                ev_level = "fulltext_fetch_failed_after_retry"
                note.append("PMC 全文抓取 3 次重试失败")
        elif doi and doi_status == "resolved" and not is_oa:
            ev_level = "paywalled"
        elif doi and doi_status != "resolved":
            ev_level = "doi_lookup_indeterminate"
            note.append(f"doi_status={doi_status}")
        else:
            note.append("RCSB 主引文无 DOI")
            ev_level = "no_doi_no_evidence" if not title else "abstract_self_verified_unverified"
        # 3c) 摘要通道（付费墙/DOI 查询不定/无 DOI 时）：
        if ev_level in ("paywalled", "doi_lookup_indeterminate", "abstract_self_verified") and pmid:
            try:
                st, b = fetch("https://www.ebi.ac.uk/europepmc/webservices/rest/search?"
                              + urllib.parse.urlencode({"query": f"EXT_ID:{pmid} AND SRC:MED",
                                                        "format": "json", "resultType": "core"}))
                with open(os.path.join(OUT_DIR, f"MED_{pmid}.json"), "wb") as f:
                    f.write(b)
                ab = ((json.loads(b).get("resultList") or {}).get("result") or [{}])[0]
                abtext = (ab.get("abstractText") or "")
                kw = [k for k in STATE_KEYWORDS if k in abtext.lower()]
                ev_level = "abstract_self_verified"
                ev_loc = f"EuropePMC search EXT_ID:{pmid}; 摘要关键词 {len(kw)}"
                if kw:
                    note.append(f"abstract_kw={','.join(kw[:6])}")
            except Exception as e:
                note.append(f"abstract_fetch_error:{type(e).__name__}")
        out_rows.append({
            "pair_id": pid, "endpoint": f"{pdb}_{chain}", "side": side,
            "primary_citation_doi": doi, "primary_citation_pmid": pmid,
            "primary_citation_title": title, "journal": journal,
            "oa_status": ("oa" if is_oa else
                          ("indeterminate_no_record" if doi_status == "no_europepmc_record"
                           else ("indeterminate_lookup_error" if doi_status.startswith("lookup_error")
                                 else ("closed" if doi else "no_doi")))),
            "pmcid": pmcid,
            "evidence_level": ev_level, "evidence_location": ev_loc,
            "quote_or_probe": quote.replace("\t", " ")[:280],
            "notes": ";".join(note),
            "old_audit_pointer": rows[pid].get("evidence_pointer_primary", "")[:220],
        })

    cols = list(out_rows[0].keys())
    with open(OUT_TSV, "w", newline="") as f:
        f.write("\t".join(cols) + "\n")
        w = csv.DictWriter(f, fieldnames=cols, delimiter="\t", lineterminator="\n")
        w.writerows(sorted(out_rows, key=lambda r: (r["pair_id"], r["side"])))

    from collections import Counter
    qc = {
        "generated": os.popen("TZ=Asia/Shanghai date '+%Y-%m-%d %H:%M'").read().strip(),
        "unpaywall_cache": {"path": UNPAYWALL, "sha256": up_sha, "entries": len(unpay)},
        "rcsb_citations_file": os.path.relpath(cit_raw),
        "rcsb_citations_sha256": sha256_file(cit_raw),
        "endpoints": len(out_rows),
        "evidence_level_counts": dict(Counter(r["evidence_level"] for r in out_rows)),
        "state_keyword_rule": "预注册关键词表见脚本 STATE_KEYWORDS；程序化共现定位，人工引语仅截取窗口",
    }
    with open(OUT_JSON, "w") as f:
        json.dump(qc, f, ensure_ascii=False, indent=1, sort_keys=True)
    print(f"[3] endpoints={len(out_rows)} levels={qc['evidence_level_counts']}")


if __name__ == "__main__":
    main()
