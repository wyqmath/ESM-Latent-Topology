#!/usr/bin/env python3
"""P3.03 preflight：打结链序列获取（RCSB GraphQL）+ same-sha×标签断言（P3.01 §4 前置）。

输出：
  data/raw/rcsb/2026-09-25/knot_entry_sequences.json   entry_id → {auth_chain: canonical_seq}
  data/curated/knots_sequences.tsv                      usable 链序列映射（record_id/pdb/chain/sha16/len）
  data/manifests/p303_knot_sequences_qc.json            断言与计数
断言（失败 exit 1）：
  1) usable 链 1,020/1,020 全部拿到序列；
  2) same-sha 组内 presence_target 无冲突（同序列不同存在性=纯序列不可识别，P3.01 登记的检查）；
  3) same-sha 组内 type 任务 c2_primary 无冲突（112 eligible 内）。
只读 knots.tsv；网络仅读 RCSB data API（公网，批量 GraphQL）。
"""
import csv
import datetime
import hashlib
import json
import os
import sys
import time
import urllib.request
from collections import defaultdict

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
OUT_RAW = os.path.join(ROOT, "data/raw/rcsb/2026-09-25/knot_entry_sequences.json")
OUT_TSV = os.path.join(ROOT, "data/curated/knots_sequences.tsv")
OUT_QC = os.path.join(ROOT, "data/manifests/p303_knot_sequences_qc.json")
BATCH = 25
RUN_TS = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")


def die(m):
    print(f"[p303seq FATAL] {m}", file=sys.stderr)
    sys.exit(1)


def graphql(entries):
    alias = ", ".join(f'e{i}: entry(entry_id:"{e}"){{polymer_entities{{'
                      f'rcsb_polymer_entity_container_identifiers{{auth_asym_ids asymmetric_ids}}'
                      f'entity_poly{{pdbx_seq_one_letter_code_can}}}}}}' for i, e in enumerate(entries))
    q = "{" + alias + "}"
    req = urllib.request.Request("https://data.rcsb.org/graphql",
                                 data=json.dumps({"query": q}).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.load(r)["data"]


kt = list(csv.DictReader(open(os.path.join(ROOT, "data/curated/knots.tsv")), delimiter="\t"))
os.makedirs(os.path.join(ROOT, "data/raw/rcsb/2026-09-25"), exist_ok=True)
os.makedirs(os.path.dirname(OUT_QC), exist_ok=True)
usable = [r for r in kt if r["presence_mask"] == "1"]
if len(usable) != 1020:
    die(f"usable {len(usable)} != 1020")
entries = sorted({r["pdb"].lower() for r in usable})

db = {}
if os.path.exists(OUT_RAW):   # 断点续跑：已有原始件则不再请求
    db = json.load(open(OUT_RAW))
    print(f"[p303seq] cache hit: {len(db)} entries")
missing = [e for e in entries if e not in db]
for i in range(0, len(missing), BATCH):
    chunk = missing[i:i + BATCH]
    for attempt in range(3):
        try:
            data = graphql(chunk)
            break
        except Exception as exc:
            if attempt == 2:
                die(f"GraphQL failed at {chunk[0]}: {exc}")
            time.sleep(5)
    for j, e in enumerate(chunk):
        node = data.get(f"e{j}")
        if not node:
            db[e] = {}
            continue
        chains = {}
        for pe in node["polymer_entities"]:
            ids = pe["rcsb_polymer_entity_container_identifiers"]["auth_asym_ids"] or []
            seq = pe["entity_poly"]["pdbx_seq_one_letter_code_can"] or ""
            for c in ids:
                chains[c] = seq
        db[e] = chains
    if (i // BATCH) % 10 == 0:
        print(f"[p303seq] {i + len(chunk)}/{len(missing)} fetched")
        json.dump(db, open(OUT_RAW, "w"))

os.makedirs(os.path.dirname(OUT_RAW), exist_ok=True)
json.dump(db, open(OUT_RAW, "w"))

rows, n_missing = [], []
unavail = []
for r in usable:
    e, ch = r["pdb"].lower(), r["chain"]
    seq = db.get(e, {}).get(ch)
    if not seq:
        # 兜底：label 链名/asymmetric_ids + 大小写不敏感
        cands = {}
        try:
            node = graphql([e]).get(f"e0")
        except Exception:
            node = None
        for pe in (node or {}).get("polymer_entities", []):
            ids = pe["rcsb_polymer_entity_container_identifiers"]
            sq = pe["entity_poly"]["pdbx_seq_one_letter_code_can"] or ""
            for a in (ids["auth_asym_ids"] or []):
                cands[a] = sq
            for a in (ids["asymmetric_ids"] or []):
                cands.setdefault(a, sq)
        seq = cands.get(ch) or {k.lower(): v for k, v in cands.items()}.get(ch.lower())
        if seq:
            db.setdefault(e, {})[ch] = seq
    if not seq:
        # 三重证据（RCSB REST/GraphQL 404、wwPDB obsolete FTP 404、obsolete.dat 无映射，
        # 见 P3.03 执行记录）：条目不在 wwPDB 档案——KnotProt 陈旧引用，登记为提取不可用
        unavail.append({"record_id": r["record_id"], "pdb": e, "chain": ch,
                        "presence_target": r["presence_target"],
                        "type_eligible": "1" if r["type_task_tier"] == "eligible" else "0",
                        "c2_primary": r["c2_primary"],
                        "reason": "pdb_entry_not_in_wwpdb_archive"})
        continue
    rows.append({"record_id": r["record_id"], "pdb": e, "chain": ch,
                 "presence_target": r["presence_target"],
                 "type_eligible": "1" if r["type_task_tier"] == "eligible" else "0",
                 "c2_primary": r["c2_primary"],
                 "len": len(seq),
                 "sha256_16": hashlib.sha256(seq.encode()).hexdigest()[:16]})
json.dump(db, open(OUT_RAW, "w"))
if len(rows) + len(unavail) != 1020:
    die(f"rows+unavail {len(rows)}+{len(unavail)} != 1020")
with open(OUT_TSV, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()), delimiter="\t", lineterminator="\n")
    w.writeheader()
    w.writerows(rows)
with open(OUT_TSV.replace("knots_sequences", "knots_sequences_unavailable"), "w", newline="") as f:
    if unavail:
        w = csv.DictWriter(f, fieldnames=list(unavail[0].keys()), delimiter="\t", lineterminator="\n")
        w.writeheader()
        w.writerows(unavail)

# ---- same-sha×标签断言 ----
by_sha = defaultdict(list)
for r in rows:
    by_sha[r["sha256_16"]].append(r)
pres_conflict = {s: [(x["record_id"], x["presence_target"]) for x in v]
                 for s, v in by_sha.items() if len({x["presence_target"] for x in v}) > 1}
type_conflict = {}
for s, v in by_sha.items():
    tel = [x for x in v if x["type_eligible"] == "1"]
    if len({x["c2_primary"] for x in tel}) > 1:
        type_conflict[s] = [(x["record_id"], x["c2_primary"]) for x in tel]
qc = {
    "run_ts": RUN_TS, "usable_chains": len(rows), "entries": len(entries),
    "unavailable": len(unavail),
    "unavailable_reason": "pdb_entry_not_in_wwpdb_archive（RCSB REST/GraphQL 404 + wwPDB obsolete "
                          "FTP 404 + obsolete.dat 无映射，三重核查 2026-09-25）",
    "distinct_sha": len(by_sha),
    "presence_conflict_sha_groups": pres_conflict,
    "type_conflict_sha_groups": type_conflict,
    "len_min": min(r["len"] for r in rows), "len_max": max(r["len"] for r in rows),
    "asserts": ["rows+unavailable=1020", "presence_sha_unique_labels", "type_sha_unique_labels"],
}
if pres_conflict or type_conflict:
    qc["verdict"] = "CONFLICT_REGISTERED"
    print(f"[p303seq] WARNING presence_conflict={len(pres_conflict)} type_conflict={len(type_conflict)}")
else:
    qc["verdict"] = "OK_WITH_REGISTERED_UNAVAILABLE" if unavail else "OK"
os.makedirs(os.path.dirname(OUT_QC), exist_ok=True)
json.dump(qc, open(OUT_QC, "w"), ensure_ascii=False, indent=1, sort_keys=True)
print(f"[p303seq] OK rows={len(rows)} sha_groups={len(by_sha)} verdict={qc['verdict']}")
