#!/usr/bin/env python3
"""P5.07：扩库候选链序列获取（RCSB GraphQL，复用 p303_fetch_knot_sequences 模式）。

输入：data/interim/p507_expansion_candidates.json（282 链）+ 已复核清单
      data/interim/p507_verification_summary.json（265 verified + 15 discordant + 1 failed）。
输出：data/raw/rcsb/2026-09-28/knot_entry_sequences_p507.json
断言：282 链中除 CIF 缺失的 2HKR_D 外全部拿到序列。
"""
import datetime
import json
import os
import sys
import time
import urllib.request

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
OUT_RAW = os.path.join(ROOT, "data/raw/rcsb/2026-09-28/knot_entry_sequences_p507.json")
CAND = os.path.join(ROOT, "data/interim/p507_expansion_candidates.json")
VERIF = os.path.join(ROOT, "data/interim/p507_verification_summary.json")
BATCH = 5


def die(m):
    print(f"[p507seq FATAL] {m}", file=sys.stderr)
    sys.exit(1)


def graphql(entries):
    alias = ", ".join(f'e{i}: entry(entry_id:"{e}"){{polymer_entities{{'
                      f'rcsb_polymer_entity_container_identifiers{{auth_asym_ids}}'
                      f'entity_poly{{pdbx_seq_one_letter_code_can}}}}}}' for i, e in enumerate(entries))
    q = "{" + alias + "}"
    req = urllib.request.Request("https://data.rcsb.org/graphql",
                                 data=json.dumps({"query": q}).encode(),
                                 headers={"Content-Type": "application/json"})
    for att in range(4):
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                resp = json.loads(r.read())
            if resp.get("errors"):
                raise RuntimeError(f"graphql errors: {str(resp['errors'])[:150]}")
            return resp
        except Exception as e:
            if att == 3:
                die(f"GraphQL 四次失败: {e}")
            time.sleep(10 * (att + 1))


def main():
    cand = json.load(open(CAND))
    chains = sorted({c for v in cand.values() for c in v})
    verif = json.load(open(VERIF))
    ok_chains = {d["chain"] for d in verif.get("discordant", [])} | {
        c for t, n in verif["verified_as_declared"].items() for c in []}
    # verified 名单重建：summary tsv 的 verified 與 discordant 並集 = 281 ok
    import csv
    rows = list(csv.DictReader(open(os.path.join(ROOT, "data/interim/p507_topoly_summary.tsv")),
                               delimiter="\t"))
    ok_chains = {r["chain"] for r in rows if not r["error"]}
    wanted = [c for c in chains if c in ok_chains]
    dropped = sorted(set(chains) - set(wanted))
    print(f"候选 {len(chains)}，取复核成功 {len(wanted)}（剔除 {dropped}）")

    by_entry = {}
    for c in wanted:
        pdb, ch = c.split("_", 1)
        by_entry.setdefault(pdb.lower(), {})[ch] = None

    entries = sorted(by_entry)
    out = {}
    for i in range(0, len(entries), BATCH):
        batch = entries[i:i + BATCH]
        data = graphql(batch)
        for j, e in enumerate(batch):
            ent = data.get("data", {}).get(f"e{j}")
            if not ent:
                print(f"  [warn] {e}: entry null")
                continue
            m = {}
            for pe in ent.get("polymer_entities") or []:
                ids = pe["rcsb_polymer_entity_container_identifiers"]
                seq = pe["entity_poly"]["pdbx_seq_one_letter_code_can"].replace("\n", "")
                for ch in ids["auth_asym_ids"]:
                    m[ch] = seq
            out[e] = m
        print(f"  {min(i + BATCH, len(entries))}/{len(entries)} entries")
        time.sleep(0.5)

    # null 条目：逐条 REST 补拉（data.rcsb.org/rest/v1/core/entry 路由退路）
    nulls = [e for e in entries if not out.get(e)]
    for k, e in enumerate(nulls):
        time.sleep(2)
        try:
            req = urllib.request.Request(
                f"https://data.rcsb.org/rest/v1/core/polymer_entity/{e}/1",
                headers={"Accept": "application/json"})
            with urllib.request.urlopen(req, timeout=60) as r:
                pe = json.loads(r.read())
            ids = pe["rcsb_polymer_entity_container_identifiers"]
            seq = pe["entity_poly"]["pdbx_seq_one_letter_code_can"].replace("\n", "")
            m = {ch: seq for ch in ids["auth_asym_ids"]}
            out[e] = m
        except Exception as ex:
            print(f"  [rest-fail] {e}: {str(ex)[:100]}")
        if (k + 1) % 20 == 0:
            print(f"  rest-retry {k+1}/{len(nulls)}")
    os.makedirs(os.path.dirname(OUT_RAW), exist_ok=True)
    json.dump(out, open(OUT_RAW, "w"), indent=1, sort_keys=True)
    missing = [c for c in wanted
               if c.split("_", 1)[1] not in out.get(c.split("_", 1)[0].lower(), {})]
    if missing:
        die(f"无序列链 {missing}")
    n = sum(len(m) for m in [v for k, v in out.items()])
    print(f"[p507seq] OK entries={len(out)} chains={len(wanted)} -> {OUT_RAW}")


if __name__ == "__main__":
    main()
