#!/usr/bin/env python3
"""P5.07 修复（本地）：Topoly 补算——2HKR_D 首次复核 + 低支持链 c2t800 敏感性。

协议（修复规则，decisions.md 2026-09-30 范围裁决附带）：
  - 支持率 = c2t200 众数类型概率（已有 282 链数据）；
  - 支持<0.70 或闭合敏感（c1≠c2）→ c2t800 敏感性复算；
  - 判定：c2t800 众数=c2t200 众数且支持≥0.70 → 类型冻结；
    否则 → low_confidence（剔除出类型宇宙，保守）；
  - 16 条声明不一致链 → 不复算，进裁决表（排除，理由=subchain_conflict）。
输出：data/interim/p507_sensitivity/{chain}.json + summary.tsv
用法：python scripts/p507_local_topoly.py [--only 2HKR_D]
"""
import ast
import csv
import gzip
import json
import os
import re
import sys
import time
from multiprocessing.pool import ThreadPool

import topoly

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
CIFD = f"{ROOT}/data/raw/rcsb/2026-09-28/p507_cif"
OUTD = f"{ROOT}/data/interim/p507_sensitivity"
SUMMARY = f"{ROOT}/data/interim/p507_topoly_summary.tsv"


def die(m):
    print(f"[p507L FATAL] {m}", file=sys.stderr)
    sys.exit(1)


def chain_ca_xyz(cif_path, auth_chain, out_xyz):
    coords, seen = {}, set()
    op = gzip.open if cif_path.endswith(".gz") else open
    with op(cif_path, "rt", encoding="utf-8", errors="ignore") as f:
        in_atom = False
        cols = {}
        for line in f:
            if line.startswith("_atom_site."):
                in_atom = True
                cols[line.strip().split(".")[1]] = len(cols)
                continue
            if in_atom:
                if line.startswith("#"):
                    break
                parts = line.split()
                if len(parts) < len(cols):
                    continue
                if parts[cols["group_PDB"]] != "ATOM":
                    continue
                if parts[cols["label_atom_id"]] != "CA":
                    continue
                if parts[cols["auth_asym_id"]].strip() != auth_chain:
                    continue
                alt = parts[cols["label_alt_id"]].strip()
                if alt not in (".", "?", "A"):
                    continue
                seq = parts[cols["auth_seq_id"]]
                if seq in seen:
                    continue
                seen.add(seq)
                coords[seq] = (float(parts[cols["Cartn_x"]]),
                               float(parts[cols["Cartn_y"]]),
                               float(parts[cols["Cartn_z"]]))
    if len(coords) < 30:
        return 0
    with open(out_xyz, "w") as f:
        for seq in sorted(coords, key=lambda s: (len(s), s)):
            x, y, z = coords[seq]
            f.write(f"{x:.3f} {y:.3f} {z:.3f}\n")
    return len(coords)


def parse_topoly(raw):
    """返回 (type, prob)。形态：概率 dict / '3_1' 字符串 / 裸 int / '01' 编码。"""
    raw = raw.strip()
    try:
        d = ast.literal_eval(raw)
    except (ValueError, SyntaxError):
        m = re.fullmatch(r"(\d+)_(\d+)", raw)
        if m:
            return f"{m.group(1)}_{m.group(2)}", 1.0
        if raw.isdigit():
            code = raw.zfill(2)
            return f"{code[0]}_{code[1]}", 1.0
        raise
    if isinstance(d, dict):
        best, p = "0_1", 0.0
        for k, v in d.items():
            if isinstance(v, (int, float)) and v > p:
                best, p = str(k), float(v)
        return best, p
    if isinstance(d, int):
        code = str(d).zfill(2)
        return f"{code[0]}_{code[1]}", 1.0
    return str(d), 1.0


def call(xyz, kw):
    for attempt in range(4):
        try:
            return parse_topoly(str(topoly.alexander(xyz, cuda=False, **kw)))
        except (ValueError, SyntaxError):
            if attempt == 3:
                raise
            time.sleep(0.2)


def work(job):
    chain, protocol = job
    out_json = f"{OUTD}/{chain}.{protocol}.json"
    if os.path.exists(out_json):
        return json.load(open(out_json))
    pdb, auth = chain.split("_", 1)
    cif = f"{CIFD}/{pdb.lower()}.cif.gz"
    if not os.path.exists(cif):
        cif = f"{CIFD}/{pdb.lower()}.cif"
    xyz = f"{OUTD}/{chain}.xyz"
    rec = {"chain": chain, "protocol": protocol, "n_ca": 0, "type": "", "support": "", "error": ""}
    try:
        if not os.path.exists(xyz):
            n = chain_ca_xyz(cif, auth, xyz)
            rec["n_ca"] = n
            if n < 30:
                rec["error"] = "too_few_ca"
                json.dump(rec, open(out_json, "w"), indent=1)
                return rec
        else:
            rec["n_ca"] = sum(1 for _ in open(xyz))
        kw = {"closure": 2, "tries": 800} if protocol == "c2t800" else {"closure": 2, "tries": 200}
        if protocol == "c1t200":
            kw = {"closure": 1, "tries": 200}
        t, p = call(xyz, kw)
        rec["type"], rec["support"] = t, round(p, 4)
        os.remove(xyz)
    except Exception as e:
        import traceback
        rec["error"] = traceback.format_exc()[-300:]
    json.dump(rec, open(out_json, "w"), indent=1)
    return rec


def main():
    os.makedirs(OUTD, exist_ok=True)
    rows = list(csv.DictReader(open(SUMMARY), delimiter="\t"))

    def norm(v):
        return v[4][0] + "_" + v[4][1] if v.startswith("INT_") else v

    jobs = []
    # 1) 2HKR_D：首次复核（c2t200 + c1t200）
    if not os.path.exists(f"{OUTD}/2HKR_D.c2t200.json"):
        jobs += [("2HKR_D", "c2t200"), ("2HKR_D", "c1t200")]
    # 2) 低支持（<0.70）且非不一致链 → c2t800
    for r in rows:
        if r["error"]:
            continue
        if norm(r["c2t200"]) != r["knotprot_type"]:
            continue  # 不一致链走裁决表，不复算
        if float(r["c2t200_p"]) < 0.70:
            jobs.append((r["chain"], "c2t800"))
    print(f"[p507L] 待算 {len(jobs)} 项（本地 {os.cpu_count()} 核）", flush=True)
    with ThreadPool(min(8, os.cpu_count() or 4)) as pool:
        results = pool.map(work, jobs)
    errs = [r for r in results if r["error"]]
    print(f"[p507L] DONE {len(results)} 项，失败 {len(errs)}")
    for r in errs[:5]:
        print("  ERR", r["chain"], r["protocol"], r["error"][-120:])


if __name__ == "__main__":
    main()
