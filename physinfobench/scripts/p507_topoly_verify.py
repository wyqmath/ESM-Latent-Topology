#!/usr/bin/env python3
"""P5.07 步骤：扩库候选链的 Topoly Alexander 复核（集群 plm_bench env）。

协议（沿 2026-09-13 轮次冻结语义）：
  - 主判定 = topoly.alexander(xyz, closure=2, tries=200)（c2t200，knotnet-matched）；
  - 敏感性 = topoly.alexander(xyz, closure=1, tries=200)（c1t200，KnotProt 同款质心闭合）；
  - xyz=Cα 轨迹（auth 链匹配、altloc A/空、仅 ATOM、auth_seq_id 去重）；
  - 断点续跑：每链结果文件存在即跳过。
输入：data/interim/p507_expansion_candidates.json + p507_cif/*.cif.gz（集群侧）。
输出：p507_topoly_results/{chain}.json + 汇总 p507_topoly_summary.tsv。
"""
import ast
import csv
import gzip
import json
import os
import re
import sys
import time

import topoly

ROOT = "/lenovofs1/home/jyma/PLM_benchmark/p303_resplit/proj"
CIFD = f"{ROOT}/data/raw/rcsb/2026-09-28/p507_cif"
OUTD = f"{ROOT}/data/interim/p507_topoly_results"
CAND = f"{ROOT}/data/interim/p507_expansion_candidates.json"
PROTOCOLS = {"c2t200": dict(closure=2, tries=200), "c1t200": dict(closure=1, tries=200)}


def die(m):
    print(f"[p507 FATAL] {m}", file=sys.stderr)
    sys.exit(1)


def chain_ca_xyz(cif_path, auth_chain, out_xyz):
    """mmCIF atom_site → Cα xyz（auth 链、altloc A/空、ATOM、auth_seq_id 去重）。"""
    coords = {}
    seen = set()
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
                    in_atom = False
                    continue
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
        return len(coords)
    with open(out_xyz, "w") as f:
        for seq in sorted(coords, key=lambda s: (len(s), s)):
            x, y, z = coords[seq]
            f.write(f"{x:.3f} {y:.3f} {z:.3f}\n")
    return len(coords)


def call_type(xyz, kw):
    """topoly 返回三形态：c2=概率 dict；c1=类型 str（'3_1'）或裸 int（0=无结）。"""
    raw = None
    for attempt in range(4):  # topoly 内部对 '0_1' 有随机 literal_eval 缺陷，重试即可过
        try:
            out = topoly.alexander(xyz, cuda=False, **kw)
            raw = str(out).strip()
            break
        except (ValueError, SyntaxError) as e:
            if attempt == 3:
                raise
            print(f"  retry {attempt+1}: {e}", flush=True)
    raw = raw.strip()
    try:
        d = ast.literal_eval(raw)
    except (ValueError, SyntaxError):  # '0_1' 被 ast 视为前导零数字字面量→SyntaxError
        m = re.fullmatch(r"(\d+)_(\d+)", raw)  # '0_1'/'3_1' 等：本就是结型记号
        if m:
            return f"{m.group(1)}_{m.group(2)}", 1.0, raw
        if raw.isdigit():  # '01' 等两位编码：走码表
            d = raw
        else:
            raise
    if isinstance(d, dict):
        best, p = "0_1", 0.0
        for k, v in d.items():
            if isinstance(v, (int, float)) and v > p:
                best, p = str(k), float(v)
        return best, p, str(out)
    if isinstance(d, int):
        d = str(d)  # c1 的两位数编码（31=3_1、52=5_2、0/01=无结）走同一映射
    if isinstance(d, str) and len(d) <= 3 and d.isdigit():
        code = d.zfill(2)
        return f"{code[0]}_{code[1]}", 1.0, str(out)
    return str(d), 1.0, str(out)


def main():
    cand = json.load(open(CAND))
    os.makedirs(OUTD, exist_ok=True)
    summary = []
    n = 0
    for kt, chains in sorted(cand.items()):
        for chain in chains:
            n += 1
            out_json = f"{OUTD}/{chain}.json"
            if os.path.exists(out_json):
                prev = json.load(open(out_json))
                if not prev.get("error"):
                    summary.append(prev)
                    continue
                os.remove(out_json)  # 带错误的历史结果：删除重跑
            pdb, auth = chain.split("_", 1)
            cif = f"{CIFD}/{pdb.lower()}.cif.gz"
            xyz = f"{OUTD}/{chain}.xyz"
            if os.path.exists(xyz):
                os.remove(xyz)  # 陈旧 xyz 一并重建（历史缺陷：格式曾错误）
            rec = {"chain": chain, "knotprot_type": kt, "n_ca": 0,
                   "c2t200": "", "c2t200_p": "", "c1t200": "", "c1t200_p": "",
                   "error": ""}
            try:
                if not os.path.exists(xyz):
                    nca = chain_ca_xyz(cif, auth, xyz)
                    rec["n_ca"] = nca
                    if nca < 30:
                        rec["error"] = "too_few_ca"
                        json.dump(rec, open(out_json, "w"), indent=1)
                        summary.append(rec)
                        continue
                for name, kw in PROTOCOLS.items():
                    best, p, raw = call_type(xyz, kw)
                    rec[name] = best
                    rec[f"{name}_p"] = round(p, 4)
                os.remove(xyz)  # 复核完即删，省盘
                if n == 1 and not rec["c2t200"]:
                    die("fail-fast: 首链 topoly 无输出（格式/环境问题），终止作业")
            except Exception as e:
                import traceback
                rec["error"] = traceback.format_exc()[-600:]
            json.dump(rec, open(out_json, "w"), indent=1)
            summary.append(rec)
            print(f"[{n}] {chain} K={kt} c2={rec['c2t200']} c1={rec['c1t200']} "
                  f"n={rec['n_ca']} err={rec['error'][:60]}", flush=True)
            time.sleep(0.05)
    with open(f"{OUTD}/p507_topoly_summary.tsv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["chain", "knotprot_type", "n_ca", "c2t200",
                                          "c2t200_p", "c1t200", "c1t200_p", "error"],
                           delimiter="\t", lineterminator="\n")
        w.writeheader()
        w.writerows(summary)
    print(f"[p507] DONE {len(summary)} chains")


if __name__ == "__main__":
    main()
