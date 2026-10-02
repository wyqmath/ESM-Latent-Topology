#!/usr/bin/env python3
"""P5.08 阶段 2：X-ray 未观测残基对比集构建（一手数据法，源冻结）。

方法（外部源可达性降级后的预注册方案，decisions.md 2026-09-30 条目）：
  - 检索：X-RAY ≤2.5Å 全部 entry（163,493），seed=2026 随机抽 2,200 个 entry；
  - 标签：per label_asym 链——poly_seq_scheme 全实体残基中，atom_site 有 CA 观测=有序(0)，
    无观测(pdbx_unobserved/缺失坐标)=无序(1)；插入码残基丢弃并计数；
  - 候选门槛（可行性统计后冻结）：链长 80–1000、两类各 ≥15% 且 ≥10 残基（截断 1022 内）；
  - 排除（阶段 3）：UniProt acc 与项目 DisProt 3337 重叠；mmseqs ≥30% 同源；sha 重复。
输出：data/interim/p508_candidate_table.tsv + feasibility.json（断点续跑）。
"""
import csv
import gzip
import json
import os
import sys
import time
import urllib.request
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

import gemmi

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RAW = os.path.join(ROOT, "data/raw/rcsb/2026-09-30/p508_cif")
SAMPLED = os.path.join(ROOT, "data/raw/rcsb/2026-09-30/p508_sampled_entries.json")
OUT_TSV = os.path.join(ROOT, "data/interim/p508_candidate_table.tsv")
OUT_FEAS = os.path.join(ROOT, "data/interim/p508_feasibility.json")

THREE_TO_ONE = {}
for one, threes in {
    'A': ('ALA',), 'R': ('ARG',), 'N': ('ASN',), 'D': ('ASP',), 'C': ('CYS',),
    'Q': ('GLN',), 'E': ('GLU',), 'G': ('GLY',), 'H': ('HIS',), 'I': ('ILE',),
    'L': ('LEU',), 'K': ('LYS',), 'M': ('MET',), 'F': ('PHE',), 'P': ('PRO',),
    'S': ('SER',), 'T': ('THR',), 'W': ('TRP',), 'Y': ('TYR',), 'V': ('VAL',),
}.items():
    for t in threes:
        THREE_TO_ONE[t] = one


def die(m):
    print(f"[p508 FATAL] {m}", file=sys.stderr)
    sys.exit(1)


def fetch_cif(entry):
    out = os.path.join(RAW, f"{entry.lower()}.cif.gz")
    if os.path.exists(out) and os.path.getsize(out) > 1000:
        return entry, True
    for att in range(3):
        try:
            urllib.request.urlretrieve(
                f"https://files.rcsb.org/download/{entry}.cif.gz", out)
            return entry, os.path.getsize(out) > 1000
        except Exception:
            time.sleep(2 * (att + 1))
    return entry, False


def parse_entry(entry):
    """返回该 entry 全部蛋白链的 [(asym, seq, labels, n_ins_dropped, unp_acc)]。"""
    path = os.path.join(RAW, f"{entry.lower()}.cif.gz")
    tmp = path[:-3]  # gemmi 读未压缩文件；解压缓存
    if not os.path.exists(tmp):
        with gzip.open(path, "rb") as fin, open(tmp, "wb") as fout:
            fout.write(fin.read())
    doc = gemmi.cif.read(tmp)
    block = doc.sole_block()

    # entity → 类型与长度
    ent_poly = {}
    for row in block.find("_entity_poly.", ["entity_id", "type", "pdbx_seq_one_letter_code_can"]):
        eid = row[0]
        if "polypeptide" in row[1]:
            seq = row[2].replace("\n", "").replace(" ", "")
            ent_poly[eid] = seq
    # entity → asym 链
    asym_of = defaultdict(list)
    for row in block.find("_struct_asym.", ["id", "entity_id"]):
        if row[1] in ent_poly:
            asym_of[row[1]].append(row[0])
    # 观测集合：(asym, seq_id)
    observed = set()
    for row in block.find("_atom_site.", ["label_asym_id", "label_seq_id", "label_atom_id",
                                          "pdbx_PDB_model_num"]):
        if row[2] == "CA" and row[3] in ("1", "0"):
            if row[1] not in (".", "?") and row[0] not in (".", "?"):
                observed.add((row[0], row[1]))
    # poly_seq_scheme：asym 全残基（含未观测）
    rows_by_asym = defaultdict(list)
    for row in block.find("_pdbx_poly_seq_scheme.",
                          ["asym_id", "entity_id", "seq_id", "mon_id", "pdb_seq_num", "auth_seq_num", "hetero"]):
        rows_by_asym[row[0]].append(row)
    # UniProt acc（struct_ref db_name=UNP）
    unp = {}
    for row in block.find("_struct_ref.", ["entity_id", "db_name", "db_code"]):
        if row[1] == "UNP":
            unp[row[0]] = row[2]

    out = []
    for eid, seq_can in ent_poly.items():
        if not (80 <= len(seq_can) <= 1000):
            continue
        for asym in asym_of.get(eid, []):
            mons = rows_by_asym.get(asym, [])
            if not mons:
                continue
            letters, labels, n_ins = [], [], 0
            for r in mons:
                if r[6] == "y":  # hetero=y 为非聚合修饰位，丢弃计数
                    n_ins += 1
                    continue
                letter = THREE_TO_ONE.get(r[3])
                if letter is None:
                    n_ins += 1
                    continue
                letters.append(letter)
                labels.append(0 if (asym, r[2]) in observed else 1)
            if len(letters) < 80:
                continue
            out.append((asym, "".join(letters), labels, n_ins, unp.get(eid, "")))
    return out


def main():
    os.makedirs(RAW, exist_ok=True)
    meta = json.load(open(SAMPLED))
    entries = meta["entries"]
    print(f"[p508] 抽样 entry {len(entries)}（seed {meta['seed']}，总体 {meta['total_count']}）")
    with ThreadPoolExecutor(12) as ex:
        results = list(ex.map(fetch_cif, entries))
    ok = [e for e, s in results if s]
    failed = [e for e, s in results if not s]
    print(f"[p508] CIF 获取 {len(ok)}/{len(entries)}（失败 {len(failed)}）", flush=True)

    rows = []
    stats = defaultdict(int)
    for i, entry in enumerate(ok):
        try:
            chains = parse_entry(entry)
        except Exception as e:
            stats["parse_error"] += 1
            continue
        best = None
        for asym, seq, labels, n_ins, unp in chains:
            trunc = min(len(seq), 1022)
            lab = labels[:trunc]
            n_dis = sum(lab)
            n_ord = len(lab) - n_dis
            two_class = (n_dis >= 10 and n_ord >= 10
                         and n_dis >= 0.15 * len(lab) and n_ord >= 0.15 * len(lab))
            rec = {"entry": entry, "asym": asym, "length": len(seq), "n_disorder": n_dis,
                   "n_order": n_ord, "dis_frac": round(n_dis / len(lab), 4),
                   "n_ins_dropped": n_ins, "uniprot": unp, "two_class": two_class}
            rows.append(rec)
            if two_class and (best is None or rec["n_disorder"] > best["n_disorder"]):
                best = rec
        if (i + 1) % 200 == 0:
            print(f"  parsed {i+1}/{len(ok)}", flush=True)
    with open(OUT_TSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()), delimiter="\t", lineterminator="\n")
        w.writeheader()
        w.writerows(rows)
    n_entries_two = len({r["entry"] for r in rows if r["two_class"]})
    feas = {"entries_fetched": len(ok), "entries_failed": len(failed),
            "chains_total": len(rows), "chains_two_class": sum(1 for r in rows if r["two_class"]),
            "entries_two_class": n_entries_two, "stats": dict(stats)}
    json.dump(feas, open(OUT_FEAS, "w"), indent=1)
    print(json.dumps(feas, ensure_ascii=False))


if __name__ == "__main__":
    main()
