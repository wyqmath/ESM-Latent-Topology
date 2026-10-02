#!/usr/bin/env python3
"""并发下载 mmCIF 并抽取单链 PDB（仅 CA，ATOM 记录，auth 链匹配，altloc A/空）。"""
import csv, gzip, os, sys, urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed

BASE = os.path.dirname(os.path.abspath(__file__))
rows = list(csv.DictReader(open(os.path.join(BASE, "knot_chain_list.tsv")), delimiter="\t"))
os.makedirs(os.path.join(BASE, "cif"), exist_ok=True)
os.makedirs(os.path.join(BASE, "singe_pdb"), exist_ok=True)


def parse_and_write(gz_path, chain, out):
    headers, rows_out, in_loop = [], [], False
    with gzip.open(gz_path, "rt") as f:
        for line in f:
            s = line.strip()
            if s.startswith("loop_"):
                in_loop, headers = False, []
                continue
            if s.startswith("_atom_site."):
                in_loop = True
                headers.append(s.split()[0])
                continue
            if in_loop and s and not s.startswith(("_", "loop_", "#")):
                parts = s.split()
                if len(parts) < len(headers) or "_atom_site.label_atom_id" not in headers:
                    continue
                d = dict(zip(headers, parts))
                if (d.get("_atom_site.auth_asym_id", "") == chain
                        and d.get("_atom_site.group_PDB", "") == "ATOM"
                        and d.get("_atom_site.label_atom_id", "") == "CA"
                        and d.get("_atom_site.label_alt_id", ".") in (".", "A")):
                    rows_out.append((d.get("_atom_site.auth_seq_id", ""),
                                     d.get("_atom_site.label_comp_id", ""),
                                     d.get("_atom_site.Cartn_x", ""),
                                     d.get("_atom_site.Cartn_y", ""),
                                     d.get("_atom_site.Cartn_z", "")))
    seen, dedup = set(), []
    for r in rows_out:
        if r[0] not in seen:
            seen.add(r[0])
            dedup.append(r)
    rows_out = dedup
    if not rows_out:
        return False
    with open(out, "w") as f:
        f.write("HEADER single-chain extract\n")
        for i, (seq, comp, x, y, z) in enumerate(rows_out, 1):
            f.write(f"ATOM  {i:5d}  CA  {comp:>3s} {'A':>1s}{seq:>4s}    "
                    f"{float(x):8.3f}{float(y):8.3f}{float(z):8.3f}  1.00  0.00           C\n")
        f.write("END\n")
    return True


def work(rec):
    pdb, chain = rec["pdb"], rec["chain"]
    out = os.path.join(BASE, "singe_pdb", f"{pdb}_{chain}.pdb")
    if os.path.exists(out) and os.path.getsize(out) > 0:
        return (pdb, chain, "cached")
    gz = os.path.join(BASE, "cif", f"{pdb}.cif.gz")
    try:
        if not os.path.exists(gz) or os.path.getsize(gz) == 0:
            urllib.request.urlretrieve(f"https://files.rcsb.org/download/{pdb.upper()}.cif.gz", gz)
        ok = parse_and_write(gz, chain, out)
        return (pdb, chain, "ok" if ok else "no_chain_atoms")
    except Exception as exc:
        return (pdb, chain, f"fail:{type(exc).__name__}")


with ThreadPoolExecutor(max_workers=8) as ex:
    futs = [ex.submit(work, r) for r in rows]
    stats = {}
    for fu in as_completed(futs):
        pdb, chain, st = fu.result()
        stats[st] = stats.get(st, 0) + 1
        if st not in ("ok", "cached"):
            print(f"{st} {pdb} {chain}", file=sys.stderr)
print("stats:", stats)
ok = stats.get("ok", 0) + stats.get("cached", 0)
print(f"singe_pdb count: {len(os.listdir(os.path.join(BASE, 'singe_pdb')))}")
if ok < 1000:
    sys.exit(1)
