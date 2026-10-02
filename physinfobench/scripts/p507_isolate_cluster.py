#!/usr/bin/env python3
"""P5.07 隔离重算（集群）：新链 vs 现有打结链的序列簇 + 结构近邻边。

协议（沿用既有冻结规则）：
  - 序列隔离 = mmseqs2 easy-cluster，≥30% 同一性 + 短序列覆盖 ≥70%（P0.06 候选规则、
    KNOT-RESPLIT 同款）；
  - 结构隔离 = foldseek 预筛（-e 0.001 -a）→ US-align 确认 min TM≥0.6 → 结构边；
  - 输出仅记录边界事实，不做任何划分变更（划分变更由本地绑定图更新任务执行）。
输入：data/interim/p507_chains_meta.tsv（chain,type,seq）+ structures_mmcif/（现有结构）
输出：data/interim/p507_isolate/{seq_clusters_rep.tsv, foldseek_hits.tsv, usalign_edges_v2.tsv}
"""
import csv
import gzip
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from usalign_parser import parse_usalign_scores

ROOT = "/lenovofs1/home/jyma/PLM_benchmark/p303_resplit/proj"
P507 = f"{ROOT}/data/interim/p507_isolate"
CIFD = f"{ROOT}/data/raw/rcsb/2026-09-28/p507_cif"
MMCIF = "/lenovofs1/home/jyma/PLM_benchmark/structures_mmcif"
MMSEQS = "/lenovofs1/home/jyma/PLM_benchmark/tools_linux/mmseqs/bin/mmseqs"
FOLDSEEK = "/lenovofs1/home/jyma/PLM_benchmark/tools_linux/foldseek/bin/foldseek"
USALIGN = "/lenovofs1/home/jyma/PLM_benchmark/p303_knot_foldseek/tools/USalign/USalign"
OUT = f"{ROOT}/data/interim/p507_isolate"


def die(m):
    print(f"[p507iso FATAL] {m}", file=sys.stderr)
    sys.exit(1)


def chain_ca_from_cif(cif_path, auth_chain, out_pdb):
    coords, seen = [], set()
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
                coords.append((float(parts[cols["Cartn_x"]]),
                               float(parts[cols["Cartn_y"]]),
                               float(parts[cols["Cartn_z"]])))
    if len(coords) < 30:
        return 0
    with open(out_pdb, "w") as f:
        for i, (x, y, z) in enumerate(coords, 1):
            f.write(f"ATOM  {i:5d}  CA  ALA A{i:4d}    {x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00           C\n")
    return len(coords)


def stage1_new_chain_pdbs():
    meta = list(csv.DictReader(open(f"{ROOT}/data/interim/p507_chains_meta.tsv"), delimiter="\t"))
    os.makedirs(f"{OUT}/pdb_new", exist_ok=True)
    n = 0
    for r in meta:
        chain, pdb, auth = r["chain"], r["chain"].split("_")[0], r["chain"].split("_", 1)[1]
        out = f"{OUT}/pdb_new/{chain}.pdb"
        if os.path.exists(out):
            n += 1
            continue
        cif = f"{CIFD}/{pdb.lower()}.cif.gz"
        if not os.path.exists(cif):
            cif = f"{CIFD}/{pdb.lower()}.cif"
        if chain_ca_from_cif(cif, auth, out) == 0:
            die(f"新链 CA 提取失败: {chain}")
        n += 1
    print(f"[s1] 新链 CA-PDB: {n}", flush=True)


def stage2_existing_chain_pdbs():
    meta = list(csv.DictReader(open(f"{ROOT}/data/interim/p507_chains_meta.tsv"), delimiter="\t"))
    new_set = {r["chain"] for r in meta}
    knots = list(csv.DictReader(open(f"{ROOT}/data/curated/knots.tsv"), delimiter="\t"))
    os.makedirs(f"{OUT}/pdb_existing", exist_ok=True)
    jobs = []
    for r in knots:
        chain = r["record_id"].upper()
        if chain in new_set:
            continue
        jobs.append((chain, r["pdb"].lower(), r["chain"]))
    print(f"[s2] 现有链 {len(jobs)}", flush=True)

    def work(job):
        chain, pdb, auth = job
        out = f"{OUT}/pdb_existing/{chain}.pdb"
        if os.path.exists(out):
            return 1
        cif = f"{MMCIF}/{pdb.lower()}.cif"
        if not os.path.exists(cif):
            return 0
        return 1 if chain_ca_from_cif(cif, auth, out) else 0

    with ThreadPoolExecutor(16) as ex:
        done = sum(ex.map(work, jobs))
    print(f"[s2] 现有 CA-PDB 提取成功 {done}/{len(jobs)}", flush=True)


def stage3_mmseqs():
    rep = f"{OUT}/seq_clusters_rep.tsv"
    if os.path.exists(rep):
        print("[s3] mmseqs 幂等跳过", flush=True)
        return
    # 合成 combined fasta（序列源：p507 json + 现有 knots json）
    import json
    seqs = {}
    j1 = json.load(open(f"{ROOT}/data/raw/rcsb/2026-09-28/knot_entry_sequences_p507.json"))
    for r in csv.DictReader(open(f"{ROOT}/data/interim/p507_chains_meta.tsv"), delimiter="\t"):
        c = r["chain"]
        pdb, auth = c.split("_", 1)
        s = j1.get(pdb.lower(), {}).get(auth)
        if s:
            seqs[c] = s
    j2 = json.load(open(f"{ROOT}/data/raw/rcsb/2026-09-25/knot_entry_sequences.json"))
    knots = list(csv.DictReader(open(f"{ROOT}/data/curated/knots.tsv"), delimiter="\t"))
    new_set = {r["chain"] for r in csv.DictReader(open(f"{ROOT}/data/interim/p507_chains_meta.tsv"), delimiter="\t")}
    for r in knots:
        c = r["record_id"].upper()
        if c in new_set or c in seqs:
            continue
        s = j2.get(r["pdb"].lower(), {}).get(r["chain"])
        if s:
            seqs[c] = s
    with open(f"{OUT}/all.fasta", "w") as f:
        for c, s in sorted(seqs.items()):
            f.write(f">{c}\n{s}\n")
    print(f"[s3] combined fasta {len(seqs)}", flush=True)
    subprocess.run([MMSEQS, "easy-cluster", f"{OUT}/all.fasta", f"{OUT}/mmseqs_cluster",
                    f"{OUT}/mmseqs_tmp", "--min-seq-id", "0.3", "-c", "0.7",
                    "--cov-mode", "1"], check=True)
    src = f"{OUT}/mmseqs_cluster_cluster.tsv"
    if not os.path.exists(src):
        src = f"{OUT}/mmseqs_cluster_rep_seq_cluster.tsv"
    os.rename(src, rep)
    print("[s3] mmseqs done", flush=True)


def stage4_foldseek():
    hits = f"{OUT}/foldseek_hits.tsv"
    if os.path.exists(hits):
        print("[s4] foldseek 幂等跳过", flush=True)
        return
    for d in ("db_new", "db_existing"):
        os.makedirs(f"{OUT}/{d}", exist_ok=True)
    subprocess.run([FOLDSEEK, "createdb", f"{OUT}/pdb_new", f"{OUT}/db_new/p507new"], check=True)
    subprocess.run([FOLDSEEK, "createdb", f"{OUT}/pdb_existing", f"{OUT}/db_existing/exist"], check=True)
    subprocess.run([FOLDSEEK, "search", f"{OUT}/db_new/p507new", f"{OUT}/db_existing/exist",
                    f"{OUT}/db_new/aln", f"{OUT}/db_new/tmp", "-e", "0.001", "-a"], check=True)
    subprocess.run([FOLDSEEK, "convertalis", f"{OUT}/db_new/p507new", f"{OUT}/db_existing/exist",
                    f"{OUT}/db_new/aln", hits,
                    "--format-output", "query,target,qtmscore,ttmscore,rmsd"], check=True)
    print("[s4] foldseek done", flush=True)


def stage5_usalign():
    conf = f"{OUT}/usalign_edges_v2.tsv"
    done = set()
    valid_rows = []
    if os.path.exists(conf):
        for line in open(conf):
            p = line.rstrip("\n").split("\t")
            try:
                if p[0] == "query":
                    continue
                if len(p) >= 6 and p[5] == "0" and all(p[i] for i in (2, 3, 4)) \
                        and (p[0], p[1]) == (p[0].strip(), p[1].strip()):
                    s1, s2, sm = float(p[3]), float(p[4]), float(p[2])
                    if abs(sm - min(s1, s2)) <= 0.00011:
                        done.add((p[0], p[1]))
                        valid_rows.append(p[:6])
            except ValueError:
                continue
    cand = []
    for line in open(f"{OUT}/foldseek_hits.tsv"):
        p = line.rstrip("\n").split("\t")
        if len(p) < 3:
            continue
        q, t = p[0].rsplit(".pdb", 1)[0] if ".pdb" in p[0] else p[0], \
               p[1].rsplit(".pdb", 1)[0] if ".pdb" in p[1] else p[1]
        tm = max(float(p[2]), float(p[3]))
        if tm >= 0.45 and (q, t) not in done:
            cand.append((q, t))
    cand = sorted(set(cand))
    print(f"[s5] US-align 待确认 {len(cand)}", flush=True)

    def work(pair):
        q, t = pair
        try:
            r = subprocess.run([USALIGN, f"{OUT}/pdb_new/{q}.pdb", f"{OUT}/pdb_existing/{t}.pdb",
                                "-m", "true"], capture_output=True, text=True, timeout=120)
            if r.returncode != 0:
                return q, t, None, None, None, r.returncode
            s1, s2, tm = parse_usalign_scores(r.stdout)
            return q, t, tm, s1, s2, r.returncode
        except Exception as exc:
            print(f"[US-align failed] {q} {t}: {exc}", file=sys.stderr)
            return q, t, None, None, None, 1

    with ThreadPoolExecutor(16) as ex:
        results = list(ex.map(work, cand))
    with open(conf + ".tmp", "w", newline="") as f:
        w = csv.writer(f, delimiter="\t", lineterminator="\n")
        w.writerow(["query", "target", "tm_min", "tm_norm_input1", "tm_norm_input2", "returncode"])
        w.writerows(valid_rows)
        w.writerows([[q, t, "" if tm is None else f"{tm:.6f}",
                      "" if s1 is None else f"{s1:.6f}",
                      "" if s2 is None else f"{s2:.6f}", rc]
                     for q, t, tm, s1, s2, rc in results])
    os.replace(conf + ".tmp", conf)
    n_edges = sum(1 for _, _, tm, _, _, rc in results if rc == 0 and tm is not None and tm >= 0.6)
    print(f"[s5] US-align 边 (min TM≥0.6): {n_edges}", flush=True)


def main():
    os.makedirs(OUT, exist_ok=True)
    stage1_new_chain_pdbs()
    stage2_existing_chain_pdbs()
    stage3_mmseqs()
    stage4_foldseek()
    stage5_usalign()
    print("[p507iso] DONE")


if __name__ == "__main__":
    main()
