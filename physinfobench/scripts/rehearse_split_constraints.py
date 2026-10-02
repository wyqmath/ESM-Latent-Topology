#!/usr/bin/env python3
"""P1.17 划分可行性只读预演（G1 前补证）。

只读输入：旧项目端点表（192 端点序列）、fold_switch_global.tsv（96 对/103 UniProt）、
旧 v5 split 表（曝光核对）、SIFTS 家族文件（家族隔离）。
计算：mmseqs2 同源聚类（预注册 30%+70% 主口径；25%/40% 敏感性）→ 并查集合并约束
（pair 边 + 聚类共享边）→ 独立组数/最大组/组成；strict 家族隔离；曝光修正核对。
铁律：不写 group_map、不生成 split manifest、不做最终匹配；结构近邻（TM≥0.6）边
在 P2.01 才有数据，本预演明确标注该缺口。
"""
import csv
import hashlib
import json
import os
import subprocess
import tempfile
from collections import Counter, defaultdict

B = "/Users/yuan/Documents/Codex/2026-09-08/jie/benchmark_step1"
MMSEQS = os.path.join(B, "tools/mmseqs2-18-8cc5c/bin/mmseqs")
ENDPOINT = os.path.join(B, "manifests/fold_pair_endpoint_sequence_structure_summary.tsv")
ENDPOINT_SHA = "dee56a9fbfc3e5a814a9a7e0a93d25fcbd6e849282fd91ecb47f5a6aac1629dd"
V5 = os.path.join(B, "manifests/task_sample_splits_v5_2026-09-15.tsv")
OUT_TSV = "reports/fs_three_layer/split_rehearsal_groups.tsv"
OUT_JSON = "reports/fs_three_layer/split_rehearsal_qc.json"


class UF:
    def __init__(self):
        self.p = {}

    def find(self, x):
        self.p.setdefault(x, x)
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[ra] = rb


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def run_mmseqs(fasta, min_id, cov, tmpd):
    subprocess.run([MMSEQS, "easy-cluster", fasta, os.path.join(tmpd, "res"),
                    os.path.join(tmpd, "tmp"), "--min-seq-id", str(min_id),
                    "-c", str(cov), "--cov-mode", "0"], check=True, capture_output=True)
    clusters = defaultdict(list)
    with open(os.path.join(tmpd, "res_cluster.tsv")) as f:
        for line in f:
            c, s = line.rstrip("\n").split("\t")
            clusters[c].append(s)
    return clusters


def main():
    if sha256_file(ENDPOINT) != ENDPOINT_SHA:
        raise SystemExit("endpoint table sha mismatch")
    # 端点→pair 映射与序列
    endp = {}   # ep_id -> (pair_id, seq)
    with open(ENDPOINT) as f:
        for row in csv.DictReader(f, delimiter="\t"):
            eid = f"{row['pdb_id'].lower()}_{row['requested_chain']}"
            pair = None   # 由调用方按 pair_id 前缀解析
            endp[eid] = (row["candidate_id"], row["observed_sequence"].strip())
    pairs = {}
    with open("data/curated/fold_switch_global.tsv") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            pairs[row["pair_id"]] = (row["pdb_a"].lower() + "_" + row["chain_a"],
                                     row["pdb_b"].lower() + "_" + row["chain_b"],
                                     row["tier"])
    ep2pair = {}
    for pid, (ea, eb, _) in pairs.items():
        ep2pair[ea] = pid
        ep2pair[eb] = pid
    with tempfile.TemporaryDirectory() as tmpd:
        fa = os.path.join(tmpd, "endpoints.fasta")
        with open(fa, "w") as f:
            for eid, (_, seq) in endp.items():
                f.write(f">{eid}\n{seq}\n")
        results = {}
        for label, mid, cov in (("id30cov70", 0.3, 0.7), ("id25cov70", 0.25, 0.7), ("id40cov70", 0.4, 0.7)):
            d = os.path.join(tmpd, label)
            os.makedirs(d, exist_ok=True)
            results[label] = run_mmseqs(fa, mid, cov, d)
    # 组装：pair 为节点；聚类共享边（不同 pair 的端点同簇 → 连边）
    group_stats = {}
    for label, clusters in results.items():
        uf = UF()
        for pid in pairs:
            uf.find(pid)
        for cname, eps in clusters.items():
            ps = sorted({ep2pair[e] for e in eps if e in ep2pair})
            for i in range(1, len(ps)):
                uf.union(ps[0], ps[i])
        comps = defaultdict(list)
        for pid in pairs:
            comps[uf.find(pid)].append(pid)
        sizes = sorted((len(v) for v in comps.values()), reverse=True)
        giant = max(comps.values(), key=len)
        group_stats[label] = {
            "n_groups": len(comps), "size_dist_top10": sizes[:10], "max_size": sizes[0],
            "giant_members_tiers": dict(Counter(pairs[p][2] for p in giant)),
            "strict_in_groups_with_others": sum(
                1 for p in pairs if pairs[p][2] == "strict_state_candidate"
                and len(comps[uf.find(p)]) > 1),
        }
    # strict 家族隔离（SIFTS 家族）
    fam = defaultdict(lambda: defaultdict(set))
    fam_files = {"pfam": ("data/raw/sifts/2026-09-23/pdb_chain_pfam.csv.gz", "PFAM_ID"),
                 "cath": ("data/raw/sifts/2026-09-23/pdb_chain_cath_uniprot.csv.gz", "CATH_ID"),
                 "scop2fa": ("data/raw/sifts/2026-09-23/pdb_chain_scop2_uniprot.csv.gz", "FA_DOMID")}
    import gzip
    for system, (path, col) in fam_files.items():
        f = gzip.open(path, "rt")
        first = f.readline()
        while first.startswith("#"):
            first = f.readline()
        hdr = first.rstrip().split(",")
        iP, iS, iF = hdr.index("PDB"), hdr.index("SP_PRIMARY"), hdr.index(col)
        for row in csv.reader(f, delimiter=","):
            try:
                fam[row[iS].strip()][system].add(row[iF].strip())
            except IndexError:
                pass
    acc2pairs = defaultdict(list)
    with open("data/curated/fold_switch_global.tsv") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            for c in ("uniprot_a", "uniprot_b"):
                u = (row.get(c) or "").strip()
                if u:
                    acc2pairs[u].append(row["pair_id"])
    strict_accs = {u for u, ps in acc2pairs.items()
                   if any(pairs[p][2] == "strict_state_candidate" for p in ps)}
    # 家族共享：strict 的家族是否被其他池蛋白共享（限池内 103 accession）
    pool_accs = set(acc2pairs)
    fam_share = {}
    for sa in sorted(strict_accs):
        shared_with = set()
        for system, vals in fam.get(sa, {}).items():
            for v in vals:
                for other in pool_accs:
                    if other != sa and v in fam.get(other, {}).get(system, set()):
                        shared_with.add(other)
        fam_share[sa] = sorted(shared_with)
    # 曝光核对（v5）
    v5 = list(csv.DictReader(open(V5), delimiter="\t"))
    exp_sets = defaultdict(set)
    for r in v5:
        if r["sample"].startswith("porter_"):
            exp_sets[r["sample"]].add(r["split"])
    exposed = set(exp_sets)
    dev_only = sum(1 for s_ in exp_sets.values() if s_ == {"development"})
    test_only = sum(1 for s_ in exp_sets.values() if s_ == {"test"})
    both = sum(1 for s_ in exp_sets.values() if len(s_) > 1)
    exposure_split_detail = {"dev_only": dev_only, "test_only": test_only, "both_dev_and_test": both,
                             "test_side_total": test_only + both}
    tiers = {pid: t for pid, (a, b, t) in pairs.items()}
    exposure = {
        "v5_unique_porter_pairs": len(exposed),
        "v5_split_detail": exposure_split_detail,
        "exposed_in_current_pool": len(exposed & set(pairs)),
        "unexposed_count": len(set(pairs) - exposed),
        "unexposed_tiers": dict(Counter(tiers[p] for p in set(pairs) - exposed)),
        "exposed_tiers": dict(Counter(tiers[p] for p in exposed & set(pairs))),
        "strict_all_exposed": all(p in exposed for p in pairs if tiers[p] == "strict_state_candidate"),
        "note": "P1.09 曾称 96 对全部曝光——实测 v5 仅 35 对（28 dev+7 test）；本预演修正该声明（P1.09 产物不改动，移交 P1.18/G1）",
    }
    out_rows = []
    for label, st in group_stats.items():
        out_rows.append(["group_stats", label, "n_groups", str(st["n_groups"]), ""])
        out_rows.append(["group_stats", label, "max_size", str(st["max_size"]),
                         "top10=" + ",".join(map(str, st["size_dist_top10"]))])
        out_rows.append(["group_stats", label, "giant_tiers", json.dumps(st["giant_members_tiers"]), ""])
        out_rows.append(["group_stats", label, "strict_in_groups_with_others",
                         str(st["strict_in_groups_with_others"]), "10 个 strict 中与其他池成员同组者"])
    for sa, sh in fam_share.items():
        out_rows.append(["family_share", sa, "shared_pool_accessions", str(len(sh)), ";".join(sh[:8])])
    with open(OUT_TSV, "w", newline="") as f:
        f.write("dim\tkey\tmetric\tvalue\tnote\n")
        csv.writer(f, delimiter="\t", lineterminator="\n").writerows(out_rows)
    qc = {
        "generated": os.popen("TZ=Asia/Shanghai date '+%Y-%m-%d %H:%M'").read().strip(),
        "mmseqs": {"binary": MMSEQS, "version": subprocess.run([MMSEQS, "version"], capture_output=True, text=True).stdout.strip(),
                   "main_rule": "easy-cluster min-seq-id 0.3 -c 0.7 cov-mode 0（[待冻结] 候选阈值）",
                   "sensitivity": ["id25cov70", "id40cov70"]},
        "structure_edges_missing": "TM>=0.6 结构近邻边无现成数据（P2.01 计算后并入正式分组）；本预演仅序列+pair 边",
        "group_stats": group_stats,
        "family_share_strict": fam_share,
        "exposure": exposure,
        "assertions": {
            "v5_exposure_35_not_96": exposure["v5_unique_porter_pairs"] == 35,
            "strict_all_exposed": exposure["strict_all_exposed"],
        },
    }
    with open(OUT_JSON, "w") as f:
        json.dump(qc, f, ensure_ascii=False, indent=1, sort_keys=True)
    print("groups:", {k: (v["n_groups"], v["max_size"]) for k, v in group_stats.items()})
    print("exposure:", {k: exposure[k] for k in ("v5_unique_porter_pairs", "unexposed_count", "strict_all_exposed")})
    print("strict family sharers:", {k: len(v) for k, v in fam_share.items()})
    # 端点→pair 1:1 断言（每端点恰属一对；冲突即失败）
    from collections import Counter as _C
    ep_counts = _C(ep for pid, (ea, eb, _) in pairs.items() for ep in (ea, eb))
    assert all(v == 1 for v in ep_counts.values()), f"端点 1:1 违例: {[k for k,v in ep_counts.items() if v>1]}"
    assert all(qc["assertions"].values())
    print("DONE (只读预演；不写 group_map/split)")


if __name__ == "__main__":
    main()
