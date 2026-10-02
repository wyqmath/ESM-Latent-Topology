#!/usr/bin/env python3
"""P2.01 步骤一：提取 192 个 strict 端点链 + 16 个 PN-A 主选候选代表链为独立 PDB。

用旧项目 venv 的 gemmi 运行：$B/.venv/bin/python scripts/extract_p201_chains.py
坐标源=旧项目 structures_mmcif/<pdb>.cif（96 对端点全覆盖，P1 预验证）；
PN 候选坐标从 RCSB 下载（--skip-pn-download 时仅登记）。
输出：data/interim/p201_step1/chains/*.pdb + extract_manifest.tsv（含源 cif sha256 前 16 位）
"""
import csv
import hashlib
import gzip
import os
import sys
import time
import urllib.request

import gemmi

B = "/Users/yuan/Documents/Codex/2026-09-08/jie/benchmark_step1"
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
WORK = os.path.join(ROOT, "data/interim/p201_step1")
CH = os.path.join(WORK, "chains")
SKIP_DL = "--skip-pn-download" in sys.argv


def die(m):
    print(f"[extract FATAL] {m}", file=sys.stderr)
    sys.exit(1)


def extract_chain(cif, chain_name):
    """读 cif，清理（去 altloc/H/配体/水），保留首模型中指定 author 链，返回 (structure, n_residues)。"""
    st = gemmi.read_structure(cif)
    st.remove_alternative_conformations()
    st.remove_hydrogens()
    st.setup_entities()
    st.remove_ligands_and_waters()
    st.remove_empty_chains()
    model = st[0]  # NMR/多模型取首模型（预注册）
    name = chain_name  # author 链名大小写敏感（5lj3 同时存在 M 与 m）
    exact = [c for c in model if c.name == name]
    if exact:
        keep_name = name
    else:  # 大小写回退：仅当恰好一个不区分大小写命中时采用
        ci = [c for c in model if c.name.upper() == name.upper()]
        if len(ci) != 1:
            return None
        keep_name = ci[0].name
    if not any(c.name == keep_name for c in model):
        return None
    for i in range(len(model) - 1, -1, -1):
        if model[i].name != keep_name:
            del model[i]
    while len(st) > 1:  # 只保留首模型（NMR 多模型不得随链写出）
        del st[1]
    st.setup_entities()
    target = model[0]  # 删除后再取引用（gemmi 删除会使旧引用失效）
    if len(model) != 1 or len(target) == 0:
        return None
    return st, len(target)


os.makedirs(CH, exist_ok=True)
os.makedirs(os.path.join(ROOT, "data/raw/p201_step1_pn_chains/2026-09-24"), exist_ok=True)

# ---- strict 192 端点 ----
summ = {(r["pdb_id"].lower(), r["requested_chain"].upper()): r
        for r in csv.DictReader(open(B + "/manifests/fold_pair_endpoint_sequence_structure_summary.tsv"),
                                 delimiter="\t")}
g = list(csv.DictReader(open(ROOT + "/data/curated/fold_switch_global.tsv"), delimiter="\t"))
manifest = []
n_strict = 0
for r in g:
    n = int(r["pair_id"].split("_")[1])
    for side, pdb, ch in (("A", r["pdb_a"], r["chain_a"]), ("B", r["pdb_b"], r["chain_b"])):
        pdb_l = pdb.lower()
        s = summ[(pdb_l, ch.upper())]
        cif = os.path.join(B, "structures_mmcif", pdb_l + ".cif")
        res = extract_chain(cif, ch)
        if res is None:
            die(f"{pdb_l}{ch}: 链未找到或空")
        st, n_res = res
        st.name = f"{r['pair_id']}_{side}"
        key = f"EP_{r['pair_id']}_{side}__{pdb_l}{ch.upper()}"
        path = os.path.join(CH, key + ".pdb")
        st.write_pdb(path)
        manifest.append([key, "legacy_mmcif", cif, hashlib.sha256(open(cif, "rb").read()).hexdigest()[:16],
                         str(n_res), s["observed_length"]])
        n_strict += 1

# ---- PN-A 主选候选代表链（16 accession）----
pna = [r for r in csv.DictReader(open(ROOT + "/reports/fs_three_layer/pna_evidence_review.tsv"),
                                 delimiter="\t") if r["primary_for"]]
accs = sorted({r["uniprot_accession"] for r in pna})
chains = {}
with gzip.open(ROOT + "/data/curated/fold_switch_unlabeled_structure_chains.tsv.gz", "rt") as f:
    for c in csv.DictReader(f, delimiter="\t"):
        if c["uniprot_accession"] in accs:
            chains.setdefault(c["uniprot_accession"], []).append(c)
n_pn = 0
for acc in accs:
    best = max(chains[acc], key=lambda c: (float(c["observed_coverage"]), int(c["n_alignment_segments"]),
                                           c["pdb_id"], c["chain_id"]))
    pdb_l, chain = best["pdb_id"].lower(), best["chain_id"].upper()
    raw_dir = os.path.join(ROOT, "data/raw/p201_step1_pn_chains/2026-09-24")
    cif = os.path.join(raw_dir, pdb_l + ".cif")
    if not os.path.exists(cif):
        if SKIP_DL:
            print(f"[skip] {acc} {pdb_l} (download disabled)")
            continue
        url = f"https://files.rcsb.org/download/{pdb_l}.cif"
        for attempt in range(3):
            try:
                req = urllib.request.Request(url, headers={"User-Agent": "PhysInfoBench-P201/1.0"})
                with urllib.request.urlopen(req, timeout=60) as resp, open(cif, "wb") as fo:
                    fo.write(resp.read())
                break
            except Exception as e:
                if attempt == 2:
                    die(f"{acc} {pdb_l} 下载失败: {e}")
                time.sleep(2 + 3 * attempt)
        time.sleep(0.4)
    res = extract_chain(cif, chain)
    if res is None:
        die(f"PN {acc} {pdb_l}{chain}: 链未找到或空")
    st, n_res = res
    st.name = f"PN_{acc}"
    key = f"PN_{acc}__{pdb_l}{chain}"
    st.write_pdb(os.path.join(CH, key + ".pdb"))
    manifest.append([key, "rcsb_download", cif,
                     hashlib.sha256(open(cif, "rb").read()).hexdigest()[:16],
                     str(n_res), ""])
    n_pn += 1

with open(os.path.join(WORK, "extract_manifest.tsv"), "w", newline="") as f:
    w = csv.writer(f, delimiter="\t", lineterminator="\n")
    w.writerow(["key", "source_type", "source_path", "source_sha256_16", "n_residues", "observed_length_ref"])
    w.writerows(sorted(manifest))
print(f"[extract] strict_endpoints={n_strict} pn_candidates={n_pn} total={len(manifest)} -> {CH}")
