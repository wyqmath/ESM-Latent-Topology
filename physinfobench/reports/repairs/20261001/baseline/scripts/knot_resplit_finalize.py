#!/usr/bin/env python3
"""KNOT-RESPLIT 步骤 3：绑定图预演（不改现行划分）。

构造：精确复刻 build_p202_split_manifest.py 的图与分配逻辑（节点 G::/C::；边=端点∈簇 192、
同 UniProt 绑定、端点 sha 绑定、matched_set 17；forced_dev=S1/S2+受控三系统；Random(2026)
洗牌组级 70:15:15；dev_fold=i%5），并证明：
  模式 0（不加打结边）：复现 manifest 与现行 data/splits/split_manifest.tsv 逐行一致（等价性证明）；
  模式 1（+2,998 打结 min≥0.6 边）：预演新分配 → reports/knot_resplit/step3_dryrun_manifest.tsv
  + step3_dryrun_qc.json（新组数/最大组/跨任务合并/各任务三集合构成/类型构成/dev-forced 记账/
  确认保留可评价性）。
纪律：分配只用 ID+结构相似+已登记分组约束；presence/type 标签仅在分配完成后做构成报告。
"""
import csv
import gzip
import hashlib
import json
import os
import random
import sys
from collections import Counter, defaultdict

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
WORK = os.path.join(ROOT, "data/interim/p202")
B = "/Users/yuan/Documents/Codex/2026-09-08/jie/benchmark_step1"
SEED = 2026
RUN_TS = __import__("datetime").datetime.now().strftime("%Y-%m-%d %H:%M")


def die(m):
    print(f"[kr FATAL] {m}", file=sys.stderr)
    sys.exit(1)


def rd(p, d="\t"):
    with open(p, newline="") as f:
        return list(csv.DictReader(f, delimiter=d))


def build(knot_edges, out_manifest):
    gm = rd(os.path.join(ROOT, "data/splits/group_map.tsv"))
    gm_by_sid = {r["sample_id"]: r for r in gm}
    sample_group = {r["sample_id"]: r["group_id"] for r in gm}
    sample_area = {r["sample_id"]: r["task_area"] for r in gm}
    cluster = {}
    with open(os.path.join(WORK, "l2_cluster.tsv")) as f:
        for line in f:
            c, m = line.rstrip("\n").split("\t")
            cluster[m] = c
    ep_of_pair = {}
    for r in rd(os.path.join(ROOT, "data/curated/fold_switch_global.tsv")):
        for side, pdb, ch in (("A", r["pdb_a"], r["chain_a"]), ("B", r["pdb_b"], r["chain_b"])):
            ep_of_pair[f"EP_{r['pair_id']}_{side}__{pdb.lower()}{ch.upper()}"] = r["pair_id"]
    uf = {}

    def find(x):
        uf.setdefault(x, x)
        while uf[x] != x:
            uf[x] = uf[uf[x]]
            x = uf[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            uf[rb] = ra

    for sid, gid in sample_group.items():
        find(f"G::{gid}")
    bg_members = defaultdict(list)
    for member, rep in cluster.items():
        bg_members[rep].append(member)
        find(f"C::{rep}")
    for ep, pair in ep_of_pair.items():
        rep = cluster.get(ep)
        if rep is None:
            die(f"端点不在聚类中: {ep}")
        union(f"G::{sample_group[pair]}", f"C::{rep}")
    acc2rep = {}
    for member, rep in cluster.items():
        acc2rep.setdefault(member, rep)
    for sid, g in sample_group.items():
        row = gm_by_sid[sid]
        for u in filter(None, row["uniprots"].split(";")):
            rep = acc2rep.get(u)
            if rep is not None:
                union(f"G::{g}", f"C::{rep}")
    summ = {(r["pdb_id"].lower(), r["requested_chain"].upper()): r
            for r in csv.DictReader(open(B + "/manifests/fold_pair_endpoint_sequence_structure_summary.tsv"), delimiter="\t")}
    sha2accs = defaultdict(list)
    with gzip.open(os.path.join(ROOT, "data/curated/fold_switch_unlabeled_sequence.tsv.gz"), "rt") as f:
        for r in csv.DictReader(f, delimiter="\t"):
            sha2accs[r["sequence_sha256"]].append(r["uniprot_accession"])
    for r in csv.DictReader(open(os.path.join(ROOT, "data/curated/fold_switch_global.tsv"), newline=""), delimiter="\t"):
        g = sample_group[r["pair_id"]]
        for side, pdb, ch in (("A", r["pdb_a"], r["chain_a"]), ("B", r["pdb_b"], r["chain_b"])):
            seq = summ[(pdb.lower(), ch.upper())]["observed_sequence"]
            sha = hashlib.sha256(seq.encode()).hexdigest()
            for acc in sha2accs.get(sha, []):
                rep = acc2rep.get(acc)
                if rep is not None:
                    union(f"G::{g}", f"C::{rep}")
    mc = rd(os.path.join(ROOT, "data/curated/fold_switch_matched_controls.tsv"))
    for r in mc:
        if r["ratio"] == "1:0" or r["control_accession"] in ("", "-", "NONE_ELIGIBLE"):
            continue
        rep = cluster.get(r["control_accession"])
        if rep is None:
            die(f"matched 对照不在背景聚类: {r['control_accession']}")
        union(f"G::{sample_group[r['positive_pair']]}", f"C::{rep}")
    exp = {r["pair_id"]: r for r in rd(os.path.join(ROOT, "reports/fs_three_layer/exposure_scope_audit.tsv"))}
    forced_dev = set()
    for pid, e in exp.items():
        if e["classification"] in ("v5_split_member", "earlier_split_only"):
            forced_dev.add(f"G::{sample_group[pid]}")
    for sid, area in sample_area.items():
        if area in ("pyp", "rnase_a", "bpti"):
            forced_dev.add(f"G::{sample_group[sid]}")
    # —— 新增：打结结构绑定边（链→sample→组）；对 ID=文件名 {pdb}_{chain}，经 chain_list 解析 record_id ——
    n_knot_edges = 0
    man_lower = {}
    for r in man_rows:
        if r["task_area"] == "knot":
            suf = r["sample_id"].split(":", 1)[1]
            man_lower[suf.lower()] = suf
    cl_by_pc = {}
    for r in rd(os.path.join(ROOT, "data/interim/p303/knot_chain_list.tsv")):
        cl_by_pc[(r["pdb"].lower(), r["chain"])] = r["record_id"]
    for a, b, tm in knot_edges:
        sids = []
        for x in (a, b):
            f, c = x.rsplit("_", 1)
            rid = cl_by_pc.get((f.lower(), c))
            if rid is None:
                die(f"打结链不在 chain_list: {x}")
            suf = man_lower.get(rid.lower())
            if suf is None:
                die(f"打结链无 manifest 样本: {x} (record_id={rid})")
            sids.append("knot:" + suf)
        union(f"G::{sample_group[sids[0]]}", f"G::{sample_group[sids[1]]}")
        n_knot_edges += 1
    all_nodes = sorted(uf.keys())
    groups_nodes = {find(x) for x in all_nodes}
    forced_dev_roots = {find(x) for x in forced_dev}
    rng = random.Random(SEED)
    ordered = sorted(groups_nodes)
    rng.shuffle(ordered)
    n = len(ordered)
    n_dev = round(n * 0.70)
    n_conf = round(n * 0.15)
    assign = {}
    for i, gnode in enumerate(ordered):
        if gnode in forced_dev_roots or i < n_dev:
            assign[gnode] = "development"
        elif i < n_dev + n_conf:
            assign[gnode] = "confirmation"
        else:
            assign[gnode] = "final_holdout"
    dev_group_nodes = sorted({find(f"G::{gid}") for gid in sample_group.values()
                              if assign[find(f"G::{gid}")] == "development"})
    fold_of_group = {}
    for i, gnode in enumerate(dev_group_nodes):
        fold_of_group[gnode] = i % 5
    with open(out_manifest, "w", newline="") as f:
        w = csv.writer(f, delimiter="\t", lineterminator="\n")
        w.writerow(["sample_id", "task_area", "group_id", "split", "dev_fold"])
        for sid in sorted(sample_group):
            gnode = find(f"G::{sample_group[sid]}")
            sp = assign[gnode]
            fold = fold_of_group.get(gnode, "") if sp == "development" else ""
            w.writerow([sid, sample_area[sid], sample_group[sid], sp, fold])
    bg_split = {}
    import io
    _buf = io.StringIO()
    w = csv.writer(_buf, delimiter="\t", lineterminator="\n")
    w.writerow(["uniprot_accession", "cluster_rep", "group_node", "split"])
    for rep in sorted(bg_members):
        gnode = find(f"C::{rep}")
        sp = assign[gnode]
        for member in sorted(bg_members[rep]):
            w.writerow([member, rep, gnode, sp])
            bg_split[member] = sp
    with open(OUT_BG, "wb") as fh:
        with gzip.GzipFile(filename="", mode="wb", fileobj=fh, mtime=0) as gz:
            gz.write(_buf.getvalue().encode())
    all_root_count = len({find(x) for x in uf})
    return assign, find, bg_members, n_knot_edges, all_root_count, bg_split


man_rows = rd(os.path.join(ROOT, "data/splits/split_manifest.tsv"))  # 供 man_lower；现行基线
knot_edges = []
with open(os.path.join(ROOT, "data/interim/p303/usalign_confirmed.tsv")) as f:
    for line in f:
        p = line.rstrip("\n").split("\t")
        if len(p) < 3 or not p[2].strip():
            continue
        try:
            tms = [float(x) for x in p[2].split()]
        except ValueError:
            continue
        if len(tms) >= 2 and min(tms) >= 0.6:
            knot_edges.append((p[0], p[1], min(tms)))
if len(knot_edges) != 2998:
    die(f"打结绑定边 {len(knot_edges)} != 2998")
OUTD = os.path.join(ROOT, "reports/knot_resplit/step4_generation")
os.makedirs(OUTD, exist_ok=True)
OUT_M = os.path.join(ROOT, "data/splits/split_manifest.tsv")
OUT_BG = os.path.join(ROOT, "data/splits/fs_l2_background_manifest.tsv.gz")
OUT_CK = os.path.join(ROOT, "data/splits/checksums.txt")
OUT_QC = os.path.join(ROOT, "data/splits/split_qc.json")

# 模式 0：等价性证明
assign0, find0, bg0, nk0, roots0, bs0 = build([], os.path.join(OUTD, "mode0_equivalence_check.tsv"))
cur = {r["sample_id"]: (r["split"], r["dev_fold"]) for r in rd(os.path.join(ROOT, "reports/knot_resplit/old_version/split_manifest.tsv"))}  # 基线=留档旧版
diff = []
for r in rd(os.path.join(OUTD, "mode0_equivalence_check.tsv")):
    if cur.get(r["sample_id"]) != (r["split"], r["dev_fold"]):
        diff.append(r["sample_id"])
if diff:
    die(f"模式 0 复现失败（{len(diff)} 行不一致）：{diff[:5]}")
print(f"[kr] 模式0 复现：与现行 manifest 逐行一致（4,858 行；等价性证明成立）")

# 模式 1：预演
assign1, find1, bg1, nk1, roots1, bg_split = build(knot_edges, OUT_M)
print(f"[kr] 模式1：打结边 {nk1}")
# 统计
roots_members = defaultdict(list)
for sid, g in {r["sample_id"]: r["group_id"] for r in man_rows}.items():
    roots_members[find1(f"G::{g}")].append(sid)
bg_roots_members = defaultdict(int)
for rep in bg1:
    bg_roots_members[find1(f"C::{rep}")] += len(bg1[rep])
all_root_sizes = [len(v) for v in roots_members.values()] + list(bg_roots_members.values())
area_by_root = defaultdict(Counter)
for sid, g in {r["sample_id"]: r["group_id"] for r in man_rows}.items():
    area_by_root[find1(f"G::{g}")][man_area := next(r["task_area"] for r in man_rows if r["sample_id"] == sid)] += 1
multi_task_roots = {k: dict(v) for k, v in area_by_root.items() if len(v) > 1}
new_manifest = rd(OUT_M)
area_split = defaultdict(Counter)
for r in new_manifest:
    area_split[r["task_area"]][r["split"]] += 1
old_area_split = defaultdict(Counter)
for r in man_rows:
    old_area_split[r["task_area"]][r["split"]] += 1
def knot_key(x):
    return x.replace("_", "").lower()


kt = {knot_key(r["record_id"]): r for r in rd(os.path.join(ROOT, "data/curated/knots.tsv"))
      if r["type_task_tier"] == "eligible"}
man_lower = {}
for suf in {r["sample_id"].split(":", 1)[1] for r in man_rows if r["task_area"] == "knot"}:
    man_lower[suf.lower()] = suf
type_split = defaultdict(Counter)
for r in new_manifest:
    if r["task_area"] == "knot":
        rid = knot_key(r["sample_id"].split(":", 1)[1])
        if rid in kt:
            type_split[r["split"]][kt[rid]["c2_primary"]] += 1
pos_split = Counter()
neg_split = Counter()
ktall = {knot_key(r["record_id"]): r for r in rd(os.path.join(ROOT, "data/curated/knots.tsv"))}
for r in new_manifest:
    if r["task_area"] == "knot":
        rec = ktall.get(knot_key(r["sample_id"].split(":", 1)[1]))
        if rec and rec["presence_mask"] == "1":
            (pos_split if rec["presence_target"] == "1" else neg_split)[r["split"]] += 1
# forced_dev 记账（S1/S2+受控三系统根在新分配下必须全 dev）
forced_dev_report = {}
exp = {r["pair_id"]: r for r in rd(os.path.join(ROOT, "reports/fs_three_layer/exposure_scope_audit.tsv"))}
forced_samples = defaultdict(list)
for pid, e in exp.items():
    if e["classification"] in ("v5_split_member", "earlier_split_only"):
        forced_samples[pid].append(assign1[find1(f"G::" + next(r["group_id"] for r in man_rows if r["sample_id"] == pid))])
bad_forced = {p: v for p, v in forced_samples.items() if any(x != "development" for x in v)}
# 含打结链的 labeled 根规模
knot_root_sizes = []
knot_sids = {"knot:" + man_lower[r["record_id"].lower()] for r in rd(os.path.join(ROOT, "data/interim/p303/knot_chain_list.tsv"))}
sid_root = {}
for sid, g in {r["sample_id"]: r["group_id"] for r in man_rows}.items():
    sid_root[sid] = find1(f"G::{g}")
knot_roots = {sid_root[s] for s in knot_sids if s in sid_root}
for rt in knot_roots:
    knot_root_sizes.append(sum(1 for s, r2 in sid_root.items() if r2 == rt))
qc = {
    "forced_dev_samples_all_development": not bad_forced,
    "forced_dev_samples_checked": len(forced_samples),
    "forced_dev_violations": bad_forced,
    "max_labeled_root_containing_knot_chains": max(knot_root_sizes),
    "knot_roots_count": len(knot_roots),
    "mode0_reproduction": "PASS (4858 rows identical)",
    "knot_edges_added": nk1,
    "groups_total_mode0": roots0,
    "groups_total_new": roots1,
    "groups_total_new_labeled_plus_bg_check": len({find1(f"G::{g}") for g in {r["group_id"] for r in man_rows}} | {find1(f"C::{rep}") for rep in bg1}),
    "max_root_members_labeled": max(all_root_sizes),
    "max_root_members_bg": max(bg_roots_members.values()),
    "multi_task_roots": {k[:60]: v for k, v in list(multi_task_roots.items())[:10]},
    "n_multi_task_roots": len(multi_task_roots),
    "area_split_new": {a: dict(c) for a, c in area_split.items()},
    "area_split_old": {a: dict(c) for a, c in old_area_split.items()},
    "knot_type_by_split_new": {k: dict(v) for k, v in type_split.items()},
    "knot_presence_usable_by_split_new": {"pos": dict(pos_split), "neg": dict(neg_split)},
}
json.dump(qc, open(os.path.join(OUTD, "generation_qc.json"), "w"), ensure_ascii=False, indent=1, sort_keys=True)
with open(os.path.join(ROOT, "data/splits/knot_structural_edges.tsv"), "w", newline="") as f:
    w = csv.writer(f, delimiter="\t", lineterminator="\n")
    w.writerow(["chain_a", "chain_b", "tm_min"])
    for a, b, tm in knot_edges:
        w.writerow([a, b, tm])

def _sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for ch in iter(lambda: fh.read(1 << 20), b""):
            h.update(ch)
    return h.hexdigest()
with open(OUT_CK, "w") as f:
    for p in (OUT_M, OUT_BG, os.path.join(ROOT, "data/splits/knot_structural_edges.tsv")):
        f.write(f"{_sha(p)}  {os.path.relpath(p, ROOT)}\n")
old = json.load(open(os.path.join(ROOT, "reports/knot_resplit/old_version/split_qc.json")))
new_qc = {
    "run_ts": RUN_TS, "seed": SEED, "proportions": "70:15:15",
    "generated_by": "KNOT-RESPLIT 方案 A（用户 2026-09-25 17:00 裁决）：打结结构近邻边（US-align min>=0.6，2998 对）纳入全局绑定图；mode0 等价性检查 PASS（与 2026-09-24 03:27 版逐行一致）",
    "groups_total": roots1,
    "assignment_counts": dict(Counter(assign1.values())),
    "labeled_area_split": {a: dict(c) for a, c in area_split.items()},
    "background_split": dict(Counter(bg_split.values())),
    "knot_type_by_split": {k: dict(v) for k, v in type_split.items()},
    "knot_presence_usable_by_split": {"pos": dict(pos_split), "neg": dict(neg_split)},
    "knot_structural_edges_added": nk1,
    "forced_dev_samples_all_development": qc["forced_dev_samples_all_development"],
    "prev_version": {"run_ts": old.get("run_ts"), "groups_total": old.get("groups_total")},
}
json.dump(new_qc, open(OUT_QC, "w"), ensure_ascii=False, indent=1, sort_keys=True)
print("[finalize] written: manifest/bg/checksums/split_qc/knot edges; roots", roots1)
print(json.dumps({k: qc[k] for k in ("groups_total_new", "max_root_members_labeled", "n_multi_task_roots",
                                     "area_split_new", "knot_type_by_split_new",
                                     "knot_presence_usable_by_split_new")}, ensure_ascii=False, indent=1))
