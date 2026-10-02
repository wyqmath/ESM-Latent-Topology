#!/usr/bin/env python3
"""P2.02：生成发现/确认/保留集 manifest（冻结参数执行）。

输入：data/splits/group_map.tsv（4,858 有标注样本→4,357 组）、data/interim/p202/l2_cluster.tsv
（背景 A+192 端点聚类）、p201_step1_structural_edges（不用于背景）、
exposure_scope_audit.tsv（S1/S2 历史方法选择暴露→强制 development）、
fold_switch_matched_controls.tsv（matched_set 绑定）。
规则（全部已冻结，split_protocol.yaml 2026-09-24 版）：
  A. 组为分配原子；比例 70:15:15（development:confirmation:final_holdout）；种子 2026。
  B. 含 S1/S2（旧 v1–v5 任一划分表收录）样本的组 → 强制 development（历史方法选择暴露不得
     重新包装成未见确认/保留样本；S3 legacy/S4 构建使用=历史接触记录，不剥夺资格——用户
     2026-09-24 限定 4 裁决）。
  C. matched_set 绑定：病例组与对照所在背景簇组同集合。
  D. 受控三系统（案例级）组 → development（干预案例研究，不承担泛化确认角色——如实命名）。
  E. 发现集内部 5 折分组隔离折（早停/选层/调参），种子 2026，仅 development 内部。
输出：data/splits/split_manifest.tsv（有标注样本）；data/splits/fs_l2_background_manifest.tsv.gz
（背景 A 每蛋白组+集合）；data/splits/checksums.txt；reports/split_manifest_report.md；qc json。
注：聚类宇宙=背景 A 484,179+192 端点+7 个不在 A 的 matched 对照（UniProt REST 补取，
manifest=data/manifests/p202_matched_controls_fasta.tsv）。
不满足比例/可评价量时如实报告，不拆组、不放宽。
"""
import csv
import datetime
import gzip
import hashlib
import json
import os
import random
import sys
from collections import defaultdict, Counter

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
WORK = os.path.join(ROOT, "data/interim/p202")
OUT_M = os.path.join(ROOT, "data/splits/split_manifest.tsv")
OUT_BG = os.path.join(ROOT, "data/splits/fs_l2_background_manifest.tsv.gz")
OUT_CK = os.path.join(ROOT, "data/splits/checksums.txt")
OUT_R = os.path.join(ROOT, "reports/split_manifest_report.md")
QC = os.path.join(ROOT, "data/splits/split_qc.json")
SEED = 2026
RUN_TS_STR = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")


def die(m):
    print(f"[p202 FATAL] {m}", file=sys.stderr)
    sys.exit(1)


def rd(p, d="\t"):
    with open(p, newline="") as f:
        return list(csv.DictReader(f, delimiter=d))


# ---------- 1) 有标注样本的全局组（P2.01 冻结产物） ----------
gm = rd(os.path.join(ROOT, "data/splits/group_map.tsv"))
gm_by_sid = {r["sample_id"]: r for r in gm}
sample_group = {r["sample_id"]: r["group_id"] for r in gm}
sample_area = {r["sample_id"]: r["task_area"] for r in gm}

# ---------- 2) 背景聚类 → 背景组（与端点绑定） ----------
cluster = {}
try:
    with open(os.path.join(WORK, "l2_cluster.tsv")) as f:
        for line in f:
            c, m = line.rstrip("\n").split("\t")
            cluster[m] = c
except FileNotFoundError:
    die("l2_cluster.tsv 缺失——先跑 run_p202_background_cluster.sh")
ep_of_pair = {}
for r in rd(os.path.join(ROOT, "data/curated/fold_switch_global.tsv")):
    for side, pdb, ch in (("A", r["pdb_a"], r["chain_a"]), ("B", r["pdb_b"], r["chain_b"])):
        ep_of_pair[f"EP_{r['pair_id']}_{side}__{pdb.lower()}{ch.upper()}"] = r["pair_id"]

# 统一节点空间：labeled 组节点 G::<gid>；背景簇节点 C::<rep>；边=端点∈簇、matched_set 绑定
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
bind_edges = 0
for ep, pair in ep_of_pair.items():
    rep = cluster.get(ep)
    if rep is None:
        die(f"端点不在聚类中: {ep}")
    union(f"G::{sample_group[pair]}", f"C::{rep}")
    bind_edges += 1

# P2.03 审计驱动补绑（2026-09-24）：同 UniProt 与同序列 sha 的有标注↔背景绑定
acc2rep = {}
for member, rep in cluster.items():
    acc2rep.setdefault(member, rep)
n_bind_uni = n_bind_sha = 0
for sid, g in sample_group.items():
    row = gm_by_sid[sid]
    for u in filter(None, row["uniprots"].split(";")):
        rep = acc2rep.get(u)
        if rep is not None:
            union(f"G::{g}", f"C::{rep}")
            n_bind_uni += 1
B = "/Users/yuan/Documents/Codex/2026-09-08/jie/benchmark_step1"
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
                before = (find(f"G::{g}"), find(f"C::{rep}"))
                union(f"G::{g}", f"C::{rep}")
                if before != (find(f"G::{g}"), find(f"C::{rep}")):
                    n_bind_sha += 1

# matched_set 绑定：对照蛋白（背景）簇组 ↔ 病例组
mc = rd(os.path.join(ROOT, "data/curated/fold_switch_matched_controls.tsv"))
n_ms = 0
for r in mc:
    if r["ratio"] == "1:0" or r["control_accession"] in ("", "-", "NONE_ELIGIBLE"):
        continue
    rep = cluster.get(r["control_accession"])
    if rep is None:
        die(f"matched 对照不在背景聚类: {r['control_accession']}")
    union(f"G::{sample_group[r['positive_pair']]}", f"C::{rep}")
    n_ms += 1

# ---------- 3) 历史方法选择暴露（S1/S2）→ 组强制 development ----------
exp = {r["pair_id"]: r for r in rd(os.path.join(ROOT, "reports/fs_three_layer/exposure_scope_audit.tsv"))}
forced_dev = set()
for pid, e in exp.items():
    if e["classification"] in ("v5_split_member", "earlier_split_only"):
        forced_dev.add(f"G::{sample_group[pid]}")
# 受控三系统 → development（案例级）
for sid, area in sample_area.items():
    if area in ("pyp", "rnase_a", "bpti"):
        forced_dev.add(f"G::{sample_group[sid]}")

# ---------- 4) 组级分配 70:15:15（种子 2026；确定性） ----------
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

# ---------- 5) 输出 manifest（有标注样本） ----------
# 发现集内部 5 折：development 组按排序后 i%5 轮转（确定性；组为原子；无随机数参与）
dev_group_nodes = sorted({find(f"G::{gid}") for gid in sample_group.values()
                          if assign[find(f"G::{gid}")] == "development"})
fold_of_group = {}
for i, gnode in enumerate(dev_group_nodes):
    fold_of_group[gnode] = i % 5
with open(OUT_M, "w", newline="") as f:
    w = csv.writer(f, delimiter="\t", lineterminator="\n")
    w.writerow(["sample_id", "task_area", "group_id", "split", "dev_fold"])
    for sid in sorted(sample_group):
        gnode = find(f"G::{sample_group[sid]}")
        sp = assign[gnode]
        fold = fold_of_group.get(gnode, "") if sp == "development" else ""
        w.writerow([sid, sample_area[sid], sample_group[sid], sp, fold])

# ---------- 6) 背景manifest（gzip） ----------
with gzip.open(OUT_BG, "wt", newline="") as f:
    w = csv.writer(f, delimiter="\t", lineterminator="\n")
    w.writerow(["uniprot_accession", "cluster_rep", "group_node", "split"])
    for rep in sorted(bg_members):
        gnode = find(f"C::{rep}")
        for member in sorted(bg_members[rep]):
            w.writerow([member, rep, gnode, assign[gnode]])

# ---------- 7) 一致性断言与统计 ----------
# 断言1：任一 FS pair 两端点同集合（sample=pair 天然）；matched_set 组同集合
ms_sets = defaultdict(set)
for r in mc:
    if r.get("control_accession") not in ("", "-", "NONE_ELIGIBLE") and r["ratio"] != "1:0":
        gnode = find(f"G::{sample_group[r['positive_pair']]}")
        crep = cluster[r["control_accession"]]
        ms_sets[r["matched_set_id"]].add(assign[gnode])
        ms_sets[r["matched_set_id"]].add(assign[find(f"C::{crep}")])
bad_ms = {k: v for k, v in ms_sets.items() if len(v) > 1}
if bad_ms:
    die(f"matched_set 跨集合: {bad_ms}")
# 断言2：forced_dev 组全部 development
for gnode in forced_dev:
    root = find(gnode)
    if assign[root] != "development":
        die(f"forced_dev 组被分配到 {assign[root]}: {gnode}")

area_split = defaultdict(Counter)
for sid in sample_group:
    gnode = find(f"G::{sample_group[sid]}")
    area_split[sample_area[sid]][assign[gnode]] += 1
bg_split = Counter(assign[find(f"C::{rep}")] for rep in bg_members)
knot_type_by_split = defaultdict(Counter)
kt = {r["record_id"]: r for r in rd(os.path.join(ROOT, "data/curated/knots.tsv"))}
for sid in sample_group:
    if sample_area[sid] == "knot":
        rec = kt.get(sid.split(":", 1)[1])
        if rec and rec["type_task_tier"] == "eligible":
            knot_type_by_split[assign[find(f"G::{sample_group[sid]}")]][rec["c2_primary"]] += 1

qc = {
    "run_ts": RUN_TS_STR,
    "seed": SEED, "proportions": "70:15:15",
    "groups_total": len(groups_nodes),
    "assignment_counts": dict(Counter(assign.values())),
    "forced_dev_groups": len(forced_dev),
    "binding_edges": {"endpoint_cluster": bind_edges, "matched_set": n_ms,
                      "labeled_uniprot_to_background": n_bind_uni,
                      "endpoint_sha_to_background": n_bind_sha},
    "labeled_area_split": {a: dict(c) for a, c in area_split.items()},
    "background_split": dict(bg_split),
    "knot_type_by_split": {k: dict(v) for k, v in knot_type_by_split.items()},
    "matched_sets_checked": len(ms_sets),
}
with open(QC, "w") as f:
    json.dump(qc, f, ensure_ascii=False, indent=1, sort_keys=True)

# ---------- 8) checksums ----------
def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()
with open(OUT_CK, "w") as f:
    for p in (OUT_M, OUT_BG):
        f.write(f"{sha(p)}  {os.path.relpath(p, ROOT)}\n")

# ---------- 9) 报告 ----------
dev = sum(1 for v in assign.values() if v == "development")
conf = sum(1 for v in assign.values() if v == "confirmation")
hold = sum(1 for v in assign.values() if v == "final_holdout")
L = ["# P2.02 划分 manifest 报告（冻结参数执行）", "",
     f"时间：{RUN_TS_STR}；种子 2026；比例目标 70:15:15（组级）。", "",
     "## 分配结果",
     f"- 组总数 {len(groups_nodes)}；development {dev}（{dev/n:.1%}）/ confirmation {conf}（{conf/n:.1%}） / final_holdout {hold}（{hold/n:.1%}）。",
     f"- 强制 development 的组 {len(forced_dev)} 个（S1/S2 历史方法选择暴露 + 受控三系统案例组）。",
     f"- 绑定边：端点∈背景簇 {bind_edges}；matched_set {n_ms}（全部同集合断言通过）。",
     "", "## 各任务可评价量（每集合）",
     json.dumps({a: dict(c) for a, c in area_split.items()}, ensure_ascii=False, indent=1),
     "", "## knots type 任务按集合类构成",
     json.dumps({k: dict(v) for k, v in knot_type_by_split.items()}, ensure_ascii=False, indent=1),
     "", "## L2 背景（fs_l2_background_manifest.tsv.gz）",
     f"- 背景蛋白+FS 端点统一聚类；各集合簇数/蛋白分布={dict(bg_split)}。",
     "", "## 纪律与边界",
     "- 组为原子；未拆组、未放宽门槛；S1/S2 暴露组强制 development（历史方法选择不重包装为未见）。",
     "- S3 legacy/S4 构建使用=历史接触记录（用户 2026-09-24 限定 4），不剥夺 confirmation/holdout 资格；正式模型评价历史=0（S5）。",
     "- knots 结构近邻边未纳入绑定（登记残留，P2.03 Foldseek 补算复核；补算后如触发跨集合合并须重跑划分并保留旧版本）。",
     "- confirmation/final_holdout 标签读取纪律：first_read 登记制（P5.01/P5.02）；本任务只生成 ID 映射，未读取任何标签进方法。"]
with open(OUT_R, "w") as f:
    f.write("\n".join(L) + "\n")
print(f"[p202] groups={len(groups_nodes)} dev/conf/hold={dev}/{conf}/{hold} "
      f"forced_dev={len(forced_dev)} ms_bound={n_ms} ep_bind={bind_edges}")
