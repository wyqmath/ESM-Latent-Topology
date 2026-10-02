#!/usr/bin/env python3
"""P2.03：泄露与预训练曝光审计。

检验（全部基于冻结产物，只读）：
  L1 同 UniProt 跨集合（有标注样本 uniprots ∩ 背景集合 accession）；
  L2 同序列 sha 跨集合（背景 A sha 唯一性 + FS 端点序列 sha 与背景比对）；
  L3 FS pair/matched_set 跨集合（复验 P2.02 断言）；
  L4 结构近邻残余：max 口径（单侧相似）跨集合边清点（min 口径下无绑定边跨界——单侧相似为已登记残余）；
  L5 dev_fold 仅 development。
预训练曝光：按候选模型登记 已知重叠/未发现重叠/无法核实 三态（当前无正式模型实验，逐蛋白不可
核实部分如实保留为局限；knots 任务源=PDB，结构感知模型的 PDB 结构收录风险单列）。
"""
import csv
import datetime
import gzip
import hashlib
import json
import os
import sys
from collections import defaultdict, Counter

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
OUT = os.path.join(ROOT, "reports/leakage_audit.md")
QC = os.path.join(ROOT, "reports/leakage_audit_qc.json")
RUN_TS = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")


def die(m):
    print(f"[p203 FATAL] {m}", file=sys.stderr)
    sys.exit(1)


def rd(p, d="\t"):
    with open(p, newline="") as f:
        return list(csv.DictReader(f, delimiter=d))


# ---- 载入冻结划分 ----
man = rd(os.path.join(ROOT, "data/splits/split_manifest.tsv"))
split_of = {r["sample_id"]: r["split"] for r in man}
area_of = {r["sample_id"]: r["task_area"] for r in man}
gm = {r["sample_id"]: r for r in rd(os.path.join(ROOT, "data/splits/group_map.tsv"))}
bg_split = {}
with gzip.open(os.path.join(ROOT, "data/splits/fs_l2_background_manifest.tsv.gz"), "rt") as f:
    for r in csv.DictReader(f, delimiter="\t"):
        bg_split[r["uniprot_accession"]] = r["split"]

problems = []

# ---- L1 同 UniProt 跨集合 ----
uni_splits = defaultdict(set)
for sid, r in gm.items():
    sp = split_of[sid]
    for u in filter(None, r["uniprots"].split(";")):
        uni_splits[u].add(sp)
cross_uni_labeled = {u: s for u, s in uni_splits.items() if len(s) > 1}
cross_uni_vs_bg = {u: (sorted(s), bg_split[u]) for u, s in uni_splits.items()
                   if u in bg_split and bg_split[u] not in s}
if cross_uni_labeled:
    problems.append(f"L1 有标注跨集合 UniProt: {len(cross_uni_labeled)}")
if cross_uni_vs_bg:
    problems.append(f"L1 有标注-背景跨集合 UniProt: {len(cross_uni_vs_bg)}")

# ---- L2 序列 sha ----
bg_sha = {}
with gzip.open(os.path.join(ROOT, "data/curated/fold_switch_unlabeled_sequence.tsv.gz"), "rt") as f:
    for r in csv.DictReader(f, delimiter="\t"):
        bg_sha.setdefault(r["sequence_sha256"], set()).add(bg_split.get(r["uniprot_accession"], "?"))
bad_bg_sha = {k: v for k, v in bg_sha.items() if len(v - {"?"}) > 1}
if bad_bg_sha:
    problems.append(f"L2 背景 sha 跨集合: {len(bad_bg_sha)}")
# FS 端点序列 sha vs 背景（端点序列与背景同序列=直接重叠通道）
B = "/Users/yuan/Documents/Codex/2026-09-08/jie/benchmark_step1"
summ = {(r["pdb_id"].lower(), r["requested_chain"].upper()): r
        for r in rd(B + "/manifests/fold_pair_endpoint_sequence_structure_summary.tsv")}
ep_sha_split = defaultdict(set)
for r in rd(os.path.join(ROOT, "data/curated/fold_switch_global.tsv")):
    sp = split_of[r["pair_id"]]
    for side, pdb, ch in (("A", r["pdb_a"], r["chain_a"]), ("B", r["pdb_b"], r["chain_b"])):
        seq = summ[(pdb.lower(), ch.upper())]["observed_sequence"]
        ep_sha_split[hashlib.sha256(seq.encode()).hexdigest()].add(sp)
# sha 重叠本身=P1.10 设计内（pending/extension 端点可在背景）；泄露=重叠两侧分属不同集合
ep_in_bg_violation = {}
for sha, sps in ep_sha_split.items():
    if sha not in bg_sha:
        continue
    bg_splits = {x for x in bg_sha[sha] if x != "?"}
    if bg_splits and (bg_splits - sps or sps - bg_splits):
        ep_in_bg_violation[sha] = (sorted(sps), sorted(bg_splits))
if ep_in_bg_violation:
    problems.append(f"L2 FS 端点序列与背景同序列但跨集合: {len(ep_in_bg_violation)}")

# ---- L3 pair/matched_set（复验） ----
pair_ids = [r["sample_id"] for r in man if r["task_area"] == "fold_switch"]
# pair 两端点同属一样本（pair 为样本单元）→ 天然同集合；校验 matched_set
mc = rd(os.path.join(ROOT, "data/curated/fold_switch_matched_controls.tsv"))
ms_bad = []
# 复用 P2.02 逻辑的产物级复验：matched_set 主集全部 dev
for r in mc:
    if r["ratio"] == "1:1_main":
        if split_of[r["positive_pair"]] != "development":
            ms_bad.append(r["matched_set_id"])
if ms_bad:
    problems.append(f"L3 matched_set 主集不在 dev: {ms_bad}")

# ---- L4 max 口径残余（单侧相似跨界清点） ----
# 背景簇=min 绑定后仍可能有单侧 max 相似跨界；此处清点 FS 池内的 max 边跨界情况
edges = rd(os.path.join(ROOT, "reports/fs_three_layer/p201_step1_structural_edges.tsv"))
pair_of_ep = {}
for r in rd(os.path.join(ROOT, "data/curated/fold_switch_global.tsv")):
    for side, pdb, ch in (("A", r["pdb_a"], r["chain_a"]), ("B", r["pdb_b"], r["chain_b"])):
        pair_of_ep[f"EP_{r['pair_id']}_{side}__{pdb.lower()}{ch.upper()}"] = r["pair_id"]
max_cross = []
for e in edges:
    if e["edge_max_0.6"] != "yes" or e["edge_min_0.6"] == "yes":
        continue
    pa, pb = pair_of_ep.get(e["a"]), pair_of_ep.get(e["b"])
    if pa and pb and split_of[pa] != split_of[pb]:
        max_cross.append({"a": pa, "b": pb, "tm_max": e["tm_sym_max"], "tm_min": e["tm_sym_min"]})

# ---- L6 打结结构绑定跨集合（KNOT-RESPLIT 方案 A 新增；US-align min>=0.6 必须同集合） ----
knot_edge_path = os.path.join(ROOT, "data/splits/knot_structural_edges.tsv")
knot_unverified = os.path.join(ROOT, "data/curated/knots_sequences_unavailable.tsv")
l6_bad, l6_checked = [], 0
if os.path.exists(knot_edge_path):
    chain2sid_norm = {}
    for sid in split_of:
        if sid.startswith("knot:"):
            chain2sid_norm[sid.split(":", 1)[1].replace("_", "").lower()] = sid
    for e in rd(knot_edge_path):
        a, b = e["chain_a"], e["chain_b"]
        sa_id = chain2sid_norm.get(a.replace("_", "").lower())
        sb_id = chain2sid_norm.get(b.replace("_", "").lower())
        if sa_id is None or sb_id is None:
            die(f"L6 打结链不在 manifest: {a} {b}")
        l6_checked += 1
        if split_of[sa_id] != split_of[sb_id]:
            l6_bad.append({"a": a, "b": b, "split_a": split_of[sa_id], "split_b": split_of[sb_id]})
else:
    die("L6 knot_structural_edges.tsv 缺失")
n_unverified = sum(1 for _ in open(knot_unverified)) - 1
# 打结宇宙 max 单侧相似（信息性）：US-align 确认表中 max>=0.6>min 的跨集合对
knot_uni_cross = 0
conf_path = os.path.join(ROOT, "data/interim/p303/usalign_confirmed.tsv")
if os.path.exists(conf_path):
    sp_knot = {}
    for sid in split_of:
        if sid.startswith("knot:"):
            sp_knot[sid.split(":", 1)[1].replace("_", "").lower()] = split_of[sid]
    for line in open(conf_path):
        p = line.rstrip("\n").split("\t")
        if len(p) < 3 or not p[2].strip():
            continue
        try:
            tms = [float(x) for x in p[2].split()]
        except ValueError:
            continue
        if len(tms) >= 2 and max(tms) >= 0.6 > min(tms):
            a_key = p[0].replace("_", "").lower()
            b_key = p[1].replace("_", "").lower()
            sa, sb = sp_knot.get(a_key), sp_knot.get(b_key)
            if sa and sb and sa != sb:
                knot_uni_cross += 1

# ---- L5 dev_fold ----
bad_fold = [r["sample_id"] for r in man if r["split"] != "development" and r["dev_fold"] != ""]
if bad_fold:
    problems.append(f"L5 非 dev 样本带 fold: {len(bad_fold)}")
if l6_bad:
    problems.append(f"L6 打结结构绑定跨集合: {len(l6_bad)}")

qc = {
    "run_ts": RUN_TS,
    "checks": {
        "L1_labeled_cross_uniprot": len(cross_uni_labeled),
        "L1_labeled_vs_background_cross_uniprot": len(cross_uni_vs_bg),
        "L2_background_sha_cross_split": len(bad_bg_sha),
        "L2_endpoint_sha_cross_split_violations": len(ep_in_bg_violation),
        "L3_matched_set_violations": len(ms_bad),
        "L4_max_rule_cross_split_edges_fs": len(max_cross),
        "L5_fold_leaks": len(bad_fold),
        "L6_knot_structural_binding_cross_split": len(l6_bad),
        "L6_knot_structural_binding_checked": l6_checked,
        "L6_knot_chains_unverified_no_structure": n_unverified,
        "L4b_knot_unilateral_max06_cross_split": knot_uni_cross,
    },
    "problems": problems,
}
with open(QC, "w") as f:
    json.dump(qc, f, ensure_ascii=False, indent=1, sort_keys=True)

top_max = sorted(max_cross, key=lambda x: -float(x["tm_max"]))[:10]
L = ["# 泄露与预训练曝光审计（P2.03）", "",
     f"时间：{RUN_TS}。对象=冻结划分（split_manifest 4,858 + 背景 484,378；commit 记录于 P2.02）。", "",
     "## 1. 划分完整性检验（问题=0 才通过）",
     f"- L1 同 UniProt 跨集合：有标注内部 {len(cross_uni_labeled)}、有标注↔背景 {len(cross_uni_vs_bg)}（0=通过）。",
     f"- L2 序列 sha：背景内跨集合 {len(bad_bg_sha)}；FS 端点序列与背景精确重叠 {len(set(ep_sha_split) & set(bg_sha))} 处（P1.10 设计内：pending/extension 端点可入背景）——其中跨集合违规 {len(ep_in_bg_violation)}（0=通过；P2.03 已驱动补绑同 UniProt/同序列簇，见 §3）。",
     f"- L3 pair/matched_set：matched 主集全 development 复验={'通过' if not ms_bad else ms_bad}。",
     f"- L4 结构近邻残余（**如实披露**）：min 口径（冻结绑定规则）下跨集合绑定边=0；但 max 口径（单侧相似）在 FS 池内有 {len(max_cross)} 条跨集合边未构成绑定——即存在单侧 TM≥0.6 的跨集合结构相似对，已登记为已知残余（冻结理由见 P2.01；缓解=其 tm_max 值随边表可查，后续如需强绑定可改口径重划分）。最高 10 条：",
     "```json",
     json.dumps(top_max, ensure_ascii=False),
     "```",
     f"- L5 dev_fold 泄露：{len(bad_fold)}（0=通过）。",
     f"- L6 打结结构绑定跨集合（KNOT-RESPLIT 新增）：检查 {l6_checked} 条 US-align min≥0.6 边，跨集合 {len(l6_bad)}（0=通过）；"
     f"无结构链 {n_unverified} 条保持未验证（不当作通过）；打结宇宙跨集合单侧 max≥0.6 残余 {knot_uni_cross} 对（信息性）。", "",
     "## 2. 预训练曝光（三态登记；当前无正式模型实验）",
     "| 通道 | 状态 | 说明 |",
     "|---|---|---|",
     "| 序列 PLM（ESM-2 类，UniRef50/100 语料） | 已知存在同源重叠（数据库级） | 样本序列来自 Swiss-Prot/PDB，其同源序列大概率在 UniRef；逐蛋白是否入训练集无法核实 → 无法核实（逐蛋白） |",
     "| 结构感知 PLM（SaProt 类，含 PDB 结构） | 已知重叠通道（knots 任务尤甚） | knots 任务源=PDB；结构/序列可能被结构语料收录 → 无法核实（逐蛋白） |",
     "| 本项目数据是否进过任何训练 | 未发现（自证） | 项目内部无任何训练/方法选择实验已运行（P2 阶段前无模型）；exposure_log 仅 3 条历史读取登记 |",
     "", "结论措辞边界：下游一切'未见'主张只能指本项目划分语义（组/簇隔离），**不得**指预训练语料未见；未知预训练曝光保留为局限。", "",
     "## 3. 登记残留（阻塞与计划）",
     "- knots 宇宙结构近邻补算（Foldseek 预筛+US-align 确认）：**阻塞**——sblab 集群 2026-09-24 03:22 ssh 超时不可达；恢复后执行（如触发跨集合绑定合并，将重跑 P2.02 划分并保留旧版本）。",
     "- P2.04 模型路由：HF 直连不通（12s 超时实测）；hf-mirror.com 可达（200/0.29s）、pypi 可达——路由方案就绪，待 P2.04 执行。", "",
     "## 4. 缓存/归一化/特征选择范围",
     "- 当前无任何训练、缓存、归一化拟合或特征选择发生（P2.05/P2.06 未开始）；本项为占位核查，训练启动后在 P3 各任务复检。"]
with open(OUT, "w") as f:
    f.write("\n".join(L) + "\n")
print(f"[p203] problems={len(problems)} max_cross={len(max_cross)} report written")
