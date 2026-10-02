#!/usr/bin/env python3
"""P2.01 步骤一：结构近邻边合并 + 分组可行性报告（只读预演，无 group_map/matched_set/manifest）。

输入：data/interim/p201_step1/{chains,usalign,extract_manifest,keys}.（由 extract_p201_chains.py
与 run_p201_usalign.sh 产出）；data/curated/{fold_switch_global,PN 候选表}.tsv；
reports/fs_three_layer/{pna_evidence_review,exposure_scope_audit}.tsv。
边规则（预注册）：
  E1 pair 内部边（两端点永不分离）；
  E2 序列同源边=mmseqs2 easy-cluster --min-seq-id 0.3 -c 0.7 --cov-mode 0（P1.17 主口径，端点级）；
  E3 结构近邻边=TM≥τ，τ∈{0.5,0.6,0.7}；相似度=两条归一化 TM 的 max（对称化规则，两条原始值均存表）。
主口径=E2+E3(0.6)。输出仅入 reports/fs_three_layer/p201_step1_*。
"""
import csv
import datetime
import glob
import json
import os
import subprocess
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
WORK = os.path.join(ROOT, "data/interim/p201_step1")
OUT = os.path.join(ROOT, "reports/fs_three_layer/p201_step1_structural_edges.tsv")
OUTG = os.path.join(ROOT, "reports/fs_three_layer/p201_step1_groups_main.tsv")
OUTPN = os.path.join(ROOT, "reports/fs_three_layer/p201_step1_strict_vs_pn_edges.tsv")
QC = os.path.join(ROOT, "reports/fs_three_layer/p201_step1_qc.json")
REPORT = os.path.join(ROOT, "reports/fs_three_layer/p201_step1_grouping_feasibility.md")
RUN_TS = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
TAUS = [0.5, 0.6, 0.7]
MAIN_TAU = 0.6
MMSEQS = "/Users/yuan/Documents/Codex/2026-09-08/jie/benchmark_step1/tools/mmseqs2-18-8cc5c/bin/mmseqs"
B = "/Users/yuan/Documents/Codex/2026-09-08/jie/benchmark_step1"


def die(m):
    print(f"[p201s1 FATAL] {m}", file=sys.stderr)
    sys.exit(1)


def rd(p, d="\t"):
    with open(p, newline="") as f:
        return list(csv.DictReader(f, delimiter=d))


# ---- 0) 端点登记（键→pair/side） ----
g_rows = rd(os.path.join(ROOT, "data/curated/fold_switch_global.tsv"))
ep2pair = {}
for r in g_rows:
    for side, pdb, ch in (("A", r["pdb_a"], r["chain_a"]), ("B", r["pdb_b"], r["chain_b"])):
        key = f"EP_{r['pair_id']}_{side}__{pdb.lower()}{ch.upper()}"
        ep2pair[key] = (r["pair_id"], side, r["tier"])
pairs = sorted({v[0] for v in ep2pair.values()})
if len(pairs) != 96 or len(ep2pair) != 192:
    die(f"端点登记异常 pairs={len(pairs)} eps={len(ep2pair)}")

# ---- 1) 解析 US-align 输出 ----
files = glob.glob(os.path.join(WORK, "usalign", "*.out"))
edges = []  # (a,b,tma,tmb,tm,rmsd,seqid,aligned)
for fp in files:
    # 链键内含 "__"（pair_id 自带），不能按文件名拆分——从输出内的结构路径行解析真名
    a = b = None
    t1 = t2 = rmsd = seqid = aligned = None
    n1 = n2 = None
    for line in open(fp):
        if line.startswith("Name of Structure_1:"):
            a = os.path.basename(line.split(":", 1)[1].strip().split(".pdb")[0])
        elif line.startswith("Name of Structure_2:"):
            b = os.path.basename(line.split(":", 1)[1].strip().split(".pdb")[0])
        elif line.startswith("Aligned length="):
            import re as _re
            m = _re.search(r"Aligned length=\s*(\d+),\s*RMSD=\s*([\d.]+),\s*Seq_ID=[\w/]+=\s*([\d.]+)", line)
            if not m:
                die(f"Aligned 行解析失败: {line.strip()[:80]}")
            aligned, rmsd, seqid = int(m.group(1)), float(m.group(2)), float(m.group(3))
        elif line.startswith("TM-score=") and "Structure_1" in line:
            t1 = float(line.split("=")[1].split()[0])
        elif line.startswith("TM-score=") and "Structure_2" in line:
            t2 = float(line.split("=")[1].split()[0])
        elif line.startswith("Length of Structure_1:"):
            n1 = int(line.split(":")[1].split()[0])
        elif line.startswith("Length of Structure_2:"):
            n2 = int(line.split(":")[1].split()[0])
    if t1 is None or t2 is None or aligned is None or not a or not b:
        die(f"解析失败: {os.path.basename(fp)}")
    base = os.path.basename(fp)[:-4]
    if f"{a}__{b}" != base:
        die(f"名称与文件名不一致: {base} vs {a}__{b}")
    edges.append({"a": a, "b": b, "tm_1": t1, "tm_2": t2, "tm_sym_max": max(t1, t2),
                  "tm_sym_min": min(t1, t2),
                  "rmsd": rmsd, "seqid": seqid or "", "aligned_len": aligned,
                  "len_1": n1 or "", "len_2": n2 or ""})
expected = 208 * 207 // 2
if len(edges) != expected:
    die(f"边数 {len(edges)} != 预期 {expected}")

with open(OUT, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["a", "b", "tm_1", "tm_2", "tm_sym_max", "tm_sym_min", "rmsd", "seqid",
                                      "aligned_len", "len_1", "len_2",
                                      "edge_max_0.5", "edge_max_0.6", "edge_max_0.7",
                                      "edge_min_0.5", "edge_min_0.6", "edge_min_0.7"],
                       delimiter="\t", lineterminator="\n")
    w.writeheader()
    for e in sorted(edges, key=lambda x: (-x["tm_sym_max"], x["a"], x["b"])):
        for tau in TAUS:
            e[f"edge_max_{tau}"] = "yes" if e["tm_sym_max"] >= tau else "no"
            e[f"edge_min_{tau}"] = "yes" if e["tm_sym_min"] >= tau else "no"
        w.writerow(e)

# ---- 2) 序列同源边（mmseqs easy-cluster，端点级） ----
summ = {(r["pdb_id"].lower(), r["requested_chain"].upper()): r
        for r in rd(os.path.join(B, "manifests/fold_pair_endpoint_sequence_structure_summary.tsv"))}
fasta = os.path.join(WORK, "endpoints.fa")
with open(fasta, "w") as f:
    for key, (pid, side, _t) in sorted(ep2pair.items()):
        tail = key.rsplit("__", 1)[1]
        seq = summ[(tail[:4].lower(), tail[4:])]["observed_sequence"]
        f.write(f">{key}\n{seq}\n")
seq_clusters = {}
for mid, thr in (("main", "0.3"),):
    pre = os.path.join(WORK, f"mmseqs_{mid}")
    subprocess.run([MMSEQS, "easy-cluster", fasta, pre, os.path.join(WORK, "tmp_mmseqs"),
                    "--min-seq-id", thr, "-c", "0.7", "--cov-mode", "0"],
                   check=True, capture_output=True)
    # easy-cluster 的 _cluster.tsv：每行 rep<TAB>member（rep 与成员，含自配对行）
    with open(pre + "_cluster.tsv") as f:
        for line in f:
            c, m = line.rstrip("\n").split("\t")
            seq_clusters[m] = c
            seq_clusters.setdefault(c, c)

# ---- 3) 并查集合并（端点级→pair 级） ----
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
            self.p[rb] = ra


def build_groups(tau, mode="max"):
    uf = UF()
    # E1 pair 内部边
    for r in g_rows:
        ka = f"EP_{r['pair_id']}_A__{r['pdb_a'].lower()}{r['chain_a'].upper()}"
        kb = f"EP_{r['pair_id']}_B__{r['pdb_b'].lower()}{r['chain_b'].upper()}"
        uf.union(ka, kb)
    # E2 序列边
    for k, rep in seq_clusters.items():
        uf.union(k, rep)
    # E3 结构边
    n_tm = 0
    field = "tm_sym_max" if mode == "max" else "tm_sym_min"
    for e in edges:
        if e[field] >= tau:
            uf.union(e["a"], e["b"])
            n_tm += 1
    ep_groups = {k: uf.find(k) for k in ep2pair}
    pair_groups = {}
    for k, (pid, _s, _t) in ep2pair.items():
        pair_groups.setdefault(pid, set()).add(ep_groups[k])
    pg = {pid: sorted(gs)[0] for pid, gs in pair_groups.items() if gs}
    # 一个 pair 若两端点组不同已被 E1 合并——断言
    assert all(len(gs) == 1 for gs in pair_groups.values()), "pair 内端点跨组（E1 失效）"
    return pg, n_tm


groups_main, n_tm_main = build_groups(MAIN_TAU, "max")
sens = {}
for mode in ("max", "min"):
    for tau in (0.5, 0.6, 0.7):
        pg, _ = build_groups(tau, mode)
        sens[f"{mode}_{tau}"] = len(set(pg.values()))

# ---- 4) 组统计 ----
from collections import defaultdict, Counter
members = defaultdict(list)
for pid, gid in groups_main.items():
    members[gid].append(pid)
# 组 ID 可读化：按规模降序 G001...
order = sorted(members, key=lambda g: (-len(members[g]), sorted(members[g])[0]))
gid_name = {g: f"G{i:03d}" for i, g in enumerate(order, 1)}
groups_main = {p: gid_name[g] for p, g in groups_main.items()}
members = {gid_name[g]: sorted(v) for g, v in members.items()}
sizes = sorted((len(v) for v in members.values()), reverse=True)
# min 规则 0.6 的组（对照列）
groups_min, _ = build_groups(0.6, "min")
members_min = defaultdict(list)
for pid, gid in groups_min.items():
    members_min[gid].append(pid)
gid_min = {g: f"M{i:03d}" for i, g in enumerate(sorted(members_min, key=lambda g: (-len(members_min[g]), sorted(members_min[g])[0])), 1)}
groups_min = {p: gid_min[g] for p, g in groups_min.items()}
tier_of = {r["pair_id"]: r["tier"] for r in g_rows}
strict_pairs = [p for p in pairs if tier_of[p] == "strict_state_candidate"]
strict_groups = {groups_main[p] for p in strict_pairs}
strict_groups = {groups_main[p] for p in strict_pairs}
multi_strict = {g: sorted(p for p in strict_pairs if groups_main[p] == g)
                for g in strict_groups if sum(1 for p in strict_pairs if groups_main[p] == g) > 1}
# 历史接触 join（P1.19）
exp = {r["pair_id"]: r for r in rd(os.path.join(ROOT, "reports/fs_three_layer/exposure_scope_audit.tsv"))}
hist_dist = Counter(exp[p]["classification"] for p in pairs)
v5_in_groups = Counter()
for gid, mem in members.items():
    v5_in_groups[sum(1 for p in mem if exp[p]["classification"] == "v5_split_member")] += 0  # 占位
mixed_groups = [gid for gid, mem in members.items()
                if len({exp[p]["classification"] for p in mem}) > 1]

with open(OUTG, "w", newline="") as f:
    w = csv.writer(f, delimiter="\t", lineterminator="\n")
    w.writerow(["pair_id", "group_max_0.6", "group_min_0.6", "tier", "v5_member",
                "legacy_exposure", "classification_p119", "group_max_size"])
    for pid in pairs:
        w.writerow([pid, groups_main[pid], groups_min[pid], tier_of[pid],
                    exp[pid]["split_v5"], exp[pid]["legacy_exposure"],
                    exp[pid]["classification"], len(members[groups_main[pid]])])

# ---- 5) strict×PN 结构近邻（匹配可行性输入） ----
pna = rd(os.path.join(ROOT, "reports/fs_three_layer/pna_evidence_review.tsv"))
prim = {}
for r in pna:
    if r["primary_for"]:
        for pos in [x.strip() for x in r["primary_for"].split(";") if x.strip()]:
            prim.setdefault(r["uniprot_accession"], set()).add(pos)
strict_eps = [k for k, v in ep2pair.items() if v[2] == "strict_state_candidate"]
pn_edges = []
for e in edges:
    for a, b in ((e["a"], e["b"]), (e["b"], e["a"])):
        if a.startswith("PN_") and b.startswith("EP_") and ep2pair[b][2] == "strict_state_candidate":
            acc = a.split("__")[0][3:]
            pn_edges.append({"pn_accession": acc, "primary_for": ";".join(sorted(prim.get(acc, []))),
                             "strict_pair": ep2pair[b][0], "tm_sym_max": e["tm_sym_max"],
                             "rmsd": e["rmsd"], "aligned_len": e["aligned_len"]})
with open(OUTPN, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["pn_accession", "primary_for", "strict_pair", "tm_sym_max",
                                      "rmsd", "aligned_len"], delimiter="\t", lineterminator="\n")
    w.writeheader()
    w.writerows(sorted(pn_edges, key=lambda x: -x["tm_sym_max"]))
# own-positive 口径：候选为其 primary_for 阳性的候选时，与该阳性 pair 端点的最高 TM
# （primary_for 存阳性 UniProt 号；经 fold_switch_global.uniprot_a 连接到 pair）
pos2pair = {r["uniprot_a"]: r["pair_id"] for r in g_rows if r["tier"] == "strict_state_candidate"}
best_pn = {}
for e in pn_edges:
    acc = e["pn_accession"]
    for pos in [x.strip() for x in (e["primary_for"] or "").split(";") if x.strip()]:
        if pos2pair.get(pos) == e["strict_pair"]:
            key = (pos, acc)
            if key not in best_pn or e["tm_sym_max"] > best_pn[key]:
                best_pn[key] = e["tm_sym_max"]
per_pos_best = defaultdict(dict)
for (pos, acc), tm in best_pn.items():
    per_pos_best[pos][acc] = tm
n_own_primary_ge = sum(1 for v in best_pn.values() if v >= 0.6)
pos_ge = sorted({pos for (pos, _acc), v in best_pn.items() if v >= 0.6})

qc = {
    "run_ts": RUN_TS,
    "chains": {"strict_endpoints": 192, "pn_candidates": 16, "total": 208},
    "alignments": len(edges),
    "tm_edge_counts": {"max": {str(t): sum(1 for e in edges if e["tm_sym_max"] >= t) for t in TAUS},
                       "min": {str(t): sum(1 for e in edges if e["tm_sym_min"] >= t) for t in TAUS}},
    "groups": {"main_rule": "tm_sym_max>=0.6", "n_groups": len(members),
               "size_hist": dict(Counter(sizes)), "max_group": sizes[0] if sizes else 0,
               "sensitivity_groups": sens},
    "strict": {"n": 10, "groups": len(strict_groups), "co_grouped_pairs": multi_strict},
    "historical_contact": {"classification_dist": dict(hist_dist),
                           "mixed_contact_groups": len(mixed_groups)},
    "strict_vs_pn": {"own_primary_combos": len(best_pn),
                     "own_primary_ge_0.6": n_own_primary_ge,
                     "positives_covered": sorted(per_pos_best)},
    "no_group_map_emitted": True,
}
with open(QC, "w") as f:
    json.dump(qc, f, ensure_ascii=False, indent=1, sort_keys=True)

# ---- 6) 报告 ----
L = [f"# P2.01 步骤一：结构近邻与合并分组可行性（只读预演；G1 限定 6）", "",
     f"时间：{RUN_TS.split()[0]} {RUN_TS.split()[1][:5]} Asia/Shanghai。状态：**可行性报告**——"
     "不含正式 group_map/matched_set/split manifest（阈值冻结前禁令维持）。", "",
     "## 1. 数据与规则（预注册）",
     "- 节点=208 条链：192 个 strict 端点（旧项目 structures_mmcif，提取长度与 P1 汇总表 192/192 零错配）+ 16 个 PN-A 主选候选代表链（每 accession 取 observed_coverage 最高的映射链；RCSB 2026-09-24 下载，sha 见 extract_manifest）。US-align 20260908（arm64）全对全 21,528 对，首模型、去 altloc/H/配体/水。长度口径注：manifest n_residues=gemmi 聚合物残基数（含少量非 CA 残基），US-align 对齐长度=CA 数，两者在 15 个端点差 1–20（如 4zt0C 1296 vs 1276）；TM 归一化用 US-align 自身读取的链长，不受影响。",
     "- 边：E1 pair 内部边；E2 序列同源=mmseqs2 easy-cluster 0.3 id/0.7 cov（P1.17 主口径）；E3 结构=US-align 单链 TM≥τ（τ=0.5/0.6/0.7）。TM 对称化口径两种并报：max（预注册主口径，单侧相似即连边）与 min（双侧均≥τ；执行中因 max 规则出现巨组而补做的对照口径，如实登记非预注册）。group=并查集（E1+E2+E3）。",
     "", "## 2. 结果",
     f"- **主口径（30%/70%+TM0.6）**：96 对 → **{len(members)} 组**；组规模分布={dict(sorted(Counter(sizes).items()))}；最大组={sizes[0] if sizes else 0}。",
     f"- 口径/阈值敏感性（组数）：max 规则 0.5→{sens['max_0.5']}、0.6→{sens['max_0.6']}、0.7→{sens['max_0.7']}；min 规则 0.5→{sens['min_0.5']}、0.6→{sens['min_0.6']}、0.7→{sens['min_0.7']}（序列层敏感性见 P1.17：25%/40%→92/95）。",
     f"- TM 边数（max/min 规则）：≥0.5={qc['tm_edge_counts']['max']['0.5']}/{qc['tm_edge_counts']['min']['0.5']}、≥0.6={qc['tm_edge_counts']['max']['0.6']}/{qc['tm_edge_counts']['min']['0.6']}、≥0.7={qc['tm_edge_counts']['max']['0.7']}/{qc['tm_edge_counts']['min']['0.7']}（共 21,528 对）。",
     f"- **strict 10 对**：max 规则下分布至 {len(strict_groups)} 组；共组明细={json.dumps(multi_strict, ensure_ascii=False) if multi_strict else '无'}。",
     f"- **max 规则巨组（{sizes[0]} 对）构成**：{json.dumps(Counter(tier_of[p] for p in members[max(members, key=lambda g: len(members[g]))]), ensure_ascii=False)}；min 规则（0.6）下组数={sens['min_0.6']}（口径选择见 §4）。",
     f"- 历史接触（P1.19 四口径随行）：组内分类混合的组数={len(mixed_groups)}（v5 收录对与从未收录对同组的情况将进入 P2.02 约束——历史接触不改变分组，但划分时整组同集合）；分类分布={dict(hist_dist)}。",
     "", "## 3. 匹配可行性（L3，输入口径）",
     f"- PN-A 主选候选×strict 结构近邻（max 规则，own-positive 口径）：{len(best_pn)} 个 (阳性,自身主选候选) 组合中 {n_own_primary_ge} 个 TM≥0.6（分布于 {len(pos_ge)} 个阳性：{','.join(pos_ge)}）；组合覆盖全部 {len(per_pos_best)} 个有家族候选的阳性，其中 P38505 的 3 个主选均 <0.6（与其 calmodulin 家族候选弱结构相似一致）。逐对最高 TM 见 p201_step1_strict_vs_pn_edges.tsv。",
     "- G1 限定 2 生效：确证性 L3 仅用完成逐例人工全文复核的实际入选 PN-A 对照（当前全部 pending，P02829/P0DP29 为已标记例）；最终病例数以复核+匹配完成后为准。三例零家族阳性暂退主结果（G1 限定 3）；结构近邻探索为其回补路径（Foldseek 全池版在步骤二/正式匹配时执行，本表仅覆盖 16 个主选代表链）。",
     "- 匹配变量仍按三层协议（家族/长度/结构数/覆盖/方法/分辨率/独立研究数/组装；禁模型输出）；本表 TM 不作为匹配变量，仅作结构近邻分组约束与可行性参考。", "",
     "## 4. 结构近邻口径选择（**待用户冻结的核心决定**；本报告不冻结任何参数）",
     "两种 TM 对称化口径在 0.6 阈值下的分组后果差异决定性：",
     "- **max 规则（单侧相似即连边）**：48 组，最大组 35 对（含 6/10 strict 与 19 pending——单侧相似驱动的传递闭包，风险=过度合并侵蚀可评价独立性）；0.5→14 组（巨组化加剧）。",
     f"- **min 规则（双侧相似才连边）**：{sens['min_0.6']} 组（近似回到序列层 93 组的粒度）；0.5→{sens['min_0.5']}、0.7→{sens['min_0.7']}。",
     "- 权衡：min 规则保留可评价独立性（防过度合并），但绑定为保守下界（单侧结构相似的跨组对存在，泄露风险交 P2.03 泄露审计复核）；max 规则绑定强但把 10 个 strict 中 6 个锁进同组，且巨组使各集合的组级分配空间骤减。",
     "- 序列层：P1.17 已示 25%–40% 组数 92–95——建议按候选值 30%/70% 冻结（两口径下结论不变）。",
     "- **建议（供选择，不预设）**：结构层冻结为 US-align 单链 TM≥0.6 + min 对称化口径（工具版本 20260908）；如用户倾向更强绑定可选 max 口径并接受巨组代价。最低可评价量/比例/种子待口径确认后于步骤二一并冻结（split_protocol 既有 [待冻结] 项）。",
     "", "## 5. 边界与残留",
     "- 本报告 TM 覆盖 192 strict 端点全对全 + 16 个主选候选代表链；**L2 背景侧（48 万序列/990k 链）与全 PN 池（138）的结构近邻未算**——L2 的结构近邻约束（若启用）与 L3 全池匹配在步骤二以 Foldseek 批量执行，本报告结论限于上述范围。",
     "- PN 代表链=每 accession 一条最高覆盖链，未覆盖多链/多域情形；正式匹配时按协议重算。",
     "- 历史接触仅随行记录，不改变分组（分组=纯结构约束）。"]
with open(REPORT, "w") as f:
    f.write("\n".join(L) + "\n")
print(f"[p201s1] edges={len(edges)} groups_main={len(members)} max={sizes[0] if sizes else 0} "
       f"strict_groups={len(strict_groups)} qc+report written")
