#!/usr/bin/env python3
"""P2.01 步骤二：正式分组（全局 group_map + fs_group_edges）与 L3 主匹配（matched_controls）。

冻结参数（logs/decisions.md 2026-09-24 02:34，用户委托授权）：
  序列=mmseqs 0.30/0.70 cov-mode 0；结构=US-align 20260908 单链 TM≥0.60 **min 对称化**；
  比例 70:15:15（组级，P2.02 使用）；种子 2026（P2.02 使用）。
范围（预注册）：
  group_map=有标注任务样本（FS 96 对/knots 1401 链/disorder 蛋白/受控三系统样本行）；
    背景 A/B（未标注宇宙）不在 group_map，其分组在 P2.02 L2 manifest 以同一冻结参数生成。
  跨任务绑定=UniProt 交集（knots 链经 SIFTS pdb_chain_uniprot 映射，可能多映射→交集语义）；
  FS 边=E1 pair 内+E2 序列簇（跨对）+E3 TM min≥0.6（跨对），全部来自步骤一冻结产物。
匹配（L3 主集）：病例=6 个有可用 PN-A 的阳性（P19726 池无 PN-A→不入主集；3 零家族=G1 限定 3 暂退）；
  对照资格=pn_tier==PN-A 且 P1.16 post_check_usable=='yes'；选择=每病例取 match_rank 最小者；
  1:1 主集 + ≥3 资格者的 1:3 敏感性集；全部对照 review_status=pending_manual_fulltext（G1 限定 2）。
"""
import csv
import datetime
import gzip
import json
import os
import sys
from collections import defaultdict, Counter

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
WORK = os.path.join(ROOT, "data/interim/p201_step1")
OUT_GM = os.path.join(ROOT, "data/splits/group_map.tsv")
OUT_E = os.path.join(ROOT, "data/splits/fs_group_edges.tsv")
OUT_MC = os.path.join(ROOT, "data/curated/fold_switch_matched_controls.tsv")
OUT_BAL = os.path.join(ROOT, "reports/fs_three_layer/matching_balance.tsv")
OUT_BALM = os.path.join(ROOT, "reports/fs_three_layer/matching_balance.md")
OUT_MD = os.path.join(ROOT, "reports/fs_three_layer/p201_step2_formal_grouping.md")
QC = os.path.join(ROOT, "reports/fs_three_layer/p201_step2_qc.json")
SIFTS = os.path.join(ROOT, "data/raw/sifts/2026-09-23/pdb_chain_uniprot.csv.gz")
RUN_TS = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def die(m):
    print(f"[p201s2 FATAL] {m}", file=sys.stderr)
    sys.exit(1)


def rd(p, d="\t"):
    with open(p, newline="") as f:
        return list(csv.DictReader(f, delimiter=d))


# ---------- 样本登记 ----------
samples = {}   # sample_id -> dict(area, uniprots set, extra)
def add(sid, area, unis):
    if sid in samples:
        die(f"sample_id 冲突: {sid}")
    samples[sid] = {"area": area, "unis": set(u for u in unis if u)}

g = rd(os.path.join(ROOT, "data/curated/fold_switch_global.tsv"))
if len(g) != 96:
    die("global 表行数异常")
pos_uniprot = {}
for r in g:
    add(r["pair_id"], "fold_switch", {r["uniprot_a"], r["uniprot_b"]})
    if r["tier"] == "strict_state_candidate":
        pos_uniprot[r["uniprot_a"]] = r["pair_id"]

knots = rd(os.path.join(ROOT, "data/curated/knots.tsv"))
need_pdb = {(r["pdb"].lower(), r["chain"].upper()) for r in knots}
knot_uni = defaultdict(set)
with gzip.open(SIFTS, "rt") as f:
    for row in csv.DictReader(l for l in f if not l.startswith("#")):
        key = (row["PDB"].lower(), row["CHAIN"].upper())
        if key in need_pdb:
            knot_uni[key].add(row["SP_PRIMARY"])
knot_nomap = 0
for r in knots:
    u = knot_uni.get((r["pdb"].lower(), r["chain"].upper()), set())
    if not u:
        knot_nomap += 1
    add(f"knot:{r['record_id']}", "knot", u)

dis = rd(os.path.join(ROOT, "data/curated/disorder.tsv"))
dis_uni = defaultdict(set)
for r in dis:
    dis_uni[r["disprot_id"]].add(r["uniprot_acc"])
for dp, us in dis_uni.items():
    add(f"disorder:{dp}", "disorder", us)

for r in rd(os.path.join(ROOT, "data/curated/pyp_pairs.tsv")):
    sid = r.get("sample_id") or f"{r['row_kind']}:{r.get('pair_group_id','')}:{r.get('experiment_id','')}"
    add(f"pyp:{sid}", "pyp", {"P16113"} if sid else set())
for r in rd(os.path.join(ROOT, "data/curated/rnase_a_pairs.tsv")):
    sid = r.get("sample_id") or f"{r['row_kind']}:{r.get('experiment_id','')}"
    add(f"rnase:{sid}", "rnase_a", {"P61823"} if sid else set())
for r in rd(os.path.join(ROOT, "data/curated/bpti_pairs.tsv")):
    sid = r.get("variant_id") or r.get("pair_group_id") or r.get("experiment_id") or r.get("row_kind")
    add(f"bpti:{r['row_kind']}:{sid}", "bpti", {"P00974"})

# ---------- 边 ----------
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

edges_out = []   # (a,b,reason,value)
uf = UF()

# E0 同 UniProt（跨样本）
uni2samples = defaultdict(list)
for sid, s in samples.items():
    for u in s["unis"]:
        uni2samples[u].append(sid)
n_e0 = 0
for u, sids in uni2samples.items():
    sids = sorted(set(sids))
    for i in range(1, len(sids)):
        uf.union(sids[0], sids[i])
        edges_out.append([sids[0], sids[i], "E0_same_uniprot", u])
        n_e0 += 1

# FS E1/E2/E3（跨对样本）
pair_of_ep = {}
for r in g:
    for side, pdb, ch in (("A", r["pdb_a"], r["chain_a"]), ("B", r["pdb_b"], r["chain_b"])):
        pair_of_ep[f"EP_{r['pair_id']}_{side}__{pdb.lower()}{ch.upper()}"] = r["pair_id"]
n_e1 = n_e2 = n_e3 = 0
with open(os.path.join(WORK, "mmseqs_main_cluster.tsv")) as f:
    for line in f:
        c, m = line.rstrip("\n").split("\t")
        pa, pb = pair_of_ep.get(c), pair_of_ep.get(m)
        if pa and pb and pa != pb:
            uf.union(pa, pb)
            edges_out.append([pa, pb, "E2_seq_cluster_0.30_0.70", c.split("__")[0]])
            n_e2 += 1
tm_edges = rd(os.path.join(ROOT, "reports/fs_three_layer/p201_step1_structural_edges.tsv"))
for e in tm_edges:
    if e["edge_min_0.6"] != "yes":
        continue
    pa = pair_of_ep.get(e["a"])
    pb = pair_of_ep.get(e["b"])
    if pa and pb and pa != pb:
        uf.union(pa, pb)
        edges_out.append([pa, pb, "E3_tm_min_ge_0.6", f"{e['tm_sym_min']}|{e['rmsd']}"])
        n_e3 += 1
# E1 只在组表内体现（sample=pair），不计跨样本边

# 完全重复边去重（同原因同值的多条簇/结构边只记一条）
edges_out = sorted({tuple(e) for e in edges_out})
edges_out = [list(e) for e in edges_out]
cnt = Counter(e[2] for e in edges_out)
n_e0, n_e2, n_e3 = cnt.get("E0_same_uniprot", 0), cnt.get("E2_seq_cluster_0.30_0.70", 0), cnt.get("E3_tm_min_ge_0.6", 0)

# ---------- group_map ----------
members = defaultdict(list)
for sid in samples:
    members[uf.find(sid)].append(sid)
order = sorted(members, key=lambda gx: (-len(members[gx]), sorted(members[gx])[0]))
gid = {gx: f"G{i:03d}" for i, gx in enumerate(order, 1)}
os.makedirs(os.path.dirname(OUT_GM), exist_ok=True)
with open(OUT_GM, "w", newline="") as f:
    w = csv.writer(f, delimiter="\t", lineterminator="\n")
    w.writerow(["sample_id", "task_area", "uniprots", "group_id", "group_size"])
    for sid in sorted(samples):
        gx = gid[uf.find(sid)]
        w.writerow([sid, samples[sid]["area"], ";".join(sorted(samples[sid]["unis"])),
                    gx, len(members[uf.find(sid)])])
with open(OUT_E, "w", newline="") as f:
    w = csv.writer(f, delimiter="\t", lineterminator="\n")
    w.writerow(["sample_a", "sample_b", "binding_reason", "value"])
    w.writerows(sorted(edges_out))

# ---------- L3 匹配 ----------
pn = rd(os.path.join(ROOT, "data/curated/fold_switch_putative_negative_candidates.tsv"))
pna_rev = {r["uniprot_accession"]: r for r in rd(os.path.join(ROOT, "reports/fs_three_layer/pna_evidence_review.tsv"))}
ZERO_FAMILY = {"B9W5G6", "Q58AD3", "Q8E473"}   # G1 限定 3：暂退 L3 主结果
NO_PNA = {"P19726"}                            # 池内无 PN-A（P1.11/P1.16）：不入主集（G1：PN-C 不凑主结果）
cases = [u for u in sorted(pos_uniprot) if u not in ZERO_FAMILY and u not in NO_PNA]
eligible = defaultdict(list)
excluded_log = []
for r in pn:
    pos = r["matched_strict_positive"]
    acc = r["uniprot_accession"]
    if r["pn_tier"] != "PN-A":
        continue
    rev = pna_rev.get(acc)
    usable = (rev is not None and rev["post_check_usable"] == "yes")
    rec = {"pos": pos, "acc": acc, "match_rank": int(r["match_rank"]),
           "sp_length": r["sp_length"], "length_ratio": r["length_ratio_to_positive"],
           "n_pdb": r["n_pdb_entries"], "studies": r["n_independent_studies"],
           "coverage": r["mapped_coverage_max"], "review": (rev or {}).get("textual_corroboration_level", "not_reviewed")}
    if usable:
        eligible[pos].append(rec)
    else:
        excluded_log.append({**rec, "why": "PN-A 但未过 P1.16 复核可用判定" if rev else "PN-A 未入 P1.16 复核集"})

mc_rows = []
balance_rows = []
case_obs_len = {}
for r2 in rd(os.path.join(ROOT, "reports/fs_three_layer/sequence_identity_review.tsv")):
    case_obs_len.setdefault(r2["pair_id"], r2["obs_len"])  # 端点 A 的观测长度（首见）
for pos in cases:
    cands = sorted(eligible.get(pos, []), key=lambda x: (x["match_rank"], x["acc"]))
    if not cands:
        mc_rows.append({"matched_set_id": f"MS_{pos}", "positive_uniprot": pos,
                        "positive_pair": pos_uniprot[pos], "ratio": "1:0",
                        "control_accession": "NONE_ELIGIBLE", "match_rank": "",
                        "review_status": "-", "note": "无资格对照（资格=PN-A∧P1.16 usable）"})
        continue
    k_main = 1
    k_sens = min(3, len(cands))
    for i, c in enumerate(cands[:k_sens]):
        mc_rows.append({
            "matched_set_id": f"MS_{pos}_{i+1:02d}",
            "positive_uniprot": pos, "positive_pair": pos_uniprot[pos],
            "ratio": ("1:1_main" if i < k_main else "1:3_sensitivity"),
            "control_accession": c["acc"], "match_rank": c["match_rank"],
            "review_status": "pending_manual_fulltext",
            "sp_length": c["sp_length"], "length_ratio": c["length_ratio"],
            "n_pdb": c["n_pdb"], "n_independent_studies": c["studies"],
            "mapped_coverage": c["coverage"],
            "pna_textual_corroboration": c["review"],
            "note": ""})
        if i < k_main:
            pair_id = pos_uniprot[pos]
            balance_rows.append({
                "matched_set": f"MS_{pos}", "side": "case",
                "sp_length": case_obs_len.get(pair_id, ""),
                "length_ratio": "1.0", "n_pdb": "", "studies": "", "coverage": "",
                "control_accession": "-"})
            balance_rows.append({
                "matched_set": f"MS_{pos}", "side": "control",
                "sp_length": c["sp_length"], "length_ratio": c["length_ratio"],
                "n_pdb": c["n_pdb"], "studies": c["studies"], "coverage": c["coverage"],
                "control_accession": c["acc"]})

cols = ["matched_set_id", "positive_uniprot", "positive_pair", "ratio", "control_accession",
        "match_rank", "review_status", "sp_length", "length_ratio", "n_pdb",
        "n_independent_studies", "mapped_coverage", "pna_textual_corroboration", "note"]
for r in mc_rows:
    for c in cols:
        r.setdefault(c, "")
with open(OUT_MC, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=cols, delimiter="\t", lineterminator="\n")
    w.writeheader()
    w.writerows(mc_rows)
with open(OUT_BAL, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(balance_rows[0]), delimiter="\t", lineterminator="\n")
    w.writeheader()
    w.writerows(balance_rows)

# 平衡统计（6 个 1:1 主集；n 小，只报原始配对值不做群体推断）
bal_lines = ["# L3 匹配平衡（1:1 主集；n=6，案例级报告）", "",
             "选择规则（预注册）：每病例在资格集（PN-A∧P1.16 usable）中取 match_rank 最小者；",
             "1:3 敏感集追加 rank 次小者。全部对照 review_status=pending_manual_fulltext（G1 限定 2）。", "",
             "| matched_set | case_len(obs,A端) | control_len(sp) | length_ratio | control_n_pdb | control_studies | control_coverage |",
             "|---|---|---|---|---|---|---|"]
by_set = defaultdict(dict)
for r in balance_rows:
    by_set[r["matched_set"]][r["side"]] = r
for ms in sorted(by_set):
    cse, ctl = by_set[ms].get("case", {}), by_set[ms].get("control", {})
    bal_lines.append(f"| {ms} | {cse.get('sp_length','')} | {ctl.get('sp_length','')} | "
                     f"{ctl.get('length_ratio','')} | {ctl.get('n_pdb','')} | {ctl.get('studies','')} | {ctl.get('coverage','')} |")
bal_lines += ["", "## 资格集与排除",
              f"- 每病例资格数={dict((p, len(eligible.get(p, []))) for p in cases)}。",
              f"- PN-A 未入资格 {len(excluded_log)} 个（P1.16 复核不可用/未复核），PN-B/C/EXITED 不入主集。",
              "- 平衡判读边界：n=6 为案例级匹配，不做群体 SMD 推断；长度比 range 见上表；"
              "对照结构数/研究数普遍高于病例（PN-A 门槛使然），已在协议匹配变量语义下登记。"]
with open(OUT_BALM, "w") as f:
    f.write("\n".join(bal_lines) + "\n")

qc = {
    "run_ts": RUN_TS,
    "frozen_params": {"seq": "mmseqs 0.30/0.70 cov-mode 0", "struct": "USalign 20260908 TM>=0.6 min",
                      "proportions": "70:15:15 group-level", "seed": 2026,
                      "authority": "user delegation 2026-09-24 (decisions.md 02:34)"},
    "group_map": {"samples": len(samples),
                  "by_area": dict(Counter(s["area"] for s in samples.values())),
                  "n_groups": len(members),
                  "size_hist": dict(Counter(sorted((len(v) for v in members.values()), reverse=True))),
                  "knot_chains_without_sifts_uniprot": knot_nomap},
    "fs_edges": {"E0_same_uniprot": n_e0, "E2_seq_cross": n_e2, "E3_tm_min_cross": n_e3},
    "matching": {"cases": cases,
                 "eligible_counts": {p: len(eligible.get(p, [])) for p in cases},
                 "main_1v1": sum(1 for r in mc_rows if r["ratio"] == "1:1_main"),
                 "sensitivity_1v3": sum(1 for r in mc_rows if r["ratio"] == "1:3_sensitivity"),
                 "excluded_pna": len(excluded_log),
                 "review_status": "pending_manual_fulltext（全部；G1 限定 2）"},
    "no_split_manifest_emitted": True,
}
with open(QC, "w") as f:
    json.dump(qc, f, ensure_ascii=False, indent=1, sort_keys=True)

md = ["# P2.01 步骤二：正式分组与 L3 主匹配（冻结参数执行）", "",
      f"时间：{RUN_TS}。授权：用户委托（logs/decisions.md 2026-09-24 02:34）；冻结参数见 configs/split_protocol.yaml（binding_rules/allocation）。",
      "", "## 1. 全局 group_map（有标注样本）",
      f"- 样本 {len(samples)} 个：{dict(Counter(s['area'] for s in samples.values()))}；背景 A/B 未标注宇宙不在 group_map（P2.02 L2 manifest 以同一冻结参数单独生成）。",
      f"- 合并绑定后 **{len(members)} 组**；规模分布={dict(sorted(Counter((len(v) for v in members.values())).items(), key=lambda kv: -kv[0]))}。",
      f"- 边：E0 同 UniProt={n_e0}（knots 链经 SIFTS 映射，{knot_nomap} 链无映射→仅靠其它边绑定）；FS 跨对序列簇边={n_e2}；FS TM(min)≥0.6 跨对边={n_e3}。",
      "", "## 2. L3 主匹配（matched_controls）",
      f"- 病例={len(cases)}（10−3 零家族[B9W5G6/Q58AD3/Q8E473，G1 限定 3]−1 P19726[池无 PN-A，PN-C 不凑主结果]）；每病例资格数（PN-A∧P1.16 usable）={ {p: len(eligible.get(p, [])) for p in cases} }。",
      f"- 主集 1:1×{qc['matching']['main_1v1']}；敏感性 1:3 追加 {qc['matching']['sensitivity_1v3']} 对照；未入资格的 PN-A {qc['matching']['excluded_pna']} 个如实排除。",
      "- 全部对照 review_status=pending_manual_fulltext（G1 限定 2）：未复核前仅敏感性可用，确证主分析等复核任务完成后收敛。",
      "", "## 3. 残留与移交",
      "- knots 宇宙结构近邻边未算（Foldseek，P2.03 审计任务补算/复核——登记残留）。",
      "- L2 背景分组与全局划分 manifest 归 P2.02（下一任务）。"]
with open(OUT_MD, "w") as f:
    f.write("\n".join(md) + "\n")
print(f"[p201s2] samples={len(samples)} groups={len(members)} E0={n_e0} E2={n_e2} E3={n_e3} "
      f"main={qc['matching']['main_1v1']} sens={qc['matching']['sensitivity_1v3']}")
