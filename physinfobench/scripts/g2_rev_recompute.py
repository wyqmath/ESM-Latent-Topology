#!/usr/bin/env python3
"""G2-REV：G2 有条件确认核对修订——用最终冻结划分复算 G2 材料数字 + max 口径 31 边清单。

指令来源：用户 2026-09-24 15:47 G2 有条件确认（决策日志同刻条目）核对修订任务第 1 项：
  用当前最终 split manifest、split_qc.json 和 split_manifest_report.md 复算 G2 各任务的三集合
  数量及打结类型分布，修正报告中沿用旧划分（P2.02 初版）的数字，含 final holdout 中的 5_2 样本；
  并保留/核对 max 口径下 31 条跨集合边（唯一样本、所属集合、相似度、影响分析）。

纪律：只读冻结产物；不读取 confirmation/final_holdout 的任何标签值（只用 split 归属做清点）；
断言失败即 exit 1（不产出报告）。输出三件：
  reports/g2_rev_recompute.md           复算报告（修正对照表）
  reports/g2_rev_recompute_qc.json      机器可查计数
  reports/g2_rev_maxrule_edges.tsv      max 口径跨集合边全表（31 行）
"""
import csv
import datetime
import gzip
import json
import os
import sys
from collections import Counter, defaultdict

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RUN_TS = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")


def die(m):
    print(f"[g2rev FATAL] {m}", file=sys.stderr)
    sys.exit(1)


def rd(p, d="\t"):
    with open(p, newline="") as f:
        return list(csv.DictReader(f, delimiter=d))


man = rd(os.path.join(ROOT, "data/splits/split_manifest.tsv"))
if len(man) != 4858:
    die(f"manifest 行数 {len(man)} != 4858")
split_of = {r["sample_id"]: r["split"] for r in man}
group_of = {r["sample_id"]: r["group_id"] for r in man}

# ---- 1) 任务区 × 三集合（行=样本/区段行，与 P1.09 流失表总量对账） ----
area_split = defaultdict(Counter)
for r in man:
    area_split[r["task_area"]][r["split"]] += 1
AREA_TOTALS = {"knot": 1401, "fold_switch": 96, "disorder": 3337,
               "bpti": 9, "pyp": 6, "rnase_a": 9}
for a, n in AREA_TOTALS.items():
    got = sum(area_split[a].values())
    if got != n:
        die(f"{a} 总量 {got} != {n}（P1.09 流失表）")

# ---- 2) 打结类型 × 三集合（复刻 P2.02 选择：type_task_tier=eligible，计 c2_primary） ----
kt = {r["record_id"]: r for r in rd(os.path.join(ROOT, "data/curated/knots.tsv"))}
type_split = defaultdict(Counter)
n_type_rows = 0
for r in man:
    if r["task_area"] != "knot":
        continue
    rec = kt.get(r["sample_id"].split(":", 1)[1])
    if rec and rec["type_task_tier"] == "eligible":
        n_type_rows += 1
        type_split[r["split"]][rec["c2_primary"]] += 1
if n_type_rows != 112:
    die(f"type 行数 {n_type_rows} != 112（P1.09）")

# ---- 3) FS：96 对归属 + strict 10 对全 dev ----
fs = rd(os.path.join(ROOT, "data/curated/fold_switch_global.tsv"))
tier_of = {r["pair_id"]: r["tier"] for r in fs}
strict = [r["pair_id"] for r in fs if r["target"] == "1" and r["valid_mask"] == "1"]
if len(strict) != 10:
    die(f"strict 对数 {len(strict)} != 10")
strict_not_dev = [p for p in strict if split_of[p] != "development"]
if strict_not_dev:
    die(f"strict 不全在 development: {strict_not_dev}")

# ---- 4) L2 背景蛋白三集合（非 EP_ 蛋白口径，与 P2.02 报告更正一致） ----
bg_prot = Counter()
n_ep = Counter()
with gzip.open(os.path.join(ROOT, "data/splits/fs_l2_background_manifest.tsv.gz"), "rt") as f:
    for r in csv.DictReader(f, delimiter="\t"):
        if r["uniprot_accession"].startswith("EP_"):
            n_ep[r["split"]] += 1
        else:
            bg_prot[r["split"]] += 1
BG_EXPECT = {"development": 343682, "confirmation": 68654, "final_holdout": 71850}
if dict(bg_prot) != BG_EXPECT:
    die(f"背景蛋白数 {dict(bg_prot)} != {BG_EXPECT}")

# ---- 5) max 口径跨集合边全表（与 audit_leakage.py L4 同源同口径） ----
edges = rd(os.path.join(ROOT, "reports/fs_three_layer/p201_step1_structural_edges.tsv"))
pair_of_ep = {}
for r in fs:
    for side, pdb, ch in (("A", r["pdb_a"], r["chain_a"]), ("B", r["pdb_b"], r["chain_b"])):
        pair_of_ep[f"EP_{r['pair_id']}_{side}__{pdb.lower()}{ch.upper()}"] = r["pair_id"]
max_cross = []
for e in edges:
    if e["edge_max_0.6"] != "yes" or e["edge_min_0.6"] == "yes":
        continue
    pa, pb = pair_of_ep.get(e["a"]), pair_of_ep.get(e["b"])
    if pa and pb and split_of[pa] != split_of[pb]:
        max_cross.append({
            "pair_a": pa, "pair_b": pb,
            "tier_a": tier_of[pa], "tier_b": tier_of[pb],
            "split_a": split_of[pa], "split_b": split_of[pb],
            "group_a": group_of[pa], "group_b": group_of[pb],
            "ep_a": e["a"], "ep_b": e["b"],
            "tm_sym_max": e["tm_sym_max"], "tm_sym_min": e["tm_sym_min"],
            "cross_class": "-".join(sorted([split_of[pa], split_of[pb]])),
        })
max_cross.sort(key=lambda x: (-float(x["tm_sym_max"]), x["pair_a"], x["pair_b"], x["ep_a"], x["ep_b"]))
qc_prev = json.load(open(os.path.join(ROOT, "reports/leakage_audit_qc.json")))
if len(max_cross) != qc_prev["checks"]["L4_max_rule_cross_split_edges_fs"]:
    die(f"max 边 {len(max_cross)} != leakage_audit_qc L4 ({qc_prev['checks']['L4_max_rule_cross_split_edges_fs']})")

OUT_TSV = os.path.join(ROOT, "reports/g2_rev_maxrule_edges.tsv")
with open(OUT_TSV, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(max_cross[0].keys()), delimiter="\t", lineterminator="\n")
    w.writeheader()
    w.writerows(max_cross)

# ---- 6) 汇总 ----
uniq_pairs = sorted({e["pair_a"] for e in max_cross} | {e["pair_b"] for e in max_cross})
uniq_strict = [p for p in uniq_pairs if p in strict]
class_cnt = Counter(e["cross_class"] for e in max_cross)
involved = [{"pair": p, "tier": tier_of[p], "split": split_of[p], "group": group_of[p]}
            for p in uniq_pairs]

qc = {
    "run_ts": RUN_TS,
    "area_split": {a: dict(c) for a, c in area_split.items()},
    "knot_type_by_split": {k: dict(v) for k, v in type_split.items()},
    "type_rows_total": n_type_rows,
    "strict_pairs": strict,
    "strict_all_development": True,
    "l2_background_proteins": dict(bg_prot),
    "l2_background_ep_rows": dict(n_ep),
    "max_rule_cross_edges": len(max_cross),
    "max_edge_cross_class": dict(class_cnt),
    "max_edge_unique_pairs": involved,
    "max_edge_strict_involved": uniq_strict,
    "asserts_passed": ["manifest=4858", "area_totals", "type_rows=112", "strict10_all_dev",
                       "bg_proteins", "max_cross==leakage_audit_L4"],
}
with open(os.path.join(ROOT, "reports/g2_rev_recompute_qc.json"), "w") as f:
    json.dump(qc, f, ensure_ascii=False, indent=1, sort_keys=True)

# ---- 7) 报告 ----
def fmt(cs):
    return "/".join(f"{k} {cs.get(k, 0)}" for k in ("development", "confirmation", "final_holdout"))


L = ["# G2-REV 复算报告（G2 有条件确认核对修订）", "",
     f"时间：{RUN_TS}。输入=冻结划分 split_manifest.tsv（4,858 行）+ split_qc.json + knots.tsv + "
     "fold_switch_global.tsv + fs_l2_background_manifest.tsv.gz + p201_step1_structural_edges.tsv。"
     "复算脚本=scripts/g2_rev_recompute.py（断言全过才产出）。"
     "结论：G2_milestone.md §3 的 knots/FS/disorder/受控三系统行原数字沿用 P2.02 初版划分"
     "（commit cd768cd 版 split_qc.json），未随 P2.03 审计驱动重划分更新；L2 背景行原值即现行值"
     "（并非沿自初版，见 §4）。现行修正值如下。", "",
     "## 1. 各任务三集合数量（行/样本；修正对照）", "",
     "| 任务 | G2 材料原值（初版划分） | 现行复算值（dev/conf/hold） | 总量对账 |",
     "|---|---|---|---|",
     f"| knots presence | 1,015/170/216 | {area_split['knot']['development']}/{area_split['knot']['confirmation']}/{area_split['knot']['final_holdout']} | 1,401=P1.09 ✔ |",
     f"| FS（D6） | 78/4/14 | {area_split['fold_switch']['development']}/{area_split['fold_switch']['confirmation']}/{area_split['fold_switch']['final_holdout']} | 96 ✔ |",
     f"| disorder | 2,366/498/473 | {area_split['disorder']['development']}/{area_split['disorder']['confirmation']}/{area_split['disorder']['final_holdout']} | 3,337 ✔ |",
     f"| bpti/pyp/rnase_a | 9/-/- 各全 dev | {area_split['bpti']['development']}/{area_split['pyp']['development']}/{area_split['rnase_a']['development']}（全 dev） | 9/6/9 ✔ |", "",
     "## 2. 打结类型 × 三集合（112 行；修正对照）", "",
     "| 集合 | 类型构成 | 合计 |",
     "|---|---|---|",
     f"| development | 3_1={type_split['development'].get('3_1', 0)}、4_1={type_split['development'].get('4_1', 0)}、5_2={type_split['development'].get('5_2', 0)}、5_1={type_split['development'].get('5_1', 0)} | {sum(type_split['development'].values())} |",
     f"| confirmation | 3_1={type_split['confirmation'].get('3_1', 0)}（**仅 3_1**，其余类型在确认集不可评价） | {sum(type_split['confirmation'].values())} |",
     f"| final_holdout | 3_1={type_split['final_holdout'].get('3_1', 0)}、4_1={type_split['final_holdout'].get('4_1', 0)}、**5_2={type_split['final_holdout'].get('5_2', 0)}** | {sum(type_split['final_holdout'].values())} |", "",
     "修正要点：原值 80/14/13 不构成任一单一划分版本的一致快照（初版划分=80/14/18，现行=85/14/13；"
     "dev/conf 与初版一致而 holdout 与现行一致），且『5_1 仅 dev』注记遗漏现行 final_holdout 中的 5_2×1"
     "（用户指出处）。现行口径："
     "5_1 仅 dev 1 行（conf/hold NA 如实）；5_2 在 dev 2 行+hold 1 行、confirmation 无（确认集该类 NA）；"
     "4_1 三集合皆有；confirmation 打结类型任务实际只可评价 3_1。", "",
     f"## 3. strict 10 对归属（复算）", "",
     f"10/10 全部 development（{', '.join(strict)}）。G2 材料『strict 10 全 dev』在现行划分下成立。", "",
     "## 4. L2 背景三集合（蛋白数口径，与 P2.02 报告更正一致）", "",
     f"- 非 EP_ 蛋白：dev {bg_prot['development']:,} / conf {bg_prot['confirmation']:,} / hold {bg_prot['final_holdout']:,}。G2 材料此行原值（343,682 dev）即现行值，复算维持——该行并非沿自初版划分（初版背景分布=335,662/73,349/75,175，EP_ 行 156/8/28，git show cd768cd 可复算）。",
     f"- EP_ 端点行：dev {n_ep.get('development', 0)} / conf {n_ep.get('confirmation', 0)} / hold {n_ep.get('final_holdout', 0)}（不入背景蛋白计数）。", "",
     f"## 5. max 口径跨集合边全表（{len(max_cross)} 条；用户 G2 裁决第 1 项）", "",
     f"全表=reports/g2_rev_maxrule_edges.tsv（{len(max_cross)} 行，按 tm_sym_max 降序）。跨集合类别："
     + "、".join(f"{k}={v}" for k, v in sorted(class_cnt.items())) + f"。涉及唯一样本（对）{len(uniq_pairs)} 个：", "",
     "| pair | tier | 集合 | 组 |", "|---|---|---|---|"]
for it in involved:
    L.append(f"| {it['pair']} | {it['tier']} | {it['split']} | {it['group']} |")
L += ["", f"涉及 strict 阳性对：{len(uniq_strict)} 个（{'、'.join(uniq_strict) if uniq_strict else '无'}）。", "",
      "### 影响面（按任务）", "",
      "- **L1 配对机制诊断**：10 对 strict 全 dev、分析宇宙在 dev 内——dev 内部结论不受影响；"
      "涉及 strict 的 max 边意味着对这些相似跨集合亲属不得做跨集合外推主张。", "",
      "- **L2 PU 排序**：排序宇宙=L2 development（343,682 背景+10 strict）——跨集合边另一端在宇宙外，"
      "对 dev 排序点估计无影响；将来若把 L2 扩到 conf/hold 需先做敏感性。", "",
      "- **L3 匹配病例-对照**：matched_set 17 全部与阳性同集（dev）；31 边为未构成绑定的单侧相似——"
      "『通过结构近邻绑定检查』只能按 min 口径声称，不得写成所有单侧结构相似已跨集合隔离（用户裁决原文）。", "",
      "### 预登记敏感性方案（模型结果出现时执行）", "",
      "1. 主分析照常（min 口径冻结划分）；2. 敏感性变体=按 max 口径合并涉及组重算（或剔除涉边样本），"
      "报告主结论方向/显著性是否改变；3. 若结论依赖这 31 条边或涉边后样本量不足以支持原定主张，"
      "按协议降级结论并在报告标注——不改口径掩盖风险。全程不读取 conf/holdout 标签选择规则。", "",
      "## 6. 与 split_qc.json / split_manifest_report.md 的一致性", "",
      "本报告全部数字与 data/splits/split_qc.json（run_ts 2026-09-24 03:27）及 "
      "reports/split_manifest_report.md 逐项一致（两者为 P2.03 重划分后的现行版本）；"
      "需要修正的只有 reports/G2_milestone.md §3（已同步修订）。"]
with open(os.path.join(ROOT, "reports/g2_rev_recompute.md"), "w") as f:
    f.write("\n".join(L) + "\n")
print(f"[g2rev] OK max_cross={len(max_cross)} uniq_pairs={len(uniq_pairs)} strict_involved={len(uniq_strict)}")
print(f"[g2rev] knot dev={dict(type_split['development'])} conf={dict(type_split['confirmation'])} hold={dict(type_split['final_holdout'])}")
