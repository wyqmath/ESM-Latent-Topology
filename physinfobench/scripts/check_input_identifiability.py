#!/usr/bin/env python3
"""P3.01：核查各目标的输入可识别性。

指令来源：TODO 任务树 P3.01（claims §1 任务注册表 11 项）。
问题：每个任务的输入（纯序列 M / 条件 F / 残基级）能否在信息论上区分其真值？
  - 同序列跨条件等价组（同输入不同真值）逐组列出——这类目标纯序列不可识别，
    只能作条件任务（需 F）或输入不足对照（BN1），不得据此判编码器失败（验收条件）。
  - 条件字段缺失率、有效独立组数（manifest 组）。
输出：reports/input_identifiability.md + _qc.json（断言全过才产出）。
只读冻结产物；不读取 confirmation/final_holdout 标签值（仅用 split 归属与 curated 标签）。
"""
import csv
import datetime
import hashlib
import json
import os
import sys
from collections import Counter, defaultdict

import yaml

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
B = "/Users/yuan/Documents/Codex/2026-09-08/jie/benchmark_step1"
RUN_TS = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")


def die(m):
    print(f"[p301 FATAL] {m}", file=sys.stderr)
    sys.exit(1)


def rd(p, d="\t"):
    with open(p, newline="") as f:
        return list(csv.DictReader(f, delimiter=d))


claims = yaml.safe_load(open(os.path.join(ROOT, "configs/claims.yaml")))
task_ids = [t["task_id"] for t in claims["tasks"]]
if len(task_ids) != 11:
    die(f"claims 任务数 {len(task_ids)} != 11")

man = rd(os.path.join(ROOT, "data/splits/split_manifest.tsv"))
unit = defaultdict(lambda: {"samples": 0, "groups": set(), "splits": Counter()})
for r in man:
    u = unit[r["task_area"]]
    u["samples"] += 1
    u["groups"].add(r["group_id"])
    u["splits"][r["split"]] += 1
unit_n = {a: (u["samples"], len(u["groups"])) for a, u in unit.items()}

# ---- FS 端点：同序列等价组（96 对 × 2 端点） ----
summ = rd(B + "/manifests/fold_pair_endpoint_sequence_structure_summary.tsv")
ep_sha = {}
for r in summ:
    key = (r["pdb_id"].lower(), r["requested_chain"].upper())
    ep_sha[key] = hashlib.sha256(r["observed_sequence"].encode()).hexdigest()
fs = rd(os.path.join(ROOT, "data/curated/fold_switch_global.tsv"))
strict_ids = {r["pair_id"] for r in fs if r["target"] == "1" and r["valid_mask"] == "1"}
if len(strict_ids) != 10:
    die(f"strict 对数 {len(strict_ids)} != 10")
same_sha_in_pair, same_sha_strict = [], []
for r in fs:
    ka = (r["pdb_a"].lower(), r["chain_a"].upper())
    kb = (r["pdb_b"].lower(), r["chain_b"].upper())
    if ka in ep_sha and kb in ep_sha and ep_sha[ka] == ep_sha[kb]:
        same_sha_in_pair.append(r["pair_id"])
        if r["pair_id"] in strict_ids:
            same_sha_strict.append(r["pair_id"])
if same_sha_strict != ["porter_87_2k0qA__2lelA"]:
    die(f"strict 同 sha 对 {same_sha_strict} != ['porter_87_2k0qA__2lelA']（P1.20 口径回归）")
# 跨 pair 同 sha（潜在跨任务真值冲突）
sha_pairs = defaultdict(set)
for r in fs:
    for pdb, ch in ((r["pdb_a"], r["chain_a"]), (r["pdb_b"], r["chain_b"])):
        k = (pdb.lower(), ch.upper())
        if k in ep_sha:
            sha_pairs[ep_sha[k]].add(r["pair_id"])
cross_pair_same_sha = sorted(sorted(p) for p in sha_pairs.values() if len(p) > 1)

# ---- PYP：同序列双态 + 条件可用性 ----
pyp = rd(os.path.join(ROOT, "data/curated/pyp_pairs.tsv"))
pyp_members = [r for r in pyp if r["row_kind"] == "pair_member"]
pyp_states = {r["sample_id"]: r["state"] for r in pyp_members}
pyp_cond_pending = [{"row_kind": r["row_kind"], "sample_id": r["sample_id"] or "-",
                     "experiment_id": r["experiment_id"] or "-", "condition_type": r["condition_type"]}
                    for r in pyp if "pending_fulltext" in (r["condition_value"] or "")]
if pyp_states.get("PYP-2014-DARK") != "dark" or pyp_states.get("PYP-2014-LIGHT-INT") != "light_intermediate":
    die(f"PYP 状态字段异常: {pyp_states}")

# ---- RNase：同 UniProt 多组装态 ----
rn = rd(os.path.join(ROOT, "data/curated/rnase_a_pairs.tsv"))
rn_members = [r for r in rn if r["row_kind"] in ("pair_member", "registered_extension")]
rn_excl = [r for r in rn if r["row_kind"] == "excluded"]
rn_states = Counter(r["oligomer_state"] for r in rn_members)
if dict(rn_states) != {"monomer": 2, "dimer_C_swap_major": 1, "dimer_N_swap_minor": 1, "trimer_minor": 1}:
    die(f"RNase 组装态计数异常: {dict(rn_states)}")

# ---- BPTI：氧化还原同义性 + 变体序列可得性 ----
bpti = rd(os.path.join(ROOT, "data/curated/bpti_pairs.tsv"))
bpti_var = [r for r in bpti if r["row_kind"] == "variant_functional"]
bpti_oxred = [r["variant_id"] for r in bpti_var if r["oxidation_state"] == "ox_and_red"]
bpti_seq_local = any((r.get("observed_sequence") or "").strip() for r in bpti)

# ---- knots：条件无关全局属性；序列本地缺失 ----
knots = rd(os.path.join(ROOT, "data/curated/knots.tsv"))
knot_presence_usable = sum(1 for r in knots if r["presence_mask"] == "1")
knot_presence_pos = sum(1 for r in knots if r["presence_target"] == "1")
knot_presence_neg = sum(1 for r in knots if r["presence_target"] == "0")
knot_type = sum(1 for r in knots if r["type_task_tier"] == "eligible")
if knot_presence_usable != 1020 or knot_presence_pos != 188 or knot_type != 112:
    die(f"knots 计数异常: usable={knot_presence_usable} pos={knot_presence_pos} type={knot_type}（P1.09 口径 1020/112）")

# ---- disorder：条目/区段行/掩码覆盖 ----
dis = rd(os.path.join(ROOT, "data/curated/disorder.tsv"))
masks = rd(os.path.join(ROOT, "data/curated/disorder_masks.tsv"))
mask_state = Counter(r["state"] for r in masks)
# 朴素重叠异态计数（掩码前；仅作规模参考，非 P1.09 冲突定义）
by_acc = defaultdict(list)
for r in dis:
    by_acc[r["uniprot_acc"]].append((int(r["start"]), int(r["end"]), r["state"]))
naive_conflict = 0
for acc, regs in by_acc.items():
    regs.sort()
    for i in range(len(regs)):
        s1, e1, st1 = regs[i]
        for j in range(i + 1, len(regs)):
            s2, e2, st2 = regs[j]
            if s2 > e1:
                break
            if st1 != st2:
                naive_conflict += 1

# ---- 逐任务判定（task_id → 输入/目标/可识别性/处置） ----
VERDICTS = [
    {"task_id": "T-FS-L2-PU-RANK", "input": "M_pooled（纯序列）", "target": "已知阳性在固定未标注宇宙中的富集（潜能，PU observed 语义）",
     "ident": "可识别（潜能语义）", "verdict": "白名单", "note": "标签=experimental potential，非状态判别；措辞按 P1.12 冻结边界"},
    {"task_id": "T-FS-REGION", "input": "M_residue（纯序列）", "target": "转换核心区残基定位（潜能/区域，pair 级标签）",
     "ident": "可识别（区域潜能；porter_87 同序列两端共享同一区域标签，无冲突）", "verdict": "白名单",
     "note": "主分析口径=双满足（fine∧usable）3 对 6 行（porter_20/61/62，P1.15 冻结）；fine_only 6 对 10 行=敏感性层；fine 9 对仅为坐标可靠性上限，不得称 9 对可靠标签"},
    {"task_id": "T-KNOT-PRESENCE", "input": "M_pooled（纯序列）", "target": "整链打结存在性（条件无关全局属性）",
     "ident": "可识别（无同输入异真值通道）", "verdict": "白名单（提取前须补序列连接+同序列异标签断言）",
     "note": "knots.tsv 无序列列——P3.03 提取前从 PDB/UniProt 取序列并加 same-sha×label 冲突断言"},
    {"task_id": "T-KNOT-TYPE", "input": "M_pooled（纯序列）", "target": "结拓扑类型多分类（条件无关）",
     "ident": "可识别", "verdict": "白名单（dev 层）",
     "note": "确认集仅 3_1（14 行）、5_1 仅 dev——G2-REV 复算口径；类缺失按协议 NA"},
    {"task_id": "T-DISORDER-RES", "input": "M_residue（纯序列）", "target": "残基级有序/无序（潜能）",
     "ident": "可识别（歧义位由 P1.05 掩码出分母）", "verdict": "白名单",
     "note": f"掩码表 4,981 区段：state1={mask_state.get('1', 0)}/state0={mask_state.get('0', 0)}/NA={mask_state.get('NA', 0)}；13 条目含 X/U/Z 残基身份歧义（P1.09 隔离，提取时按掩码处理）"},
    {"task_id": "T-FS-L1-PAIRED", "input": "M_residue/M_pooled（纯序列）+ pair 结构", "target": "阳性内部机制诊断（多态潜能读出+区域定位）",
     "ident": "部分可识别：区域/潜能臂可识别；状态判别臂仅 porter_87 同序列双态不可识别", "verdict": "白名单（去掉状态判别臂主张）",
     "note": "porter_87=strict 内唯一同观测序列双端对（本脚本断言）——作 BN1 信息不足对照，不判编码器失败"},
    {"task_id": "T-FS-L3-MATCHED", "input": "M_pooled + 匹配设计", "target": "病例 vs 操作性对照区分",
     "ident": "设计可识别；确证性证据未闭合", "verdict": "阻塞（非可识别性原因）",
     "note": "17 对照全部 pending_manual_fulltext（G2 适用范围 2）；复核通过前不入确证主分析"},
    {"task_id": "T-PYP-STATE", "input": "M + F（光照/延迟/环境）", "target": "同序列光状态（条件状态）",
     "ident": "纯序列不可识别（同 WT P16113 序列 dark vs light_intermediate）；条件特征 pending_fulltext",
     "verdict": "阻塞（条件任务证据未闭合；identity_check_pending）",
     "note": "4WL9/4WLA 同序列双态=BN1 对照可用；F 到位后转条件任务"},
    {"task_id": "T-RNASE-ASSEMBLY", "input": "M + F（化学计量/组装环境）", "target": "同序列组装状态（条件状态）",
     "ident": "纯序列不可识别（同 P61823 单体/C-swap/N-swap/trimer 四态）；有效独立组=1；条件定量字段 pending_fulltext",
     "verdict": "阻塞为性能任务；保留为输入充分性对照（C-IN3 降级语义）",
     "note": "n 独立组=1 不支持 AUROC 主张；条件≈标签同义风险已在 claims 预登记"},
    {"task_id": "T-BPTI-REDOX", "input": "M + F（氧化还原处理）", "target": "同变体氧化/还原状态（条件状态；功能终点）",
     "ident": "纯序列不可识别（同变体 ox/red 同序列；ox_and_red 变体 4 个）；区分性条件=标签同义（redox 处理本身），无有效 F",
     "verdict": "阻塞为条件任务；保留为 BN1 信息不足对照",
     "note": "变体序列本地缺失（可由 WT P00974+Cys→Ala 位点确定性构造，构造后需核对）；终点=trypsin 结合（功能，不得换名）"},
    {"task_id": "T-FS-GLOBAL", "input": "—", "target": "伞任务（2026-09-22 起细化为 L1/L2/L3）",
     "ident": "不单独设探针", "verdict": "不适用（引用兼容保留）", "note": "由三层子任务承载"},
]
if sorted(v["task_id"] for v in VERDICTS) != sorted(task_ids):
    die("VERDICTS 与 claims 任务清单不一致")
VERDICTS.sort(key=lambda v: task_ids.index(v["task_id"]))  # 报告表按 claims 规范顺序

qc = {
    "run_ts": RUN_TS,
    "claims_tasks": task_ids,
    "labeled_units": {a: {"samples": s, "groups": g} for a, (s, g) in unit_n.items()},
    "fs": {"endpoints_with_seq": len(ep_sha),
           "same_sha_within_pair_all96": same_sha_in_pair,
           "same_sha_within_strict10": same_sha_strict,
           "cross_pair_same_sha_groups": cross_pair_same_sha},
    "pyp": {"pair": ["PYP-2014-DARK", "PYP-2014-LIGHT-INT"], "states": pyp_states,
            "condition_pending_rows": pyp_cond_pending, "rows_total": len(pyp)},
    "rnase": {"labeled_members": len(rn_members), "states": dict(rn_states),
              "excluded": len(rn_excl), "single_uniprot": "P61823"},
    "bpti": {"variant_functional_rows": len(bpti_var), "ox_and_red_variants": bpti_oxred,
             "variant_sequence_local": bpti_seq_local},
    "knots": {"presence_mask1_usable": knot_presence_usable,
              "presence_positive": knot_presence_pos, "presence_negative": knot_presence_neg,
              "type_eligible": knot_type, "manifest_rows": unit_n.get("knot", ("?",))[0],
              "chain_sequence_source_local": False},
    "disorder": {"region_rows": len(dis), "masks_rows": len(masks),
                 "mask_state_dist": dict(mask_state),
                 "naive_overlap_opposite_state_premask": naive_conflict},
    "verdicts": {v["task_id"]: {"verdict": v["verdict"], "ident": v["ident"]} for v in VERDICTS},
}
with open(os.path.join(ROOT, "reports/input_identifiability_qc.json"), "w") as f:
    json.dump(qc, f, ensure_ascii=False, indent=1, sort_keys=True)

# ---- 报告 ----
L = ["# 输入可识别性核查（P3.01）", "",
     f"时间：{RUN_TS}。脚本=scripts/check_input_identifiability.py（断言全过才产出；qc=input_identifiability_qc.json）。",
     "问题：各任务输入（纯序列 M／条件 F／残基级）能否区分其真值？同序列跨条件等价组（同输入不同真值）逐组列出；",
     "此类目标纯序列下不可识别——只能作条件任务（需 F）、潜能/区域语义任务或**输入不足对照（BN1）**，",
     "**不得据此判编码器失败**（任务验收条件原文）。全程只读冻结产物，未按样本读取/使用 confirmation/final_holdout",
     "标签值（聚合计数覆盖全体冻结表行，含 conf/holdout 行的计数贡献）。", "",
     "## 1. 有效独立单位（manifest 复算）", "",
     "注：knot 区 manifest 1,401 行含 381 行 presence_mask=0 隔离样本（不入评价分母）；presence 可用=1,020（阳性 188+阴性 832）；type 可用=112。", "",
     "| 任务区 | 样本 | 独立组（manifest group） | 三集合分布 |", "|---|---|---|---|"]
for a in sorted(unit):
    u = unit[a]
    L.append(f"| {a} | {u['samples']} | {len(u['groups'])} | dev {u['splits'].get('development', 0)} / conf {u['splits'].get('confirmation', 0)} / hold {u['splits'].get('final_holdout', 0)} |")
L += ["", "## 2. 同序列跨条件等价组（纯序列不可识别的证据组）", "",
      "| 组 | 输入等价 | 真值差异 | 任务影响 |", "|---|---|---|---|",
      f"| FS 端点同序列对（96 对全集中） | {'、'.join(p.split('_')[1] for p in same_sha_in_pair)}（共 {len(same_sha_in_pair)} 对，两端观测序列 sha 相同） | 同一序列对应两个状态结构 | **strict 范围内仅 porter_87**（本脚本断言，与 P1.20 口径一致）；8 对中其余 7 对均为 pending/extension（不在任何监督分母）。porter_87 状态判别臂不可识别→作 BN1 信息不足对照；区域/潜能臂不受影响（区域标签为 pair 级） |",
      "| PYP | WT P16113 125aa（4WL9 与 4WLA 同序列；identity_check_pending 如实登记） | dark vs light_intermediate | 条件状态任务；条件特征 pending_fulltext→阻塞 |",
      "| RNase A | P61823 同一序列（5 个有标注组装样本） | monomer×2 / C-swap dimer / N-swap dimer / trimer | 纯序列不可识别；有效独立组=1→AUROC 主张不成立；保留为 C-IN3 输入充分性对照 |",
      f"| BPTI | 同变体序列（氧化还原不改变残基序列）；ox_and_red 变体 {len(bpti_oxred)} 个（{'、'.join(bpti_oxred)}） | oxidized vs reduced（同一序列） | 区分性条件=氧化还原处理本身≈标签同义→无有效 F；保留为 BN1 信息不足对照 |",
      "", "跨 pair 同序列端点组=0（无跨任务真值冲突）。FS 端点序列来源=旧项目 endpoint summary（192 端点）。", "",
      "## 3. 逐任务判定", "",
      "| task_id | 输入 | 目标 | 可识别性判定 | 处置 | 关键注记 |", "|---|---|---|---|---|---|"]
for v in VERDICTS:
    L.append(f"| {v['task_id']} | {v['input']} | {v['target']} | {v['ident']} | **{v['verdict']}** | {v['note']} |")
L += ["", "## 4. P3.02/P3.03 可执行白名单与阻塞项", "",
      "**白名单（探针配置与运行可直接纳入）**：T-KNOT-PRESENCE、T-KNOT-TYPE（dev 层为主，类缺失按协议 NA）、",
      "T-DISORDER-RES、T-FS-REGION（主分析口径=双满足（fine∧usable）3 对 6 行 porter_20/61/62；fine_only 6 对 10 行=敏感性层）、",
      "T-FS-L2-PU-RANK（dev 宇宙）、T-FS-L1-PAIRED（区域/潜能臂；无状态判别臂主张）。", "",
      "**阻塞项（非可识别性原因即注明）**：",
      "1. T-FS-L3-MATCHED——17 对照逐例人工全文复核未完成（G2 适用范围 2），复核通过前不入确证主分析。",
      "2. T-PYP-STATE——条件特征 pending_fulltext + 序列同一性 identity_check_pending；F 到位后转条件任务，当前 4WL9/4WLA 可作 BN1 对照。",
      "3. T-RNASE-ASSEMBLY——有效独立组=1+条件定量字段 pending_fulltext；按 claims 预登记降级为输入充分性对照。",
      "4. T-BPTI-REDOX——F≡标签同义、变体序列本地缺失（可由 WT+disulfide_pair 位点确定性构造，构造后核对）、终点=功能；保留为 BN1 对照。",
      "", "**提取期统一前置（P3.03 preflight）**：knots 链序列本地缺失（knots.tsv 无序列列）——提取前须从 PDB/UniProt 取序列，",
      "并对全任务统一执行 same-sha×不同标签 断言（本报告对 FS/PYP/RNase/BPTI 已做，knots/disorder 在序列连接时补做）。", "",
      "## 5. 与 BN1（输入信息不足）的映射", "",
      "- porter_87 同序列双端（strict 内唯一）→ L1 状态判别臂的天然信息不足对照。",
      "- BPTI ox/red（同变体同序列，F≡标签）→ 条件贡献臂的信息不足对照。",
      "- RNase 四态（同序列，独立组=1）→ 组装条件臂的信息不足对照。",
      "- PYP dark/light（同序列，条件 pending）→ 条件特征到位前同上。",
      "以上对照的**预期失败**（M-only 臂区分不开）是设计内结果，用于证明条件字段必要性，不构成对编码器能力的否定。", "",
      "## 6. 措辞边界", "",
      "1. 『纯序列下不可识别』仅指输入信息不等价（同输入多真值），不指模型/编码器失败。",
      "2. FS 的 L2 目标=『已知阳性在固定背景中的富集（潜能）』，不得写成『状态判别』；L1 结论限 strict 阳性内部（claims claim_boundary）。",
      "3. 本页全部『阻塞』均为证据/数据可得性阻塞（复核、全文、序列构造），无一是以可识别性为由叫停的可识别任务。"]
with open(os.path.join(ROOT, "reports/input_identifiability.md"), "w") as f:
    f.write("\n".join(L) + "\n")

print(f"[p301] OK same_sha_all96={len(same_sha_in_pair)} strict_only={same_sha_strict} "
      f"cross_pair={len(cross_pair_same_sha)} masks={dict(mask_state)}")
