#!/usr/bin/env python3
"""P3.02：校验冻结的探针配置（configs/probes.yaml）与统计参数回写。

检查（全部通过 exit 0，否则 exit 1）：
  V1 YAML 可解析；编码器与 models.yaml 一致，revision 与 P2.05 提取 manifest
     revision 列全哈希全等（models.yaml/p204 json 均无 revision 字段）；
  V2 层集合合法（⊆1..33）；residue_set ⊆ global_set；
  V3 任务矩阵 ⊆ P3.01 白名单六项（从 input_identifiability_qc.json 重推白名单）；
  V4 统计参数四项与 evaluation_protocol.yaml 一致且其冻结标注无"待确认"残留；
  V5 选择纪律：selection/readers/tasks 任何键路径不含 confirmation/final_holdout 数据通道；
  V6 L2 两阶段：stage_a 抽样参数可复现（seed/排序键齐全）、stage_b 宇宙=冻结划分数字；
  V7 冻结主指标名与 evaluation_protocol/P0.05 词汇一致（AUROC/Macro-F1/residue AUPRC/
     recall_at_k/enrichment_at_k/positive_rank_percentile）。
只读；无科研运行。
"""
import json
import os
import sys

import yaml

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def die(m):
    print(f"[p302 FATAL] {m}", file=sys.stderr)
    sys.exit(1)


p = yaml.safe_load(open(os.path.join(ROOT, "configs/probes.yaml")))
ep = yaml.safe_load(open(os.path.join(ROOT, "configs/evaluation_protocol.yaml")))
models = yaml.safe_load(open(os.path.join(ROOT, "configs/models.yaml")))
qc301 = json.load(open(os.path.join(ROOT, "reports/input_identifiability_qc.json")))

# V1 编码器一致
enc = p["encoder"]
model_ids = set()
if isinstance(models.get("models"), list):
    for m in models["models"]:
        if isinstance(m, dict):
            model_ids.add(m.get("model_id") or m.get("id"))
if enc["model_id"] != "facebook/esm2_t33_650M_UR50D" or enc["model_id"] not in model_ids:
    die(f"V1 编码器 {enc['model_id']} 不在 models.yaml {model_ids}")
# revision 全哈希比对：P2.05 提取 manifest 的 revision 列（三方一致的权威见证）
import csv as _csv
with open(os.path.join(ROOT, "results/representations_smoke/extract_manifest.tsv"), newline="") as f:
    revs = {row["revision"] for row in _csv.DictReader(f, delimiter="\t") if row.get("revision")}
if revs != {enc["revision"]}:
    die(f"V1 revision 与 P2.05 manifest 不一致: config={enc['revision']} manifest={revs}")

# V2 层集合
g, r = p["layers"]["global_set"], p["layers"]["residue_set"]
if not all(1 <= x <= 33 for x in g) or not all(1 <= x <= 33 for x in r):
    die(f"V2 层索引越界: global={g} residue={r}")
if not set(r) <= set(g):
    die(f"V2 residue_set 不是 global_set 子集: {r} ⊄ {g}")

# V3 任务 ⊆ P3.01 白名单（qc verdicts：verdict 含"白名单"且非"不适用"）
whitelist = sorted(t for t, v in qc301["verdicts"].items() if "白名单" in v["verdict"])
cfg_tasks = sorted(k for k, v in p["tasks"].items() if v.get("kind") != "umbrella")
extra = [t for t in cfg_tasks if t not in whitelist]
if extra:
    die(f"V3 配置任务超出 P3.01 白名单: {extra}（白名单={whitelist}）")
if len(cfg_tasks) != 6:
    die(f"V3 配置任务数 {len(cfg_tasks)} != 6")

# V4 统计参数一致且无待确认残留
u = ep["aggregation_and_uncertainty"]
s = p["stats"]
if u["resampling_B"] != s["bootstrap_B"] or u["confidence_level"] != s["ci"]["level"]:
    die("V4 bootstrap/CI 与 evaluation_protocol 不一致")
if list(u["seeds"]) != list(s["probe_seed_pool"]):
    die("V4 种子池与 evaluation_protocol.seeds 不一致")
if "Holm" not in str(u["multiplicity"]) or "Holm" not in str(s["multiplicity"]):
    die("V4 multiplicity 不一致")
# 各主对比族声明的种子 ⊆ 种子池
for fam in s["primary_families"]:
    used = s["seed_usage"].get(fam["family"])
    if not used or not set(used) <= set(s["probe_seed_pool"]):
        die(f"V4 主对比族 {fam['family']} 种子声明缺失或越池: {used}")
if [f["family"] for f in s["primary_families"]] != \
        ["layer_readability", "capacity_gain", "l2_vs_baselines"]:
    die("V4 主对比族与冻结定义不符")
ep_text = open(os.path.join(ROOT, "configs/evaluation_protocol.yaml")).read()
if "待确认·未冻结" in ep_text:
    die("V4 evaluation_protocol 仍存在'待确认·未冻结'标注")
if "formulas_frozen_params_pending" in ep_text:
    die("V4 evaluation_protocol meta.status 仍为待冻结态")

# V5 选择纪律：无 conf/holdout 数据通道
for section in ("selection", "readers", "tasks", "layers"):
    body = {k: v for k, v in p.get(section, {}).items() if k != "forbidden"}
    blob = json.dumps(body, ensure_ascii=False)
    for bad in ("confirmation 集读取", "final_holdout 数据", "confirmation 数据", "final_holdout 标签"):
        if bad in blob:
            die(f"V5 {section} 出现禁止数据通道字样: {bad}")
if "confirmation" in json.dumps({k: v for k, v in p["selection"].items() if k != "forbidden"},
                                ensure_ascii=False):
    die("V5 selection（禁止条款除外）提及 confirmation")

# V6 L2 两阶段
l2 = p["l2_two_stage"]
sa, sb = l2["stage_a_selection_universe"], l2["stage_b_final_universe"]
smp = sa.get("sampling")
if not isinstance(smp, dict):
    die("V6 stage_a.sampling 必须为结构化字段")
for k, want in (("sort_key", "accession 升序"), ("seed", 2026), ("n", 10000), ("replacement", False)):
    if smp.get(k) != want:
        die(f"V6 stage_a.sampling.{k} = {smp.get(k)!r} != {want!r}")
if "47,604" not in smp.get("frame", ""):
    die(f"V6 stage_a 抽样框架非 dev 簇代表口径: {smp.get('frame')}")
if "343,682" not in sb["background"]:
    die(f"V6 stage_b 宇宙非冻结全宇宙: {sb['background']}")

# V7 指标词汇：主指标字符串按分隔符切段后，每个含拉丁字母的切段必须精确为协议注册指标名
import re as _re
NAMES = {"AUROC", "AUPRC", "Macro-F1", "residue AUPRC", "recall_at_k",
         "enrichment_at_k", "positive_rank_percentile", "区段 IoU"}


def metric_ok(pm: str) -> bool:
    pm = _re.sub(r"（[^）]*）", "", pm)      # 先剥离全角括号注释
    pm = _re.sub(r"\([^)]*\)", "", pm)       # 半角括号注释
    ok = False
    for part in _re.split(r"[,，、;；+]", pm):
        t = part.split("（")[0].split("(")[0]
        if "=" in t:
            t = t.split("=", 1)[1]
        t = t.strip()
        if not t:
            continue
        if t in NAMES:
            ok = True
        elif _re.search(r"[A-Za-z]", t):
            return False          # 含拉丁字母但非注册指标名（如 "AUROC-foo"）一律拒绝
    return ok


for t, v in p["tasks"].items():
    pm = v.get("primary_metric") or v.get("primary_metrics")
    pms = [pm] if isinstance(pm, str) else list(pm or [])
    for m in pms:
        if not metric_ok(m):
            die(f"V7 {t} 主指标段未通过严格匹配: {m}")

print("[p302] ALL PASSED: V1–V7（编码器/层/白名单/统计参数/选择纪律/L2 两阶段/指标词汇）")
