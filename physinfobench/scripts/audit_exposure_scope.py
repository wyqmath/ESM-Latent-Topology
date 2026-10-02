#!/usr/bin/env python3
"""P1.19 历史曝光范围审计（96 对折叠转换样本）。

区分并逐对登记四个口径：
  (1) 旧 v5 划分表收录（task_sample_splits_v5）；
  (2) 旧划分表全版本收录（v1–v5 各版本分别）；
  (3) 前项目 legacy 曝光登记（benchmark_v0.1/samples.tsv legacy_exposure 列
      + legacy_exposure.tsv 的 fold_switching 端点行，来源 91_ESM-Latent-Topology_On-Hold）；
  (4) 旧项目产物使用（登记/复核计算/规则记录/试算/纠错/报告/交付文档/参考文献/日志/
      DSSP 输出文件名）——只登记"出现"，不自动等同评价曝光或影响方法选择。

评价使用单独核验：acceptance_report.md 须含 formal_model_evaluation=false（否则 die）。
只读审计：不写任何旧项目文件；新项目产物仅限 reports/fs_three_layer/ 与本脚本。
"""
import csv
import datetime
import glob
import json
import os
import re
import sys

NEW_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
OLD = "/Users/yuan/Documents/Codex/2026-09-08/jie/benchmark_step1"
DELIV = "/Users/yuan/Documents/Codex/2026-09-08/jie/deliverables"
OUT_TSV = os.path.join(NEW_ROOT, "reports/fs_three_layer/exposure_scope_audit.tsv")
OUT_QC = os.path.join(NEW_ROOT, "reports/fs_three_layer/exposure_scope_audit_qc.json")
RUN_TS = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

TEXT_EXT = {".tsv", ".md", ".json", ".txt", ".sha256", ".csv"}


def die(msg):
    print(f"[P1.19 FATAL] {msg}", file=sys.stderr)
    sys.exit(1)


def read_tsv(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f, delimiter="\t"))


# ---- 1) 新项目 96 对名单 ----
rows = read_tsv(os.path.join(NEW_ROOT, "data/curated/fold_switch_global.tsv"))
if len(rows) != 96:
    die(f"fold_switch_global.tsv rows={len(rows)} != 96")
pairs = {}
for r in rows:
    pid = r["pair_id"]
    m = re.match(r"porter_(\d+)_", pid)
    if not m:
        die(f"pair_id 无法解析 porter 序号: {pid}")
    n = int(m.group(1))
    pairs[pid] = {
        "n": n, "pdb_a": r["pdb_a"].lower(), "pdb_b": r["pdb_b"].lower(),
        "chain_a": r["chain_a"], "chain_b": r["chain_b"], "tier": r["tier"],
        "split": {}, "trial10": "", "corrections": "",
        "legacy": "", "legacy_endpoints": "", "v01_split_status": "",
        "v01_release_status": "", "v01_evidence_quality": "",
        "named_hits": {c: [] for c in []}, "pdb_only_hits": {},
    }
by_n = {p["n"]: p for p in pairs.values()}

# ---- 2) 旧划分表 v1–v5 ----
SPLIT_FILES = [
    ("v1", "task_sample_splits_2026-09-15.tsv"),
    ("v2", "task_sample_splits_v2_2026-09-15.tsv"),
    ("v3", "task_sample_splits_v3_2026-09-15.tsv"),
    ("v4", "task_sample_splits_v4_2026-09-15.tsv"),
    ("v5", "task_sample_splits_v5_2026-09-15.tsv"),
]
split_counts = {}
for ver, fn in SPLIT_FILES:
    seen = {}
    for r in read_tsv(os.path.join(OLD, "manifests", fn)):
        s = r["sample"]
        if not s.startswith("porter_"):
            continue
        if s not in pairs:
            die(f"{fn} 样本不在 96 对名单: {s}")
        splits = seen.setdefault(s, set())
        splits.add(r["split"])
    split_counts[ver] = len(seen)
    for pid, sp in seen.items():
        pairs[pid]["split"][ver] = "+".join(sorted(sp))
if split_counts["v5"] != 35:
    die(f"v5 porter 样本数={split_counts['v5']} != 35（与 P1.17 复核矛盾）")

# ---- 3) v0.1 发布包结构表 ----
samples = read_tsv(os.path.join(OLD, "benchmark_v0.1/samples.tsv"))
fs_samples = [r for r in samples if r["sample_kind"] == "fold_switching_pair"]
if len(fs_samples) != 96:
    die(f"samples.tsv fold_switching_pair 行数={len(fs_samples)} != 96")
legacy_yes = 0
for r in fs_samples:
    pid = r["sample_id"]
    pairs[pid]["legacy"] = r["legacy_exposure"]
    pairs[pid]["v01_split_status"] = r["split_status"]
    pairs[pid]["v01_release_status"] = r["release_status"]
    pairs[pid]["v01_evidence_quality"] = r["source_evidence_quality"]
    if r["legacy_exposure"] == "yes":
        legacy_yes += 1

v01_splits = read_tsv(os.path.join(OLD, "benchmark_v0.1/splits.tsv"))
v01_split_values = {}
for r in v01_splits:
    if r["sample_id"].startswith("porter_"):
        v01_split_values[r["sample_id"]] = r["split"]

legacy_rows = read_tsv(os.path.join(OLD, "benchmark_v0.1/legacy_exposure.tsv"))
legacy_fs_keys = set()
legacy_sources = set()
for r in legacy_rows:
    legacy_sources.add(r["legacy_source"])
    if r["category"] == "fold_switching":
        legacy_fs_keys.add((r["pdb_id"].lower(), r["chain_id"]))
for pid, p in pairs.items():
    hits = sum(1 for pdb, ch in [(p["pdb_a"], p["chain_a"]), (p["pdb_b"], p["chain_b"])]
               if (pdb, ch) in legacy_fs_keys)
    pairs[pid]["legacy_endpoints"] = f"{hits}/2"

# ---- 4) trial10 / corrections ----
for r in read_tsv(os.path.join(OLD, "manifests/fold_switching_trial10.tsv")):
    pid = r["pair_id"]
    if pid not in pairs:
        die(f"trial10 pair 不在名单: {pid}")
    pairs[pid]["trial10"] = f"rank{r['trial_rank']}"
corr = json.load(open(os.path.join(OLD, "manifests/data_corrections_2026-09-14.json")))
for k in corr:
    if k in pairs:
        pairs[k]["corrections"] = "adjudicated"

# ---- 5) 内容扫描载体（预登记清单） ----
def files_of(*paths):
    out = []
    for path in paths:
        if os.path.isfile(path):
            out.append(path)
        elif os.path.isdir(path):
            for dirpath, _dirs, names in os.walk(path):
                for nm in names:
                    fp = os.path.join(dirpath, nm)
                    ext = os.path.splitext(nm)[1].lower()
                    if ext in TEXT_EXT and os.path.getsize(fp) <= 3_000_000:
                        out.append(fp)
        elif "*" in path:
            out.extend(glob.glob(path))
    return sorted(set(out))

MAN = os.path.join(OLD, "manifests")
CARRIERS = {
    "registration": files_of(
        os.path.join(MAN, "fold_switching_pair_endpoints.tsv"),
        os.path.join(MAN, "fold_switching_pair_audit.tsv"),
        os.path.join(MAN, "fold_switching_release_eligibility.tsv"),
        os.path.join(MAN, "fold_switching_release_eligibility.tsv.pre_adjudication"),
        os.path.join(MAN, "fold_switching_release_eligibility.tsv.porter35_pre"),
        os.path.join(MAN, "fold_switching_release_eligibility.tsv.parallel_round_2026-09-15"),
    ),
    "computation_review": files_of(
        os.path.join(MAN, "fold_pair_endpoint_sequence_structure_summary.tsv"),
        os.path.join(MAN, "fold_pair_endpoint_sequence_structure_review.tsv"),
        os.path.join(MAN, "fold_pair_endpoint_residue_mapping.tsv"),
        os.path.join(MAN, "chain_validation.tsv"),
        os.path.join(MAN, "dssp_downstream_reconnection.tsv"),
    ),
    "rule_records": files_of(
        os.path.join(MAN, "eligibility_rule_reconciliation_2026-09-13.tsv"),
        os.path.join(MAN, "family_annotation_2026-09-15.tsv"),
        os.path.join(MAN, "family_layer_audit_2026-09-15.json"),
        os.path.join(MAN, "cross_split_audit_2026-09-15.json"),
        os.path.join(MAN, "cross_split_audit_v2_2026-09-15.json"),
        os.path.join(MAN, "cross_split_audit_v3_2026-09-15.json"),
        os.path.join(MAN, "cross_split_audit_v4_2026-09-15.json"),
        os.path.join(MAN, "cross_split_audit_v5_2026-09-15.json"),
    ),
    "sweep": files_of(os.path.join(MAN, "calcineurin_proline_state_sweep_2026-09-15.tsv")),
    "pilot": files_of(os.path.join(OLD, "trial10")),
    "reports_root": sorted(glob.glob(os.path.join(OLD, "*.md"))),
    "v01_release": files_of(os.path.join(OLD, "benchmark_v0.1")),
    "exec_round": files_of(os.path.join(OLD, "execution_round_2026-09-10")),
    "deliverables": sorted(glob.glob(os.path.join(DELIV, "*.md"))),
    "references": files_of(
        os.path.join(OLD, "references_case_2026-09-14"),
        os.path.join(OLD, "references_batch_2026-09-15"),
    ),
    "logs_old": files_of(os.path.join(OLD, "logs")),
}
# DSSP 输出按文件名登记（结构文件不扫内容）
dssp_names = set()
for d in ["dssp_local", "dssp_local_v2"]:
    for fp in glob.glob(os.path.join(OLD, d, "*")):
        dssp_names.add(os.path.basename(fp).split(".")[0].lower())

# calcineurin sweep 的归属以 RUN_LOG.md L222 为证据：该 sweep 是 porter_20（5c1v A/B
# cis/trans）状态核证的一部分；sweep 文件本身只含其余 Q08209 条目（5c1v 0 命中，实测）。
SWEEP_OWNER = "porter_20_5c1vA__5c1vB"
SWEEP_OWNER_EVIDENCE = "RUN_LOG.md L222（swept ALL 20 Q08209 entries; 5c1v 本身 0 命中）"

pdb_re = re.compile(
    "(?i)(?<![0-9a-z])(" + "|".join(sorted({p["pdb_a"] for p in pairs.values()} |
                                            {p["pdb_b"] for p in pairs.values()})) +
    ")(?![0-9a-z])")
porter_re = re.compile(r"(?i)(?<![0-9a-z_])porter_(\d+)(?![0-9])")
pair_ids = list(pairs)

n_files_scanned = 0
bytes_scanned = 0
for cat, flist in CARRIERS.items():
    for fp in flist:
        try:
            text = open(fp, errors="ignore").read()
        except OSError:
            continue
        n_files_scanned += 1
        bytes_scanned += len(text)
        porter_ns = {int(m.group(1)) for m in porter_re.finditer(text)}
        pdb_hits = {m.group(1).lower() for m in pdb_re.finditer(text)}
        base = os.path.relpath(fp, OLD)
        for pid in pair_ids:
            p = pairs[pid]
            named = (pid in text) or (p["n"] in porter_ns)
            pdb_only = (p["pdb_a"] in pdb_hits) or (p["pdb_b"] in pdb_hits)
            if named:
                p.setdefault("named_hits", {}).setdefault(cat, []).append(base)
            elif pdb_only:
                p.setdefault("pdb_only_hits", {}).setdefault(cat, []).append(base)

# ---- 6) 分类（预注册规则） ----
for pid, p in pairs.items():
    if "v5" in p["split"]:
        p["classification"] = "v5_split_member"
    elif p["split"]:
        p["classification"] = "earlier_split_only"
    else:
        p["classification"] = "never_in_split_tables"
    p["dssp_endpoint_files"] = str(sum(1 for pdb in [p["pdb_a"], p["pdb_b"]]
                                       if pdb in dssp_names)) + "/2"

# ---- 7) 评价使用核验 ----
acc = open(os.path.join(OLD, "benchmark_v0.1/acceptance_report.md"), errors="ignore").read()
if "formal_model_evaluation=false" not in acc:
    die("acceptance_report.md 未含 formal_model_evaluation=false，不能支持'未运行正式评价'结论")
if "locked_split=false" not in acc:
    die("acceptance_report.md 未含 locked_split=false")

# ---- 8) 断言与汇总 ----
cls_counts = {}
for p in pairs.values():
    cls_counts[p["classification"]] = cls_counts.get(p["classification"], 0) + 1
non_v5 = [p for p in pairs.values() if p["classification"] != "v5_split_member"]
non_v5_strict = sum(1 for p in non_v5 if p["tier"] == "strict_state_candidate")
if non_v5_strict != 0:
    print(f"[warn] 非 v5 strict 数={non_v5_strict}（预期 0：10 strict 应全在 v5）")

cols = (["pair_id", "porter_n", "tier_new",
         "split_v1", "split_v2", "split_v3", "split_v4", "split_v5",
         "classification", "v01_split_status", "v01_release_status",
         "legacy_exposure", "legacy_endpoint_rows", "legacy_sources",
         "trial10", "corrections", "calcineurin_sweep",
         "named_hit_files_by_category", "pdb_only_file_count_by_category",
         "dssp_endpoint_files", "v01_evidence_quality"])
qc = {
    "run_ts": RUN_TS,
    "pairs": 96,
    "split_unique_counts": split_counts,
    "split_union_unique": sum(1 for p in pairs.values() if p["split"]),
    "classification_counts": cls_counts,
    "legacy_exposure_yes": legacy_yes,
    "legacy_endpoint_covered_pairs": sum(1 for p in pairs.values()
                                         if p["legacy_endpoints"] != "0/2"),
    "legacy_sources": sorted(legacy_sources),
    "v01_splits_porter_rows": len(v01_split_values),
    "v01_split_values_dist": {v: sum(1 for x in v01_split_values.values() if x == v)
                              for v in set(v01_split_values.values())},
    "trial10_pairs": sum(1 for p in pairs.values() if p["trial10"]),
    "corrections_pairs": sum(1 for p in pairs.values() if p["corrections"]),
    "formal_model_evaluation": "false（acceptance_report.md 176-177 行）",
    "files_scanned": n_files_scanned,
    "bytes_scanned": bytes_scanned,
    "carriers": {c: len(f) for c, f in CARRIERS.items()},
}
with open(OUT_TSV, "w", newline="") as f:
    w = csv.writer(f, delimiter="\t", lineterminator="\n")
    w.writerow(cols)
    for pid in sorted(pair_ids, key=lambda x: pairs[x]["n"]):
        p = pairs[pid]
        named = ";".join(f"{c}:{'|'.join(v)}" for c, v in p.get("named_hits", {}).items())
        pdbonly = ";".join(f"{c}:{len(v)}" for c, v in p.get("pdb_only_hits", {}).items())
        w.writerow([pid, p["n"], p["tier"]] +
                   [p["split"].get(v, "-") for v in ["v1", "v2", "v3", "v4", "v5"]] +
                   [p["classification"], p["v01_split_status"], p["v01_release_status"],
                    p["legacy"], p["legacy_endpoints"], ";".join(sorted(legacy_sources)),
                    p["trial10"], p["corrections"],
                    (f"owner({SWEEP_OWNER_EVIDENCE})" if pid == SWEEP_OWNER else "-"),
                    named, pdbonly, p["dssp_endpoint_files"], p["v01_evidence_quality"]])
with open(OUT_QC, "w") as f:
    json.dump(qc, f, ensure_ascii=False, indent=1, sort_keys=True)
print(f"[P1.19] 96 对审计完成：classification={cls_counts} "
      f"split_union={qc['split_union_unique']} legacy_yes={legacy_yes} "
      f"files_scanned={n_files_scanned}")
