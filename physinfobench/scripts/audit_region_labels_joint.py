#!/usr/bin/env python3
"""P1.15 残基标签联合审计（G1 前补证；只读 fold_switch_regions.tsv + 审计表）。

交叉口径：
  fine = region_task_tier == fine_candidate（精细区段，坐标转换完备）
  usable = lit_label_decision == usable（文献证据可用）
  strict = pair ∈ fold_switch_global strict_state_candidate
逐行检查：文献区段字段、坐标转换链（pdf→label→auth）、映射缺口（internal gap/unmapped）、
有效掩码存在性；输出四象限（fine×usable，限 strict 与不限 strict）计数与 ID，
并给出残基主分析/敏感性/案例层的样本口径建议。
铁律：不因 fine=9 对就宣称 9 对可靠标签——fine 只保证坐标，usable 只保证文献证据，主分析需两者同时。
"""
import csv
import json
from collections import defaultdict

REG = "data/curated/fold_switch_regions.tsv"
GLOB = "data/curated/fold_switch_global.tsv"
OUT_TSV = "reports/fs_three_layer/region_label_joint_audit.tsv"
OUT_JSON = "reports/fs_three_layer/region_label_joint_audit_qc.json"


def main():
    strict_pairs = set()
    tiers = {}
    with open(GLOB) as f:
        for row in csv.DictReader(f, delimiter="\t"):
            tiers[row["pair_id"]] = row["tier"]
            if row["tier"] == "strict_state_candidate":
                strict_pairs.add(row["pair_id"])

    rows = list(csv.DictReader(open(REG), delimiter="\t"))
    per_pair = defaultdict(lambda: {"fine": 0, "usable": 0, "both": 0, "rows": 0,
                                    "gap_rows": 0, "unmapped_rows": 0, "ids": []})
    row_issues = []
    for r in rows:
        pid = r["pair_id"]
        fine = r.get("region_task_tier") == "fine_candidate"
        usable = r.get("lit_label_decision") == "usable"
        d = per_pair[pid]
        d["rows"] += 1
        if fine:
            d["fine"] += 1
        if usable:
            d["usable"] += 1
        if fine and usable:
            d["both"] += 1
        # 逐行检查：坐标转换链与缺口
        issues = []
        try:
            ls, le = int(r["label_seq_start"]), int(r["label_seq_end"])
            if le < ls:
                issues.append("label 区间倒置")
        except (ValueError, TypeError):
            issues.append("label 坐标缺失")
        try:
            gap_n = int(r.get("label_interval_internal_gap") or 0)
        except ValueError:
            gap_n = 0
        if gap_n > 0:
            issues.append(f"内部缺口={gap_n}")
            d["gap_rows"] += 1
        try:
            um = int(r.get("unmapped_residue_count") or 0)
            if um > 0:
                d["unmapped_rows"] += 1
                issues.append(f"未映射残基={um}")
        except ValueError:
            pass
        if fine and not (r.get("auth_start") and r.get("auth_end")):
            issues.append("fine 行缺 auth 坐标")
        if issues:
            row_issues.append({"pair_id": pid, "entry_id": r["entry_id"],
                               "region_index": r["region_index"], "issues": ";".join(issues)})

    def quad(pair_filter):
        res = {"fine_only": [], "usable_only": [], "both": [], "neither": []}
        for pid, d in sorted(per_pair.items()):
            if pair_filter and pid not in strict_pairs:
                continue
            if d["both"]:
                res["both"].append(pid)
            elif d["fine"]:
                res["fine_only"].append(pid)
            elif d["usable"]:
                res["usable_only"].append(pid)
            else:
                res["neither"].append(pid)
        return res

    qs = quad(False)
    qs_strict = quad(True)

    out_rows = []
    for label, q in (("全池", qs), ("strict 人群", qs_strict)):
        for k in ("both", "fine_only", "usable_only", "neither"):
            out_rows.append(["scope" if label == "全池" else "scope", label, k, str(len(q[k])),
                             ";".join(q[k]) if len(q[k]) <= 12 else f"{len(q[k])} 对（详见 qc）"])
    with open(OUT_TSV, "w", newline="") as f:
        f.write("dim\tscope\tquadrant\tn_pairs\tpair_ids\n")
        csv.writer(f, delimiter="\t", lineterminator="\n").writerows(out_rows)

    # 行级可用标签计数（strict 内 both 行数 = 残基主分析候选样本）
    strict_both_rows = sum(1 for r in rows if r["pair_id"] in strict_pairs
                           and r.get("region_task_tier") == "fine_candidate"
                           and r.get("lit_label_decision") == "usable")
    strict_fine_rows = sum(1 for r in rows if r["pair_id"] in strict_pairs
                           and r.get("region_task_tier") == "fine_candidate")
    strict_usable_rows = sum(1 for r in rows if r["pair_id"] in strict_pairs
                             and r.get("lit_label_decision") == "usable")
    qc = {
        "generated": __import__("os").popen("TZ=Asia/Shanghai date '+%Y-%m-%d %H:%M'").read().strip(),
        "region_rows_total": len(rows),
        "strict_quadrants": {k: v for k, v in qs_strict.items()},
        "all_pool_quadrants": {k: v for k, v in qs.items()},
        "strict_rows": {"fine": strict_fine_rows, "usable": strict_usable_rows,
                        "both": strict_both_rows},
        "row_issues_count": len(row_issues),
        "row_issues_full": row_issues,
        "assertion": {
            "strict_both_pairs == 3 (P1.09 usable 3/10 口径)": len(qs_strict["both"]) == 3,
            "strict_fine_pairs == 9": len(qs_strict["both"]) + len(qs_strict["fine_only"]) == 9,
        },
    }
    ok = all(qc["assertion"].values())
    with open(OUT_JSON, "w") as f:
        json.dump(qc, f, ensure_ascii=False, indent=1, sort_keys=True)
    print(f"strict 四象限: both={len(qs_strict['both'])} fine_only={len(qs_strict['fine_only'])} "
          f"usable_only={len(qs_strict['usable_only'])} neither={len(qs_strict['neither'])}")
    print(f"strict 行级: fine={strict_fine_rows} usable={strict_usable_rows} both={strict_both_rows}")
    print(f"断言: {qc['assertion']} → {'PASS' if ok else 'FAIL'}")
    if not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
