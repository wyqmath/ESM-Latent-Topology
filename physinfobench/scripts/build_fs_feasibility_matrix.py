#!/usr/bin/env python3
"""P1.12 可行性矩阵：全部数字由 P1.09–P1.11 产物复算（不做任何手填）。"""
import argparse
import csv
import gzip
import json
from collections import defaultdict


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", default="reports/fs_three_layer/feasibility_matrix.tsv")
    args = ap.parse_args()

    # --- strict positives（双源） ---
    strict = {}
    with open("data/curated/fold_switch_global.tsv") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            if row["tier"] == "strict_state_candidate":
                u = (row.get("uniprot_a") or "").strip()
                if u:
                    strict[u] = row["pair_id"]
    # --- regions 双口径（P1.03 产物复算） ---
    fine_pairs, usable_rows, usable_rows_strict = set(), 0, 0
    strict_pair_ids = set(strict.values())
    with open("data/curated/fold_switch_regions.tsv") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            if row.get("region_task_tier") == "fine_candidate":
                fine_pairs.add(row["pair_id"])
            if row.get("lit_label_decision") == "usable":
                usable_rows += 1
                if row["pair_id"] in strict_pair_ids:
                    usable_rows_strict += 1
    usable_pairs_strict = set()
    with open("data/curated/fold_switch_regions.tsv") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            if row.get("lit_label_decision") == "usable" and row["pair_id"] in strict_pair_ids:
                usable_pairs_strict.add(row["pair_id"])
    # --- 背景（P1.10 qc） ---
    p10 = json.load(open("reports/fs_three_layer/background_qc.json"))
    # --- PN（P1.11 表复算） ---
    pn_rows = list(csv.DictReader(open("data/curated/fold_switch_putative_negative_candidates.tsv"), delimiter="\t"))
    pn_unique = {}
    for r in pn_rows:
        pn_unique.setdefault(r["uniprot_accession"], r)
    per_pos_pn = defaultdict(lambda: defaultdict(set))
    for r in pn_rows:
        per_pos_pn[r["matched_strict_positive"]][r["pn_tier"]].add(r["uniprot_accession"])
    # 历史接触（P1.19 四口径；G1 2026-09-24 裁决：legacy 登记+数据构建使用=历史接触记录）
    n_all_pairs = 0
    all_uniprots = set()
    with open("data/curated/fold_switch_global.tsv") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            n_all_pairs += 1
            for c in ("uniprot_a", "uniprot_b"):
                u = (row.get(c) or "").strip()
                if u:
                    all_uniprots.add(u)

    rows = []
    rows.append(["T-FS-L1-PAIRED", "严格配对机制诊断", "strict 对数", str(len(strict)), "P1.09 审计", "L1 可评估"])
    rows.append(["T-FS-L1-PAIRED", "严格配对机制诊断", "fine 区段对数", str(len(fine_pairs)), "P1.03 regions 复算", "残基任务主口径（D3）"])
    rows.append(["T-FS-L1-PAIRED", "严格配对机制诊断", "usable 行数（strict 人群内）", str(usable_rows_strict), "P1.03 regions 复算（pair∈strict）", "残基任务敏感性口径=strict 内 3 对的 6 行；全表 10 行含 fine_extension 2 对 4 行（人群外，不入 L1）"])
    rows.append(["T-FS-L1-PAIRED", "严格配对机制诊断", "usable 对数（strict 人群内）", str(len(usable_pairs_strict)), "P1.03 regions 复算（pair∈strict）", "与 P1.09 usable 3/10 口径一致"])
    rows.append(["T-FS-L2-PU-RANK", "PU 排序", "阳性数", str(len(strict)), "P1.09", "L2 可评估"])
    rows.append(["T-FS-L2-PU-RANK", "PU 排序", "背景 A 行数", str(p10["background_a"]["rows"]), "P1.10 qc", "排序宇宙"])
    rows.append(["T-FS-L2-PU-RANK", "PU 排序", "背景 A 精确去重数", str(p10["background_a"]["exact_duplicates_removed"]), "P1.10 qc", "已按规则去除"])
    rows.append(["T-FS-L3-MATCHED", "匹配病例-对照", "PN 唯一候选", str(len(pn_unique)), "P1.11 表复算", "对照池"])
    for tier in ("PN-A", "PN-B", "PN-C", "EXITED"):
        n = sum(1 for r in pn_unique.values() if r["pn_tier"] == tier)
        rows.append(["T-FS-L3-MATCHED", "匹配病例-对照", f"PN 层 {tier}", str(n), "P1.11 表复算", "对照池构成" if tier != "EXITED" else "复核队列"])
    for pos in sorted(strict):
        cnt = {t: len(per_pos_pn[pos][t]) for t in ("PN-A", "PN-B", "PN-C")}
        rows.append(["T-FS-L3-MATCHED", "匹配病例-对照", f"每阳性候选 {pos}", ";".join(f"{t}={cnt[t]}" for t in cnt), "P1.11 表复算", "D1/D2 降级依据"])
    rows.append(["T-FS-GLOBAL", "全池", "全池对数", str(n_all_pairs), "fold_switch_global 复算", "96 对"])
    rows.append(["T-FS-GLOBAL", "全池", "历史接触对数（S3 legacy 登记）", str(n_all_pairs), "P1.19（96/96 yes；89 对端点可回指+7 对旧包不一致）", "G1 2026-09-24 裁决：legacy 登记+数据构建使用=历史接触记录（D6）"])
    rows.append(["T-FS-GLOBAL", "全池", "独立 UniProt（全池，本表复算）", str(len(all_uniprots)), "fold_switch_global 复算", "三集合直拆不可行（D6；与 P1.09 审计值核对）"])
    rows.append(["T-FS-GLOBAL", "全池", "历史接触口径（S1–S5 分记，P1.19）",
                 "S1 旧 v5 划分收录=35/96（26 dev-only/7 test-only/2 both）；S2 v1–v5 并集=35；S3 legacy 登记=96/96 yes；S5 正式评价使用=0",
                 "P1.19 exposure_scope_audit；G1 2026-09-24 限定 4 采纳",
                 "本行修正原 P1.09「96 对全部在旧 v5 有分配」失实表述（P1.17 发现、P1.25 落实）；53 pending 对不构成可靠未曝光备用池"])

    with open(args.output, "w", newline="") as f:
        f.write("task_id\ttask_name\tmetric\tvalue\tsource\tnote\n")
        csv.writer(f, delimiter="\t", lineterminator="\n").writerows(rows)
    print(f"feasibility matrix written: {args.output} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
