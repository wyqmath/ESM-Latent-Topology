#!/usr/bin/env python3
"""import_disorder_v1.py — P1.05 DisProt 无序标签与有效掩码导入

输入：data/raw/disprot/2026-09-22/disprot_api_search.json（完整导出，SHA256 1fdfce…，3337 条目）
输出：
  data/curated/disorder.tsv        标签区段级表（disorder/order 区段 + 条件态/功能区的处置记录）
  data/curated/disorder_masks.tsv  每残基三态记录的区间压缩表示（state ∈ {1,0,NA}，mask∈{0,1}；
                                   区间 [start,end] 闭区间逐残基展开即等价逐残基记录）
口径（对齐 TODO P1.05 与 sample_schema 约定）：
  - 仅 term_namespace=Structural state 计入标签：IDPO:0000002 disorder→state=1；IDPO:0000006 order→state=0
  - Structural transition 条目→conditional_transition，隔离出静态标签（不入三态）
  - 功能命名空间（Molecular function 等）→ excluded_from_labeling，不入标签
  - 重叠冲突（同蛋白 disorder 与 order 区间重叠）→ 重叠残基 mask=0 并逐例登记
  - ec 证据代码随行保留（ec_go/ec_id）；空证据代码单独计数
"""
import csv
import json
from collections import Counter
from datetime import datetime
from pathlib import Path

SRC = Path("data/raw/disprot/2026-09-22/disprot_api_search.json")
OUT_LABELS = Path("data/curated/disorder.tsv")
OUT_MASKS = Path("data/curated/disorder_masks.tsv")
CREATED_AT = datetime.now().strftime("%Y-%m-%d %H:%M")
SOURCE_VERSION = "DisProt API export 2026-09-22 sha256:1fdfce26a3ea9b626e63559f36abd86e023b50b612dc2c8756acbb08edead140"

LABEL_FIELDS = ["disprot_id", "uniprot_acc", "protein_length", "region_id", "label_class",
                "state", "start", "end", "term_id", "term_name", "term_namespace",
                "ec_go", "ec_id", "ec_name", "conditions_n", "construct_alterations_n",
                "cross_ref_pdb", "validated", "region_version", "disorder_mask_state",
                "source_version", "created_at"]
MASK_FIELDS = ["uniprot_acc", "disprot_id", "start", "end", "state", "mask",
               "source_region_id", "source_version", "created_at"]


def main() -> int:
    data = json.load(open(SRC))["data"]
    problems, isolated_entries = [], []

    label_rows, mask_rows = [], []
    stats = Counter()
    per_prot = []

    for e in data:
        acc = e["acc"]
        did = e["disprot_id"]
        L = int(e["length"])
        seq = e.get("sequence", "")
        if len(seq) != L:
            problems.append(f"{did}: 序列长度 {len(seq)} != length {L}")
        regions = e.get("regions", [])

        pos_state = {}   # pos -> (state, region_id)
        conflicts = set()
        n_dis = n_ord = n_trans = n_func = 0
        has_transition = has_function = False

        for r in regions:
            ns = r.get("term_namespace", "")
            tid = r.get("term_id", "")
            s, t = int(r["start"]), int(r["end"])
            if not (1 <= s <= t <= L):
                problems.append(f"{did}/{r.get('region_id')}: 区间 [{s},{t}] 越界（length={L}）")
                continue
            common = {
                "disprot_id": did, "uniprot_acc": acc, "protein_length": L,
                "region_id": r.get("region_id", ""), "term_id": tid,
                "term_name": r.get("term_name", ""), "term_namespace": ns,
                "ec_go": r.get("ec_go", ""), "ec_id": r.get("ec_id", ""),
                "ec_name": r.get("ec_name", ""),
                "conditions_n": len(r.get("conditions") or []),
                "construct_alterations_n": len(r.get("construct_alterations") or []),
                "cross_ref_pdb": ";".join(sorted({x["id"] for x in (r.get("cross_refs") or []) if x.get("db") == "PDB"})),
                "validated": r.get("validated", ""), "region_version": r.get("version", ""),
                "source_version": SOURCE_VERSION, "created_at": CREATED_AT,
            }
            if ns == "Structural state" and tid == "IDPO:0000002":
                n_dis += 1
                label_rows.append({**common, "label_class": "disorder", "state": "1",
                                   "start": s, "end": t, "disorder_mask_state": "pending"})
                for p in range(s, t + 1):
                    if p in pos_state and pos_state[p][0] != 1:
                        conflicts.add(p)
                    pos_state[p] = (1, r.get("region_id", ""))
            elif ns == "Structural state" and tid == "IDPO:0000006":
                n_ord += 1
                label_rows.append({**common, "label_class": "order", "state": "0",
                                   "start": s, "end": t, "disorder_mask_state": "pending"})
                for p in range(s, t + 1):
                    if p in pos_state and pos_state[p][0] != 0:
                        conflicts.add(p)
                    pos_state[p] = (0, r.get("region_id", ""))
            elif ns == "Structural state":
                # Structural state 下的其它术语（IDPO:0000003 非紧凑态 / 0000004 熔球态等）：
                # 既非明确无序也非明确有序，按纪律不入三态掩码，仅记录
                stats["regions_other_state"] += 1
                label_rows.append({**common, "label_class": "other_structural_state", "state": "",
                                   "start": s, "end": t, "disorder_mask_state": "excluded_other_structural_state"})
            elif ns == "Structural transition":
                n_trans += 1
                has_transition = True
                label_rows.append({**common, "label_class": "conditional_transition", "state": "",
                                   "start": s, "end": t, "disorder_mask_state": "excluded_conditional"})
            elif ns in ("Molecular function", "Disorder function", "Biological process", "Cellular component"):
                n_func += 1
                has_function = True
                label_rows.append({**common, "label_class": "excluded_from_labeling", "state": "",
                                   "start": s, "end": t, "disorder_mask_state": "excluded_function_namespace"})
            else:
                problems.append(f"{did}: 未知命名空间 {ns}/{tid}")

        if conflicts:
            isolated_entries.append((did, acc, len(conflicts)))
            for p in conflicts:
                pos_state[p] = ("CONFLICT", "")

        # 区间压缩的逐残基三态（run 在状态变化或位置不连续处均切断；间隙必须保持 unknown）
        if pos_state:
            keys = sorted(pos_state)
            runs = []
            cur_state = pos_state[keys[0]][0]
            run_start = prev = keys[0]

            def flush(end_pos):
                mask = "0" if cur_state == "CONFLICT" else "1"
                state_out = "NA" if cur_state == "CONFLICT" else str(cur_state)
                runs.append({"uniprot_acc": acc, "disprot_id": did, "start": run_start,
                             "end": end_pos, "state": state_out, "mask": mask,
                             "source_region_id": "", "source_version": SOURCE_VERSION,
                             "created_at": CREATED_AT})

            for p in keys[1:]:
                st = pos_state[p][0]
                if st != cur_state or p != prev + 1:
                    flush(prev)
                    cur_state, run_start = st, p
                prev = p
            flush(prev)
            mask_rows.extend(runs)

            # 自断言：runs 逐残基展开必须与 pos_state 完全一致
            rebuilt = {}
            for r0 in runs:
                for p in range(int(r0["start"]), int(r0["end"]) + 1):
                    rebuilt[p] = r0["state"]
            assert len(rebuilt) == len(pos_state), f"{did}: run 展开覆盖数不符"
            for p in keys:
                st = pos_state[p][0]
                want = "NA" if st == "CONFLICT" else str(st)
                assert rebuilt[p] == want, f"{did}@{p}: run 展开状态不符 {rebuilt[p]}!={want}"

        n_pos = sum(1 for p in pos_state if pos_state[p][0] == 1 and p not in conflicts)
        n_neg = sum(1 for p in pos_state if pos_state[p][0] == 0 and p not in conflicts)
        n_conf = len(conflicts)
        stats["entries"] += 1
        stats["regions_disorder"] += n_dis
        stats["regions_order"] += n_ord
        stats["regions_transition"] += n_trans
        stats["regions_function"] += n_func
        stats["res_state1"] += n_pos
        stats["res_state0"] += n_neg
        stats["res_conflict"] += n_conf
        stats["res_unknown"] += L - n_pos - n_neg - n_conf
        if has_transition:
            stats["entries_with_transition"] += 1
        if has_function:
            stats["entries_with_function_regions"] += 1
        if n_dis == 0 and n_ord == 0:
            stats["entries_no_static_label"] += 1
        per_prot.append((did, acc, L, n_pos, n_neg, n_conf, n_trans, n_func))

    # 断言
    assert stats["entries"] == 3337, stats["entries"]
    implied1 = sum((int(r["end"]) - int(r["start"]) + 1) for r in mask_rows if r["state"] == "1")
    implied0 = sum((int(r["end"]) - int(r["start"]) + 1) for r in mask_rows if r["state"] == "0")
    impliedNA = sum((int(r["end"]) - int(r["start"]) + 1) for r in mask_rows if r["state"] == "NA")
    assert implied1 == stats["res_state1"], (implied1, stats["res_state1"])
    assert implied0 == stats["res_state0"], (implied0, stats["res_state0"])
    assert impliedNA == stats["res_conflict"], (impliedNA, stats["res_conflict"])
    assert len(mask_rows) == 4981, f"run 数 {len(mask_rows)} != 预期 4981（独立审核复算值）"
    assert stats["res_state1"] + stats["res_state0"] + stats["res_conflict"] + stats["res_unknown"] >= 0

    with open(OUT_LABELS, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=LABEL_FIELDS, delimiter="\t", lineterminator="\n")
        w.writeheader()
        w.writerows(label_rows)
    with open(OUT_MASKS, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=MASK_FIELDS, delimiter="\t", lineterminator="\n")
        w.writeheader()
        w.writerows(mask_rows)

    print("WROTE", OUT_LABELS, len(label_rows), "rows")
    print("WROTE", OUT_MASKS, len(mask_rows), "runs")
    print("stats:", dict(stats))
    print("隔离条目（disorder/order 重叠冲突）:", len(isolated_entries))
    for x in isolated_entries[:10]:
        print("  conflict entry:", x)
    if problems:
        print("PROBLEMS:", len(problems))
        for p in problems[:10]:
            print("  -", p)
        return 1
    print("NO PROBLEMS")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
