#!/usr/bin/env python3
"""validate_schema_examples.py — P0.04 schema 示例校验器

用法：python3 scripts/validate_schema_examples.py [--schema configs/sample_schema.yaml]
行为：按 sample_schema.yaml 的 checks 定义校验 examples.valid（应全部通过）与
examples.invalid（每条应触发且仅触发其 expect_rule）；全部符合则退出码 0。
仅验证 schema 示例，不读取、不写入任何科研数据。
"""
import argparse
import sys
from pathlib import Path

import yaml


def load_schema(path: Path):
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def build_db(schema):
    """由 valid 示例构建表数据库：{table_name: [row, ...]}"""
    db = {name: [] for name in schema["tables"]}
    for ex in schema["examples"]["valid"]:
        db[ex["table"]].append(dict(ex["row"]))
    return db


def iter_violations(db, schema, table, row):
    """返回该行触发的 rule_id 集合（不修改 db 的调用方式：传入包含该行的 db）。"""
    rules = set()
    spec = schema["tables"][table]
    pk = spec["primary_key"]

    # CHK_PK_UNIQUE：主键在本表内出现多于一次
    if sum(1 for r in db[table] if str(r.get(pk)) == str(row.get(pk))) > 1:
        rules.add("CHK_PK_UNIQUE")

    # 通用外键（pair_group_id 由 CHK_PAIR_INTEGRITY 专查）
    for col, ref in (spec.get("foreign_keys") or {}).items():
        ref_table, ref_col = ref.split(".")
        val = row.get(col)
        if val is not None and not any(str(r.get(ref_col)) == str(val) for r in db[ref_table]):
            rules.add("CHK_FK_EXISTS")

    if table == "samples":
        # CHK_PAIR_INTEGRITY
        pg = row.get("pair_group_id")
        if pg is not None and not any(str(r.get("pair_group_id")) == str(pg) for r in db["pairs"]):
            rules.add("CHK_PAIR_INTEGRITY")
        # CHK_UNIT_REQUIRED
        cval, cunit = row.get("condition_value"), row.get("condition_unit")
        if cval is not None and cunit is None:
            try:
                float(str(cval))
                rules.add("CHK_UNIT_REQUIRED")
            except (TypeError, ValueError):
                pass
        # CHK_UNKNOWN_MASK
        target, mask = row.get("target"), row.get("valid_mask")
        if (target is None and mask == 1) or mask is None:
            rules.add("CHK_UNKNOWN_MASK")
        # CHK_NEGATIVE_EVIDENCE：真实阴性必须有 sample 级证据
        if str(target) == "0" and mask == 1:
            has_ev = any(
                e.get("subject_kind") == "sample" and str(e.get("subject_id")) == str(row.get("sample_id"))
                for e in db["evidence"]
            )
            if not has_ev:
                rules.add("CHK_NEGATIVE_EVIDENCE")
        # CHK_EXCLUSION_REASON
        status = row.get("inclusion_status")
        if status in ("excluded", "isolated") and not row.get("exclusion_reason"):
            rules.add("CHK_EXCLUSION_REASON")

    if table == "residue_labels":
        # CHK_UNKNOWN_MASK（残基表同样适用）
        target, mask = row.get("target"), row.get("mask")
        if (target is None and mask == 1) or mask is None:
            rules.add("CHK_UNKNOWN_MASK")
        # CHK_INTERVAL_BOUNDS
        s = row.get("standard_start")
        e = row.get("standard_end")
        sample = next((r for r in db["samples"] if str(r.get("sample_id")) == str(row.get("sample_id"))), None)
        if sample is not None:
            protein = next(
                (p for p in db["proteins"] if str(p.get("protein_id")) == str(sample.get("protein_id"))), None
            )
            if protein is not None and s is not None and e is not None:
                seq_len = len(protein["standard_sequence"])
                if not (1 <= s <= e <= seq_len):
                    rules.add("CHK_INTERVAL_BOUNDS")
            # 样本/蛋白查不到时归 CHK_FK_EXISTS（该行外键检查已覆盖），不重复归因区间错误
        else:
            rules.add("CHK_INTERVAL_BOUNDS")
        # 负例残基同样需要证据（并入 CHK_NEGATIVE_EVIDENCE 语义）
        if str(target) == "0" and mask == 1 and not row.get("evidence_id"):
            rules.add("CHK_NEGATIVE_EVIDENCE")

    return rules


def check_spec_integrity(schema):
    """列规格完整性：type/nullable/description 齐备；外键引用存在。"""
    problems = []
    tables = schema["tables"]
    for tname, spec in tables.items():
        if "primary_key" not in spec or spec["primary_key"] not in spec["columns"]:
            problems.append(f"{tname}: primary_key 缺失或未在 columns 定义")
        for cname, cspec in spec["columns"].items():
            for key in ("type", "nullable", "description"):
                if key not in cspec:
                    problems.append(f"{tname}.{cname}: 缺少 {key}")
        for col, ref in (spec.get("foreign_keys") or {}).items():
            rt, rc = ref.split(".")
            if rt not in tables or rc not in tables[rt]["columns"]:
                problems.append(f"{tname}.{col}: 外键引用 {ref} 不存在")
            if col not in spec["columns"]:
                problems.append(f"{tname}.{col}: 外键列未在 columns 定义")
    allowed_decl = [
        (f"{t}.{c}", cspec["allowed"])
        for t, spec in tables.items()
        for c, cspec in spec["columns"].items()
        if "allowed" in cspec
    ]
    for name, allowed in allowed_decl:
        if not isinstance(allowed, list) or not allowed:
            problems.append(f"{name}: allowed 必须为非空列表")
    return problems


def main():
    ap = argparse.ArgumentParser(description="Validate sample_schema.yaml example records")
    ap.add_argument("--schema", default="configs/sample_schema.yaml")
    args = ap.parse_args()

    schema = load_schema(Path(args.schema))
    failures = []

    spec_problems = check_spec_integrity(schema)
    if spec_problems:
        failures.append(("SPEC_INTEGRITY", spec_problems))

    declared_checks = {c["id"] for c in schema["checks"]}
    implemented = {
        "CHK_PK_UNIQUE", "CHK_FK_EXISTS", "CHK_INTERVAL_BOUNDS", "CHK_UNIT_REQUIRED",
        "CHK_UNKNOWN_MASK", "CHK_NEGATIVE_EVIDENCE", "CHK_EXCLUSION_REASON", "CHK_PAIR_INTEGRITY",
    }
    missing = declared_checks - implemented
    if missing:
        failures.append(("CHECKS_IMPLEMENTED", sorted(missing)))

    db = build_db(schema)
    # valid 示例已在 build_db 中入库，直接校验；重复追加会误报主键冲突
    for ex in schema["examples"]["valid"]:
        t, row = ex["table"], ex["row"]
        v = iter_violations(db, schema, t, row)
        if v:
            failures.append((f"VALID:{row.get(next(iter(schema['tables'][t]['columns'])))}", sorted(v)))

    for ex in schema["examples"]["invalid"]:
        t, row, expect = ex["table"], ex["row"], ex["expect_rule"]
        db[t].append(row)
        v = iter_violations(db, schema, t, row)
        db[t].pop()
        if v != {expect}:
            failures.append((f"INVALID:{ex['row'].get(schema['tables'][t]['primary_key'])}",
                             f"expect {expect}, got {sorted(v) if v else '无违反'}"))

    if failures:
        print("VALIDATION FAILED")
        for name, detail in failures:
            print(f"  - {name}: {detail}")
        return 1
    n_valid = len(schema["examples"]["valid"])
    n_invalid = len(schema["examples"]["invalid"])
    print(f"VALIDATION PASSED: {n_valid} valid examples all clean; "
          f"{n_invalid} invalid examples each triggered exactly its expected rule; "
          f"spec integrity OK; {len(declared_checks)} checks implemented.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
