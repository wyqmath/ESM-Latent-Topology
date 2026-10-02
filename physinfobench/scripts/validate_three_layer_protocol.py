#!/usr/bin/env python3
"""P1.12：jsonschema 校验三层协议结构 + 与 claims/evaluation/split 交叉核对。"""
import json
import sys

import yaml

PROTOCOL = "configs/fold_switch_three_layer_protocol.yaml"

SCHEMA = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "type": "object",
    "required": ["meta", "L1_paired_mechanism", "L2_pu_ranking", "L3_matched_case_control",
                 "entry_conditions", "downgrade_rules", "frozen_inputs"],
    "properties": {
        "meta": {"type": "object",
                 "required": ["task_id", "status", "layer_task_ids"],
                 "properties": {"layer_task_ids": {"type": "object", "minProperties": 3}}},
        "L1_paired_mechanism": {"$ref": "#/$defs/layer"},
        "L2_pu_ranking": {"$ref": "#/$defs/layer"},
        "L3_matched_case_control": {"$ref": "#/$defs/layer"},
        "entry_conditions": {"type": "object", "minProperties": 4},
        "downgrade_rules": {"type": "array", "minItems": 1,
                            "items": {"type": "object", "required": ["id", "condition"]}},
        "frozen_inputs": {"type": "object", "minProperties": 3},
    },
    "$defs": {
        "layer": {"type": "object",
                  "required": ["question", "population", "unit", "input", "observed_label",
                               "mask", "metrics", "conclusion_scope"]},
    },
}


def die(msg):
    print(f"VALIDATION FAILED: {msg}", file=sys.stderr)
    sys.exit(1)


def main():
    import jsonschema
    proto = yaml.safe_load(open(PROTOCOL))
    try:
        jsonschema.validate(proto, SCHEMA)
    except jsonschema.ValidationError as e:
        die(f"schema: {e.message} at {list(e.absolute_path)}")
    print("[1] jsonschema structure: pass")

    # ---- 交叉核对 1：层任务 ID 在 claims.yaml 注册 ----
    claims = yaml.safe_load(open("configs/claims.yaml"))
    claim_ids = {t["task_id"] for t in claims["tasks"]}
    need = set(proto["meta"]["layer_task_ids"].values()) | {proto["meta"]["umbrella_task"]}
    missing = need - claim_ids
    if missing:
        die(f"task ids not in claims.yaml: {sorted(missing)}")
    print(f"[2] task ids registered in claims ({len(need)}): pass")

    # ---- 交叉核对 2：L2/L3 指标名与 evaluation_protocol 4b 一致 ----
    ev = yaml.safe_load(open("configs/evaluation_protocol.yaml"))
    tl = ev["three_layer_metrics"]
    l2_metrics = [w for w in str(proto["L2_pu_ranking"]["metrics"]).replace("；", " ").replace(",", " ").split() if "@" in w or w.isalpha()]
    l2_registered = json.dumps(tl["L2_T_FS_L2_PU_RANK"], ensure_ascii=False)
    for m in ("recall_at_k", "enrichment_at_k", "positive_rank_percentile",
              "family_stratified_enrichment", "leave_family_out_enrichment"):
        if m not in l2_registered:
            die(f"L2 metric missing in evaluation_protocol 4b: {m}")
    l3_registered = json.dumps(tl["L3_T_FS_L3_MATCHED"], ensure_ascii=False)
    for m in ("AUROC", "AUPRC", "balanced_accuracy", "MCC", "per_class_precision_recall"):
        if m not in l3_registered:
            die(f"L3 metric missing in evaluation_protocol 4b: {m}")
    calib = tl["L3_T_FS_L3_MATCHED"].get("calibration", "") if isinstance(tl["L3_T_FS_L3_MATCHED"], dict) else str(l3_registered)
    if "disabled" not in str(calib):
        die("L3 calibration not registered as disabled in evaluation_protocol 4b")
    print("[3] metric names cross-checked with evaluation_protocol 4b (incl. calibration disabled): pass")

    # ---- 交叉核对 3：matched_set 绑定与 PU 语义在 split_protocol 3c ----
    sp = yaml.safe_load(open("configs/split_protocol.yaml"))
    t3 = json.dumps(sp["three_layer_split_rules"], ensure_ascii=False)
    for key in ("matched_set", "exposure"):
        if key not in t3:
            die(f"split_protocol three_layer_split_rules missing: {key}")
    print("[4] split_protocol 3c bindings (matched_set/exposure): pass")

    # ---- 交叉核对 4：冻结输入文件存在 ----
    import os
    for k, path in proto["frozen_inputs"].items():
        if not os.path.exists(path):
            die(f"frozen input missing: {k}={path}")
    print("[5] frozen inputs exist on disk: pass")

    # ---- 交叉核对 5：降级规则覆盖 P1.11 的关键可行性事实 ----
    dr = json.dumps(proto["downgrade_rules"], ensure_ascii=False)
    for fact in ("B9W5G6", "P00573", "usable 3/10"):
        if fact not in dr:
            die(f"downgrade rule missing key fact: {fact}")
    print("[6] downgrade rules cover P1.11 feasibility facts: pass")
    print("ALL VALIDATIONS PASSED")


if __name__ == "__main__":
    main()
