#!/usr/bin/env python3
"""Reproduce claim-scope checks without modifying historical predictions or metrics.

Uses stored P5.07 predictions, P5.08 protein summaries and TODO task trees.
The fixed-prediction permutation diagnostic is not a full retraining/selection null.
"""
from pathlib import Path
from collections import Counter, defaultdict
import csv
import hashlib
import json
import re
import subprocess
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/repairs/20261001/claims"


def rows(path):
    with path.open() as f:
        return list(csv.DictReader(f, delimiter="\t"))


def stats(yt, yp, classes):
    cm = [[sum(t == a and p == b for t, p in zip(yt, yp)) for b in classes] for a in classes]
    per = {}
    for i, t in enumerate(classes):
        tp = cm[i][i]
        actual, predicted = sum(cm[i]), sum(r[i] for r in cm)
        per[t] = {"n": actual, "recall": tp / actual if actual else None,
                  "precision": tp / predicted if predicted else 0,
                  "f1": 2 * tp / (actual + predicted) if actual + predicted else 0}
    return {"confusion": cm, "per_type": per,
            "macro_f1": sum(p["f1"] for p in per.values()) / len(classes),
            "balanced_accuracy": sum(p["recall"] for p in per.values()) / len(classes)}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    p = rows(ROOT / "results/p507_type_probe_v2/lco_predictions.tsv")
    classes = ["3_1", "4_1", "5_2"]
    yt, yp = [r["true"] for r in p], [r["pred"] for r in p]
    members = defaultdict(list)
    for r in p:
        members[r["component"]].append(r)
    component_true, component_pred = [], []
    for component, rs in members.items():
        assert len({r["true"] for r in rs}) == 1, component
        component_true.append(rs[0]["true"])
        component_pred.append(Counter(r["pred"] for r in rs).most_common(1)[0][0])
    chain = stats(yt, yp, classes)
    group = stats(component_true, component_pred, classes)
    selection_stat = float(np.mean([
        np.mean([(r["true"] == "3_1") == (r["pred"] == "3_1") for r in rs])
        for rs in members.values()]))
    all3_selection_stat = sum(r[0]["true"] == "3_1" for r in members.values()) / len(members)
    binary = stats(["3_1" if t == "3_1" else "other" for t in component_true],
                   ["3_1" if t == "3_1" else "other" for t in component_pred], ["3_1", "other"])
    t3 = [p == "3_1" for t, p in zip(component_true, component_pred) if t == "3_1"]
    rng = np.random.RandomState(2026)
    boot = [float(np.mean([t3[i] for i in rng.randint(0, len(t3), len(t3))])) for _ in range(2000)]
    ids = sorted(members)
    types = [members[c][0]["true"] for c in ids]
    perm = []
    for i in range(100):
        shuffled = np.random.RandomState(2026 + i).permutation(types)
        yperm = dict(zip(ids, shuffled))
        perm.append(stats([yperm[r["component"]] for r in p], yp, classes)["macro_f1"])
    n_ge = sum(v >= chain["macro_f1"] for v in perm)
    type_output = {"chains": chain, "components_majority_vote": group,
                   "component_binary_balanced_accuracy": binary["balanced_accuracy"],
                   "selected_score_actual_definition": "mean of within-component chain binary accuracy",
                   "selected_score_recomputed": selection_stat,
                   "all_3_1_baseline_same_selection_score": all3_selection_stat,
                   "component_recall_boot95": np.percentile(boot, [2.5, 97.5]).tolist(),
                   "fixed_prediction_component_permutation": {
                       "n": len(perm), "mean": float(np.mean(perm)), "max": float(np.max(perm)),
                       "n_at_least_observed": n_ge, "plus_one_p": (n_ge + 1) / (len(perm) + 1),
                       "scope": "Fixed selected predictions; no retraining or model-selection correction; diagnostic only"},
                   "issues": ["same nonnested LCO results used for layer/C selection and reporting",
                              "graph components do not establish biological-family counts",
                              "chain recall0.90 paired with component recall CI"]}
    (OUT / "p507_recomputed_scope.json").write_text(json.dumps(type_output, ensure_ascii=False, indent=2) + "\n")
    (OUT / "p507_fixed_prediction_permutation.tsv").write_text("seed\tmacro_f1\n" + "".join(f"{2026+i}\t{v:.12g}\n" for i, v in enumerate(perm)))
    ext = rows(ROOT / "results/p508_disorder_contrast/per_protein.tsv")
    ext_summary = {"n": len(ext), "n_residues": sum(int(r["n_res"]) for r in ext),
                   "auprc_mean": float(np.mean([float(r["auprc"]) for r in ext])),
                   "base_rate_mean": float(np.mean([float(r["base_rate"]) for r in ext])),
                   "balanced_acc_mean": float(np.mean([float(r["balanced_acc"]) for r in ext])),
                   "mcc_mean": float(np.mean([float(r["mcc"]) for r in ext])),
                   "label_implementation": "No model1/0 CA record for (label_asym_id,label_seq_id) in CIF after filters",
                   "claim_scope": "Supplementary X-ray CA coordinate non-observation discrimination; intrinsic-disorder equivalence unverified"}
    (OUT / "p508_scope.json").write_text(json.dumps(ext_summary, ensure_ascii=False, indent=2) + "\n")
    candidate_rows = rows(ROOT / "data/interim/p508_candidate_table.tsv")
    last_by_entry = {r["entry"]: r for r in candidate_rows}
    exact_by_chain = {(r["entry"], r["asym"]): r for r in candidate_rows}
    kept = json.loads((ROOT / "data/interim/p508_final_set.json").read_text())["kept"]
    mapping_checks = []
    accession_pattern = re.compile(r"(?:[OPQ][0-9][A-Z0-9]{3}[0-9]|[A-NR-Z][0-9](?:[A-Z][A-Z0-9]{2}[0-9]){1,2})$")
    for name in kept:
        entry, asym = name.split(":", 1)[1].rsplit("_", 1)
        actual = exact_by_chain[(entry, asym)]
        used = last_by_entry[entry]
        proposed = actual["uniprot"].split("_")[0]
        mapping_checks.append({"name": name, "selected_asym": asym, "used_asym": used["asym"],
                               "selected_db_code": actual["uniprot"], "used_db_code": used["uniprot"],
                               "chain_row_mismatch": asym != used["asym"],
                               "db_code_mismatch": actual["uniprot"] != used["uniprot"],
                               "prefix_nonaccession": bool(proposed) and accession_pattern.fullmatch(proposed) is None})
    (OUT / "p508_accession_mapping_diagnostic.json").write_text(json.dumps({
        "n": len(mapping_checks), "wrong_chain_row": sum(r["chain_row_mismatch"] for r in mapping_checks),
        "different_db_code": sum(r["db_code_mismatch"] for r in mapping_checks),
        "nonaccession_prefix": sum(r["prefix_nonaccession"] for r in mapping_checks),
        "checks": mapping_checks,
        "repair": "Map exact(entry,label_asym)->entity_id->_struct_ref.pdbx_db_accession; recheck exclusions"}, ensure_ascii=False, indent=2) + "\n")
    tasks = {}
    for name in ["TODO.md", "TODO_checked_20260930.md"]:
        text = (ROOT / name).read_text()
        matches = re.findall(r"^- \[([x ])\] \[(P\d\.\d\d)\b", text, re.M)
        tasks[name] = {"total": len(matches), "completed": sum(m[0] == "x" for m in matches),
                       "remaining": [m[1] for m in matches if m[0] != "x"]}
    commits = subprocess.check_output(["git", "log", "--reverse", "--format=%h %s", "5de3858..18c89ea"], cwd=ROOT, text=True)
    tasks["actual_phase5_history_oldest_first"] = commits.splitlines()
    (OUT / "governance_scope.json").write_text(json.dumps(tasks, ensure_ascii=False, indent=2) + "\n")
    files = [ROOT / "results/p507_type_probe_v2/lco_predictions.tsv", ROOT / "results/p508_disorder_contrast/per_protein.tsv",
             ROOT / "scripts/p507_type_probe_v2.py", ROOT / "scripts/p508_build.py"]
    (OUT / "source_hashes.json").write_text(json.dumps({str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}, indent=2) + "\n")
    print(json.dumps({"p507": type_output, "p508": ext_summary, "tasks": tasks}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
