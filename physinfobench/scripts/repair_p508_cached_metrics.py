#!/usr/bin/env python3
"""Replay frozen P5.08 CPU fit from existing caches; save supplementary residue predictions.

Uses historical fixed C=.01/h33/dev2279/seed2026 only. No model selection,
new test sampling, or representation extraction. Output must be a new directory.
"""
import argparse
import csv
from datetime import datetime, timezone
import gzip
import hashlib
import json
from pathlib import Path
import platform
import shutil
import sys
import numpy as np
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, balanced_accuracy_score, matthews_corrcoef
import p508_cluster_eval as legacy


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def finalize_outputs(output, names):
    derived_metrics = output / "pooled_metrics.json"
    derived = json.loads(derived_metrics.read_text())
    derived["note"] = "Posthoc supplementary correction; same-dev frozen-parameter fit replay, no selection or new test sampling"
    derived_metrics.write_text(json.dumps(derived, indent=2) + "\n")
    score_by_chain = {}
    with gzip.open(output / "residue_scores.tsv.gz", "rt") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            score_by_chain.setdefault(row["name"], []).append(row)
    exact_rows, pooled_y, pooled_sc = [], [], []
    for name in names:
        records = score_by_chain[name]
        y = np.array([int(r["label_ca_nonobserved"]) for r in records])
        scores = np.array([float(r["score"]) for r in records])
        pred = scores >= .5
        exact_rows.append({"name": name, "n_res": len(y), "base_rate": float(np.mean(y)),
                           "auprc": float(average_precision_score(y, scores)),
                           "balanced_acc": float(balanced_accuracy_score(y, pred)),
                           "mcc": float(matthews_corrcoef(y, pred))})
        pooled_y.append(y); pooled_sc.append(scores)
    with (output / "per_protein_exact.tsv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(exact_rows[0]), delimiter="\t", lineterminator="\n")
        writer.writeheader(); writer.writerows(exact_rows)
    pooled_scores = np.concatenate(pooled_sc)
    perm = []
    for i in range(100):
        rng = np.random.RandomState(2026+i)
        perm.append(float(average_precision_score(np.concatenate([rng.permutation(y) for y in pooled_y]), pooled_scores)))
    exact = {"n_proteins": len(names), "n_residues": sum(len(y) for y in pooled_y),
             "per_protein_auprc_mean": float(np.mean([r["auprc"] for r in exact_rows])),
             "per_protein_base_rate_mean": float(np.mean([r["base_rate"] for r in exact_rows])),
             "per_protein_balanced_acc_mean": float(np.mean([r["balanced_acc"] for r in exact_rows])),
             "per_protein_mcc_mean": float(np.mean([r["mcc"] for r in exact_rows])),
             "pooled_auprc_secondary": float(average_precision_score(np.concatenate(pooled_y), pooled_scores)),
             "pooled_base_rate": float(np.mean(np.concatenate(pooled_y))),
             "pooled_fixed_prediction_permutation100": {"mean": float(np.mean(perm)), "max": float(np.max(perm))},
             "kind": "posthoc identity-verified supplementary correction; frozen parameter devfit replay",
             "label_scope": "CA coordinate non-observation; intrinsic-disorder correspondence unverified"}
    (output / "metrics_exact.json").write_text(json.dumps(exact, indent=2) + "\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--correction-set", type=Path, required=True)
    ap.add_argument("--residue-map", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--finalize-only", action="store_true", help="Recompute exact metrics from already archived residue scores; no fitting")
    args = ap.parse_args()
    if not args.finalize_only and args.output.exists() and list(args.output.iterdir()):
        ap.error("Refuse nonempty output")
    args.output.mkdir(parents=True, exist_ok=True)
    cohort = json.loads(args.correction_set.read_text())
    keep = set(cohort["kept"])
    names = sorted(keep)
    if args.finalize_only:
        finalize_outputs(args.output, names)
        return
    labels = {r["name"]: r["labels"] for r in json.loads(Path(legacy.I, "p508_labels.json").read_text())}
    indexes = {r["name"]: [int(p) for p in r["resid_indices"].split(",")]
               for r in legacy.rd(str(Path(legacy.I, "p508_final_ridx.tsv")))}
    with args.residue_map.open() as f:
        residue_rows = list(csv.DictReader(f, delimiter="\t"))
    residue_by_chain = {}
    for r in residue_rows:
        residue_by_chain.setdefault(r["name"], []).append(r)
    if set(residue_by_chain) != keep:
        raise ValueError("CIF residue map and correction cohort differ")
    for name in names:
        if indexes[name] != list(range(1, len(indexes[name]) + 1)):
            raise ValueError("Historical evaluator's prefix-label assumption is invalid")
        if len(indexes[name]) != len(residue_by_chain[name]):
            raise ValueError("CIF residue map/embedding index length mismatch")
        if [int(r["sequence_index_0based"]) for r in residue_by_chain[name]] != list(range(len(indexes[name]))):
            raise ValueError("CIF residue map sequence-index order differs from cached prefix")
        if [int(r["label_ca_nonobserved"]) for r in residue_by_chain[name]] != labels[name][:len(indexes[name])]:
            raise ValueError("Reconstructed CIF labels and historical frozen labels differ")
    original_rd = legacy.rd
    def corrected_read(path):
        result = original_rd(path)
        if str(path) == str(Path(legacy.I, "p508_emb/extract_manifest.tsv")):
            result = [r for r in result if r["name"] in keep]
            if set(r["name"] for r in result) != keep:
                raise ValueError("Missing identity-verified cached representations")
        return result
    legacy.rd = corrected_read
    legacy.RES = str(args.output / "runner_metrics")
    counter = {"n": 0}
    representation_hashes = {}
    for manifest in [Path(legacy.I, "p303/emb_disorder_dev/extract_manifest.tsv"),
                     Path(legacy.I, "p508_emb/extract_manifest.tsv")]:
        for row in original_rd(str(manifest)):
            if "p508_emb" in str(manifest) and row["name"] not in keep:
                continue
            cache = manifest.parent / (row["key"] + ".npz")
            representation_hashes.setdefault(str(cache), sha(cache))
            if "p508_emb" in str(manifest):
                with np.load(cache, allow_pickle=False) as arr:
                    if arr["resid_indices"].tolist() != indexes[row["name"]]:
                        raise ValueError("Cached residual indices differ from frozen manifest")
    with gzip.open(args.output / "residue_scores.tsv.gz", "wt", newline="") as prediction_file:
        writer = csv.writer(prediction_file, delimiter="\t", lineterminator="\n")
        writer.writerow(["name", "sequence_index_0based", "cif_label_seq_id", "cif_auth_seq_num", "label_ca_nonobserved", "score", "prediction_threshold_0_5"])
        class Recorder(LogisticRegression):
            def fit(self, X, y, *a, **kw):
                result = super().fit(X, y, *a, **kw)
                np.savez(args.output / "frozen_reader_coefficients.npz", coef=self.coef_, intercept=self.intercept_, classes=self.classes_)
                return result
            def predict_proba(self, X):
                if counter["n"] >= len(names):
                    raise ValueError("Unexpected extra evaluation call")
                name = names[counter["n"]]
                probabilities = super().predict_proba(X)
                if len(probabilities) != len(indexes[name]):
                    raise ValueError("Stored index and cached feature length mismatch")
                for index, row, score in zip(indexes[name], residue_by_chain[name], probabilities[:, 1]):
                    writer.writerow([name, index - 1, row["cif_label_seq_id"], row["cif_auth_seq_num"], row["label_ca_nonobserved"], format(float(score), ".17g"), int(score >= .5)])
                counter["n"] += 1
                return probabilities
        legacy.LogisticRegression = Recorder
        legacy.step3_eval(names)
    if counter["n"] != len(names):
        raise ValueError("Incomplete supplementary scoring")
    for filename in ("pooled_metrics.json", "per_protein.tsv", "layer_audit.json"):
        produced = Path(legacy.RES) / filename
        if produced.exists():
            shutil.copy2(produced, args.output / filename)
    finalize_outputs(args.output, names)
    inputs = [args.correction_set, args.residue_map, Path(__file__), Path(legacy.__file__),
              Path(legacy.I, "p508_labels.json"), Path(legacy.I, "p508_final_ridx.tsv"),
              Path(legacy.I, "p303/disorder_resid_idx.tsv"), Path(legacy.ROOT, "data/curated/disorder_masks.tsv")]
    provenance = {"created_at_utc": datetime.now(timezone.utc).isoformat(),
                  "kind": "posthoc identity-verified supplementary correction; frozen devfit replay; no new test data",
                  "parameters": {"layer": 33, "C": .01, "seed": 2026, "class_weight": "balanced", "solver": "liblinear", "threshold": .5},
                  "software": {"python": platform.python_version(), "numpy": np.__version__, "sklearn": sklearn.__version__},
                  "input_sha256": {str(p): sha(p) for p in inputs}, "representation_sha256": representation_hashes,
                  "n_evaluated": len(names), "no_representation_extraction": True}
    (args.output / "replay_provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")


if __name__ == "__main__":
    main()
