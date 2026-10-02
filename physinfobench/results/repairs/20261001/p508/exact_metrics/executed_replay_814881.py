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
import sys
import numpy as np
import sklearn
from sklearn.linear_model import LogisticRegression
import p508_cluster_eval as legacy


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--correction-set", type=Path, required=True)
    ap.add_argument("--residue-map", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    if args.output.exists() and list(args.output.iterdir()):
        ap.error("Refuse nonempty output")
    args.output.mkdir(parents=True, exist_ok=True)
    cohort = json.loads(args.correction_set.read_text())
    keep = set(cohort["kept"])
    names = sorted(keep)
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
    legacy.RES = str(args.output)
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
