#!/usr/bin/env python3
"""Validate all fixed P508 development/external cache identities and residue order."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import numpy as np
from p508_cluster_eval import residual_layer, EXPECTED_MODEL, EXPECTED_REVISION


def read(path):
    with Path(path).open() as f:
        return list(csv.DictReader(f, delimiter="\t"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--correction-set", type=Path, required=True)
    parser.add_argument("--residue-map", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Refuse to overwrite cache audit")
    keep = set(json.loads(args.correction_set.read_text())["kept"])
    map_by_name = {}
    for row in read(args.residue_map):
        map_by_name.setdefault(row["name"], []).append(int(row["sequence_index_0based"]))
    audit = []
    for area, folder, index_file in [
            ("development", "p303/emb_disorder_dev", "p303/disorder_resid_idx.tsv"),
            ("external", "p508_emb", "p508_final_ridx.tsv")]:
        directory = args.root / "data/interim" / folder
        rows = read(directory / "extract_manifest.tsv")
        index = {r["name"]: [int(x) for x in r["resid_indices"].split(",")]
                 for r in read(args.root / "data/interim" / index_file)}
        count = 0
        for row in rows:
            if area == "external" and row["name"] not in keep:
                continue
            with np.load(directory / (row["key"] + ".npz"), allow_pickle=False) as archive:
                _, metadata = residual_layer(archive, row, 33)
                if archive["resid_indices"].tolist() != index[row["name"]]:
                    raise ValueError(f"Frozen index/cache order mismatch: {row['name']}")
                if area == "external" and map_by_name[row["name"]] != [i-1 for i in index[row["name"]]]:
                    raise ValueError("CIF sequence-index order mismatch")
                audit.append(dict(metadata, name=row["name"], cohort=area, index_order_checked=True))
                count += 1
        if count != (2279 if area == "development" else len(keep)):
            raise ValueError("Unexpected fixed cohort manifest size")
    qc = {"status": "PASS", "target_layer": 33, "model": EXPECTED_MODEL, "revision": EXPECTED_REVISION,
          "development_n": 2279, "external_n": len(keep), "all_model_revision_and_manifest_metadata_checked": True,
          "all_cached_residue_orders_match_frozen_tables": True, "external_cif_sequence_index_order_checked": True,
          "source_script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), "cache_checks": audit}
    args.output.write_text(json.dumps(qc, indent=2) + "\n")
    print(json.dumps({k:v for k,v in qc.items() if k != "cache_checks"}, indent=2))


if __name__ == "__main__":
    main()
