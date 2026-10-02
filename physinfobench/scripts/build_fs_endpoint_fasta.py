#!/usr/bin/env python3
"""Build versioned endpoint FASTA from the registered observed-chain sequences."""
import argparse
import csv
import os


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary", required=True, help="registered endpoint/structure summary TSV")
    ap.add_argument("--labels", required=True, help="fold_switch_global.tsv")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    sequences = {}
    with open(args.summary, newline="") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            sequences[(row["pdb_id"].lower(), row["requested_chain"].upper())] = row["observed_sequence"]
    labels = []
    with open(args.labels, newline="") as f:
        labels = [r for r in csv.DictReader(f, delimiter="\t")
                  if r["target"] == "1" and r["valid_mask"] == "1"]
    records = {}
    for row in labels:
        for side, pdb, chain in (("A", row["pdb_a"], row["chain_a"]),
                                 ("B", row["pdb_b"], row["chain_b"])):
            name = f"{row['pair_id']}_{side}"
            key = (pdb.lower(), chain.upper())
            if key not in sequences or not sequences[key]:
                raise ValueError(f"Missing observed endpoint sequence: {key}")
            records[name] = sequences[key]
    if len(labels) != 10 or len(records) != 20:
        raise ValueError(f"Expected 10 positive pairs and 20 endpoint records; got {len(labels)}, {len(records)}")
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    tmp = args.out + ".tmp"
    with open(tmp, "w") as out:
        for name, seq in sorted(records.items()):
            out.write(f">{name}\n{seq}\n")
    os.replace(tmp, args.out)
    print(f"saved {len(records)} endpoint records from {len(labels)} positive pairs: {args.out}")


if __name__ == "__main__":
    main()
