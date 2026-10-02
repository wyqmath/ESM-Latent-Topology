#!/usr/bin/env python3
"""Check the current L2 stage-B sequence FASTAs cover the frozen dev representatives."""
import argparse
import csv
import gzip
import hashlib
import json
from pathlib import Path


def fasta_names(path):
    names = []
    with open(path) as source:
        for line in source:
            if line.startswith(">"):
                name = line[1:].strip().split()[0] if line[1:].strip() else ""
                if not name:
                    raise ValueError(f"empty FASTA header: {path}")
                names.append(name)
    if not names:
        raise ValueError(f"no FASTA records: {path}")
    if len(names) != len(set(names)):
        raise ValueError(f"duplicate FASTA headers: {path}")
    return set(names)


def expected_representatives(manifest):
    opener = gzip.open if str(manifest).endswith(".gz") else open
    with opener(manifest, "rt", newline="") as source:
        rows = csv.DictReader(source, delimiter="\t")
        return {r["cluster_rep"] for r in rows
                if r["split"] == "development" and not r["uniprot_accession"].startswith("EP_")}


def audit_union(manifest, base_fasta, delta_fasta):
    expected = expected_representatives(manifest)
    base = fasta_names(base_fasta)
    delta = fasta_names(delta_fasta)
    overlap = base & delta
    missing = expected - base - delta
    delta_extra = delta - expected
    if not expected:
        raise ValueError("frozen development representative set is empty")
    if overlap or missing or delta_extra:
        raise ValueError(
            f"L2 FASTA union mismatch: expected={len(expected)} base={len(base)} delta={len(delta)} "
            f"base_delta_overlap={len(overlap)} missing={len(missing)} delta_extra={len(delta_extra)}; "
            f"examples missing={sorted(missing)[:5]} delta_extra={sorted(delta_extra)[:5]}"
        )
    return {
        "status": "PASS",
        "current_development_representatives": len(expected),
        "base_fasta_records": len(base),
        "base_current_overlap": len(base & expected),
        "base_extra_historical_representatives": len(base - expected),
        "delta_fasta_records": len(delta),
        "delta_current_overlap": len(delta & expected),
        "base_delta_overlap": 0,
        "missing_current_representatives": 0,
        "delta_extra_representatives": 0,
        "manifest_sha256": hashlib.sha256(Path(manifest).read_bytes()).hexdigest(),
        "base_fasta_sha256": hashlib.sha256(Path(base_fasta).read_bytes()).hexdigest(),
        "delta_fasta_sha256": hashlib.sha256(Path(delta_fasta).read_bytes()).hexdigest(),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--base-fasta", required=True)
    parser.add_argument("--delta-fasta", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    result = audit_union(args.manifest, args.base_fasta, args.delta_fasta)
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    tmp = output.with_suffix(output.suffix + ".tmp")
    tmp.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    tmp.replace(output)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
