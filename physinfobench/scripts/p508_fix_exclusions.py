#!/usr/bin/env python3
"""R5.04 exact-chain CIF accession audit; historical data/results never overwritten.

Unknown or ambiguous identities are quarantined from the identity-verified
descriptive subset. Reuses already saved protein metrics; no reader retraining.
"""
import argparse
import csv
from datetime import datetime
import gzip
import hashlib
import json
from pathlib import Path
import re
from zoneinfo import ZoneInfo
import gemmi

ROOT = Path(__file__).resolve().parents[1]
ACCESSION = re.compile(r"(?:[OPQ][0-9][A-Z0-9]{3}[0-9]|[A-NR-Z][0-9](?:[A-Z][A-Z0-9]{2}[0-9]){1,2})$")


def read_tsv(path):
    with Path(path).open() as f:
        return list(csv.DictReader(f, delimiter="\t"))


def canonical_accession(value):
    base = re.sub(r"-\d+$", "", str(value).strip())
    return base if ACCESSION.fullmatch(base) else None


def resolve_accessions(block, asym):
    entities = {r[1] for r in block.find("_struct_asym.", ["id", "entity_id"]) if r[0] == asym}
    if len(entities) != 1:
        return {"entity_id": "", "accessions": [], "status": "missing_or_ambiguous_asym_entity", "invalid_values": []}
    entity = next(iter(entities))
    accessions, invalid = set(), []
    for r in block.find("_struct_ref.", ["entity_id", "db_name", "pdbx_db_accession"]):
        if r[0] != entity or r[1].upper() not in ("UNP", "UNIPROT"):
            continue
        accession = canonical_accession(r[2])
        if accession:
            accessions.add(accession)
        else:
            invalid.append(r[2])
    status = ("invalid_accession" if invalid else "no_uniprot_reference" if not accessions
              else "ambiguous_multi_accession" if len(accessions) > 1 else "resolved")
    return {"entity_id": entity, "accessions": sorted(accessions), "status": status, "invalid_values": invalid}


def load_block(path):
    if str(path).endswith(".gz"):
        with gzip.open(path, "rt") as f:
            return gemmi.cif.read_string(f.read()).sole_block()
    return gemmi.cif.read(str(path)).sole_block()


def write_tsv(path, records):
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0]), delimiter="\t", lineterminator="\n")
        writer.writeheader(); writer.writerows(records)


def summarize(rs):
    result = {"n_proteins": len(rs), "n_residues": sum(int(r["n_res"]) for r in rs)}
    for col in ["auprc", "base_rate", "balanced_acc", "mcc"]:
        result[col + "_mean"] = sum(float(r[col]) for r in rs) / len(rs) if rs else None
    result["auprc_minus_base_mean"] = sum(float(r["auprc"]) - float(r["base_rate"]) for r in rs) / len(rs) if rs else None
    valid = [float(r["segment_recall_iou030"]) for r in rs if r["segment_recall_iou030"] not in ("", "None")]
    result["segment_recall_mean"] = sum(valid) / len(valid) if valid else None
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input-set", type=Path, default=ROOT / "data/interim/p508_final_set.json")
    ap.add_argument("--raw-dir", type=Path, default=ROOT / "data/raw/rcsb/2026-09-30/p508_cif")
    ap.add_argument("--output", type=Path, default=ROOT / "results/repairs/20261001/p508")
    args = ap.parse_args()
    if args.output.exists() and list(args.output.iterdir()):
        ap.error("Nonempty output directory; refuse overwrite")
    args.output.mkdir(parents=True, exist_ok=True)
    kept = json.loads(args.input_set.read_text())["kept"]
    if len(kept) != len(set(kept)):
        raise ValueError("Duplicate historical evaluation IDs")
    disprot_path = ROOT / "data/curated/disorder.tsv"
    disprot = {canonical_accession(r["uniprot_acc"]) for r in read_tsv(disprot_path)} - {None}
    candidates_path = ROOT / "data/interim/p508_candidate_table.tsv"
    candidates = read_tsv(candidates_path)
    exact = {(r["entry"].upper(), r["asym"]): r for r in candidates}
    if len(exact) != len(candidates):
        raise ValueError("Duplicate candidate(entry,label_asym)")
    legacy = {r["entry"].upper(): r for r in candidates}
    audit, eligible, overlaps, quarantine = [], [], {}, {}
    hashes = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
              for p in [args.input_set, disprot_path, candidates_path]}
    for name in sorted(kept):
        entry, asym = name.split(":", 1)[1].rsplit("_", 1)
        entry = entry.upper()
        row = exact[(entry, asym)]
        path = args.raw_dir / (entry.lower() + ".cif")
        if not path.exists():
            path = args.raw_dir / (entry.lower() + ".cif.gz")
        info = resolve_accessions(load_block(path), asym)
        hashes[str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
        shared = sorted(set(info["accessions"]) & disprot)
        if shared:
            action = "exclude_disprot_accession_overlap"; overlaps[name] = shared
        elif info["status"] != "resolved":
            action = "quarantine_unresolved_identity"; quarantine[name] = info["status"]
        else:
            action = "retain_identity_verified"; eligible.append(name)
        audit.append({"name": name, "entry": entry, "label_asym": asym, "entity_id": info["entity_id"],
                      "true_accessions": ";".join(info["accessions"]), "identity_status": info["status"],
                      "invalid_accession_values": ";".join(info["invalid_values"]),
                      "disprot_overlap": ";".join(shared), "action": action,
                      "legacy_used_asym": legacy[entry]["asym"], "selected_db_code": row["uniprot"],
                      "legacy_used_db_code": legacy[entry]["uniprot"]})
    write_tsv(args.output / "accession_audit.tsv", audit)
    correction = {"kept": eligible, "excluded_accession_overlap": overlaps, "quarantined_identity": quarantine,
                  "original_evaluated_n": len(kept), "kind": "posthoc identity-verified supplementary subset",
                  "policy": "Use actual CIF accessions for exact label_asym/entity; quarantine unknown/ambiguous; exclude any overlap"}
    (args.output / "correction_set.json").write_text(json.dumps(correction, ensure_ascii=False, indent=2) + "\n")
    stored_path = ROOT / "results/p508_disorder_contrast/per_protein.tsv"
    stored_rows = read_tsv(stored_path)
    stored = {r["name"]: r for r in stored_rows}
    if len(stored) != len(stored_rows) or set(stored) != set(kept):
        raise ValueError("Stored protein metrics and historical evaluation IDs differ")
    subset = [stored[n] for n in eligible]
    if not subset:
        raise ValueError("No identity-verified supplementary rows")
    write_tsv(args.output / "per_protein_identity_verified.tsv", subset)
    hashes[str(stored_path.relative_to(ROOT))] = hashlib.sha256(stored_path.read_bytes()).hexdigest()
    summary = {"created_at": datetime.now(ZoneInfo("Asia/Shanghai")).strftime("%Y-%m-%d %H:%M"),
               "timezone": "Asia/Shanghai", "original_105_from_saved_summary": summarize(stored_rows),
               "identity_verified_subset": summarize(subset), "n_accession_overlaps": len(overlaps),
               "n_quarantined": len(quarantine), "actual_overlap_found": bool(overlaps),
               "label_scope": "model1/0 CA coordinate non-observation; intrinsic-disorder correspondence unverified",
               "interpretation": "Posthoc supplementary identity audit; no retraining, new test access or threshold selection",
               "limitations": ["Means recomputed from stored four-decimal per-protein summaries.",
                                "Raw residue prediction scores unavailable locally; no revised pooled metric generated.",
                                "Historical p508_final_set has empty dropped histories after repeated in-place repair; pre-exclusion239 not reconstructed."],
               "input_sha256": hashes}
    (args.output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"input_n": len(kept), "verified_n": len(eligible), "overlaps": overlaps,
                      "quarantine": quarantine, "metrics": summary["identity_verified_subset"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
