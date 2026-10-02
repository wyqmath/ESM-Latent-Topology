#!/usr/bin/env python3
"""Metadata-only exact-chain audit of all239 historical candidates; no expanded evaluation."""
import csv
from collections import defaultdict
import hashlib
import json
from pathlib import Path
from p508_fix_exclusions import ROOT, resolve_accessions, load_block, canonical_accession, read_tsv, write_tsv
from p508_build import THREE_TO_ONE


def main():
    out = ROOT / "results/repairs/20261001/p508"
    out.mkdir(parents=True, exist_ok=True)
    disprot = {canonical_accession(r["uniprot_acc"]) for r in read_tsv(ROOT / "data/curated/disorder.tsv")} - {None}
    verified = set(json.loads((out / "correction_set.json").read_text())["kept"])
    historical = set(json.loads((ROOT / "data/interim/p508_final_set.json").read_text())["kept"])
    selected = json.loads((ROOT / "data/interim/p508_labels.json").read_text())
    frozen_sequences = {}; sequence_name = None
    for line in (ROOT / "data/interim/p508_seqs.fa").read_text().splitlines():
        if line.startswith(">"):
            sequence_name = line[1:].strip(); frozen_sequences[sequence_name] = ""
        else:
            frozen_sequences[sequence_name] += line.strip()
    input_hashes = {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
                    for path in [ROOT / "data/interim/p508_labels.json", ROOT / "data/interim/p508_seqs.fa",
                                 ROOT / "data/curated/disorder.tsv", out / "correction_set.json", Path(__file__)]}
    if len({r["name"] for r in selected}) != len(selected):
        raise ValueError("Duplicate historical candidate chain name")
    audit, residues = [], []
    for record in selected:
        name, entry, asym = record["name"], record["entry"], record["asym"]
        if name != f"p508:{entry}_{asym}":
            raise ValueError("Candidate chain key does not match(entry,asym)")
        path = ROOT / "data/raw/rcsb/2026-09-30/p508_cif" / (entry.lower() + ".cif")
        if not path.exists():
            path = Path(str(path) + ".gz")
        block = load_block(path)
        input_hashes[str(path.relative_to(ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
        mapping = resolve_accessions(block, asym)
        overlap = sorted(set(mapping["accessions"]) & disprot)
        audit.append({"name": name, "entity_id": mapping["entity_id"], "identity_status": mapping["status"],
                      "accessions": ";".join(mapping["accessions"]), "disprot_overlap": ";".join(overlap),
                      "identity_action": "exclude_overlap" if overlap else "quarantine" if mapping["status"] != "resolved" else "verified",
                      "historically_evaluated": int(name in historical)})
        if name not in verified:
            continue
        observed = {(r[0], r[1]) for r in block.find("_atom_site.", ["label_asym_id", "label_seq_id", "label_atom_id", "pdbx_PDB_model_num"])
                    if r[2] == "CA" and r[3] in ("0", "1")}
        labels, seq = [], []
        for r in block.find("_pdbx_poly_seq_scheme.", ["asym_id", "seq_id", "mon_id", "auth_seq_num", "hetero"]):
            if r[0] != asym or r[4] == "y" or r[2] not in THREE_TO_ONE:
                continue
            label = int((asym, r[1]) not in observed)
            index = len(seq)
            seq.append(THREE_TO_ONE[r[2]])
            labels.append(label)
            residues.append({"name": name, "sequence_index_0based": index,
                             "cif_label_seq_id": r[1], "cif_auth_seq_num": r[3], "label_ca_nonobserved": label})
        if labels != record["labels"]:
            raise ValueError(f"CIF labels differ from frozen historical labels: {name}")
        if "".join(seq) != frozen_sequences[name]:
            raise ValueError(f"CIF filtered sequence differs from frozen sequence: {name}")
    qc = {"n_candidate_chains": len(audit), "n_actual_accession_overlap": sum(bool(r["disprot_overlap"]) for r in audit),
          "n_unresolved": sum(r["identity_status"] != "resolved" for r in audit),
          "overlap_chains": [r["name"] for r in audit if r["disprot_overlap"]],
          "never_expanded_evaluation": True, "residue_map_rows": len(residues),
          "scope": "Metadata-only audit of all239 historical selectedchains; current correction evaluation remains99 of historical105"}
    # Deterministic outputs only. Refuse to alter an existing different result.
    for filename, records in [("candidate239_accession_audit.tsv", audit), ("residue_identity_map.tsv", residues)]:
        target = out / filename
        temporary = out / (filename + ".verify")
        write_tsv(temporary, records)
        if target.exists() and target.read_bytes() != temporary.read_bytes():
            temporary.unlink()
            raise ValueError(f"Existing audit differs: {filename}")
        if target.exists():
            temporary.unlink()
        else:
            temporary.rename(target)
    content = json.dumps(qc, indent=2) + "\n"
    target = out / "candidate239_qc.json"
    if target.exists() and target.read_text() != content:
        raise ValueError("Existing candidate QC differs")
    target.write_text(content)
    (out / "candidate239_input_hashes.json").write_text(json.dumps(input_hashes, indent=2) + "\n")
    cluster_path = out / "historical_mmseqs_cluster.tsv"
    if cluster_path.exists():
        groups = defaultdict(list)
        for line in cluster_path.read_text().splitlines():
            representative, member = line.split("\t")
            groups[representative].append(member)
        names = {r["name"] for r in audit}
        excluded_homology = {n for members in groups.values() if any(n.startswith("disprot:") for n in members)
                             for n in members if n in names}
        sequence_path = ROOT / "data/interim/p508_seqs.fa"
        seq = {}; name = None
        for line in sequence_path.read_text().splitlines():
            if line.startswith(">"):
                name = line[1:].strip(); seq[name] = ""
            else:
                seq[name] += line.strip()
        representative_of = {}; dedup = {}
        for name in sorted(names - excluded_homology):
            key = hashlib.sha256(seq[name].encode()).hexdigest()
            representative_of.setdefault(key, name)
            dedup[name] = representative_of[key]
        reconstructed = {name for name, representative in dedup.items() if name == representative}
        reconstructed_rows = [dict(r, reconstructed_homology_excluded=int(r["name"] in excluded_homology),
                                   sequence_representative=dedup.get(r["name"], ""),
                                   reconstructed_historical_retained=int(r["name"] in reconstructed)) for r in audit]
        recon = {"candidate_n": len(names), "homology_excluded_n": len(excluded_homology),
                 "posthomology_n": len(names - excluded_homology), "exactsequence_excluded_n": len(dedup) - len(reconstructed),
                 "reconstructed_historical_n": len(reconstructed), "difference_from_historical_105": sorted(reconstructed ^ historical),
                 "actual_accession_overlaps_already_homology_excluded": sum(bool(r["disprot_overlap"]) and r["name"] in excluded_homology for r in audit),
                 "kind": "Retrospective derivation from retrieved historical cluster table; does not reconstruct original timestamps",
                 "source_sha256": {str(cluster_path.relative_to(ROOT)): hashlib.sha256(cluster_path.read_bytes()).hexdigest(),
                                   str(sequence_path.relative_to(ROOT)): hashlib.sha256(sequence_path.read_bytes()).hexdigest()}}
        for filename, records in [("candidate239_exclusion_reconstruction.tsv", reconstructed_rows)]:
            write_tsv(out / filename, records)
        (out / "exclusion_reconstruction_qc.json").write_text(json.dumps(recon, indent=2) + "\n")
    print(json.dumps(qc, indent=2))


if __name__ == "__main__":
    main()
