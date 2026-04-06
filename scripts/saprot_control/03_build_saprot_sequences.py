#!/usr/bin/env python3
"""
Step 3: Build SaProt interleaved sequences (AA + 3Di).
SaProt input format: uppercase AA + lowercase 3Di, e.g. "MdEvVeKi..."
Input: saprot_control/data/foldseek_aa.fasta, saprot_control/data/3di_sequences.fasta
Output: saprot_control/data/saprot_sequences.json
"""
import os, json, csv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT)


def read_fasta(path):
    """Read FASTA, return dict {id: sequence}."""
    seqs = {}
    current_id = None
    current_seq = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line.startswith('>'):
                if current_id:
                    seqs[current_id] = ''.join(current_seq)
                # Foldseek header: >filename.pdb or >filename
                current_id = line[1:].split()[0].replace('.pdb', '')
                current_seq = []
            elif current_id:
                current_seq.append(line)
        if current_id:
            seqs[current_id] = ''.join(current_seq)
    return seqs


def interleave(aa_seq, tdi_seq):
    """Build SaProt interleaved string: AaTbCc... (uppercase AA + lowercase 3Di)."""
    min_len = min(len(aa_seq), len(tdi_seq))
    parts = []
    for i in range(min_len):
        parts.append(aa_seq[i].upper())
        parts.append(tdi_seq[i].lower())
    return ''.join(parts)


def main():
    print("=" * 70)
    print("Step 3: Build SaProt interleaved sequences")
    print("=" * 70)

    aa_seqs = read_fasta('saprot_control/data/foldseek_aa.fasta')
    tdi_seqs = read_fasta('saprot_control/data/3di_sequences.fasta')
    manifest = list(csv.DictReader(open('saprot_control/data/manifest.csv')))

    print(f"  AA sequences: {len(aa_seqs)}")
    print(f"  3Di sequences: {len(tdi_seqs)}")
    print(f"  Manifest entries: {len(manifest)}")

    results = []
    mismatches = 0
    missing = 0

    for row in manifest:
        seq_id = row['seq_id']
        label = row['label']

        # Try exact match first, then MODEL_1 fallback (NMR structures)
        aa_key = seq_id
        tdi_key = seq_id
        if seq_id not in aa_seqs:
            model1_key = f"{seq_id}_MODEL_1_{row['chain']}"
            if model1_key in aa_seqs:
                aa_key = model1_key
            else:
                missing += 1
                if missing <= 5:
                    print(f"  WARN: {seq_id} not in AA FASTA")
                continue

        if aa_key not in tdi_seqs:
            # Also try MODEL_1 for 3Di
            model1_key = f"{seq_id}_MODEL_1_{row['chain']}"
            if model1_key in tdi_seqs:
                tdi_key = model1_key
            else:
                missing += 1
                if missing <= 5:
                    print(f"  WARN: {seq_id} not in 3Di FASTA")
                continue
        else:
            tdi_key = aa_key

        aa = aa_seqs[aa_key]
        tdi = tdi_seqs[tdi_key]

        if len(aa) != len(tdi):
            mismatches += 1
            if mismatches <= 5:
                print(f"  WARN: Length mismatch {seq_id}: AA={len(aa)}, 3Di={len(tdi)} (truncating)")

        saprot_seq = interleave(aa, tdi)

        # Collapse label: fold_switching_conf1/conf2 -> fold_switching
        analysis_label = 'fold_switching' if label.startswith('fold_switching') else label

        results.append({
            'seq_id': seq_id,
            'label': label,
            'analysis_label': analysis_label,
            'saprot_sequence': saprot_seq,
            'aa_length': min(len(aa), len(tdi)),
            'saprot_length': len(saprot_seq),
        })

    # Save
    output_path = Path('saprot_control/data/saprot_sequences.json')
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n  Total SaProt sequences: {len(results)}")
    print(f"  Missing: {missing}")
    print(f"  Length mismatches (truncated): {mismatches}")

    # Stats by label
    from collections import Counter
    label_counts = Counter(r['analysis_label'] for r in results)
    for label, count in sorted(label_counts.items()):
        print(f"    {label}: {count}")

    lengths = [r['aa_length'] for r in results]
    print(f"  Length range: {min(lengths)}-{max(lengths)} aa")
    print(f"  Mean length: {sum(lengths)/len(lengths):.0f} aa")

    print(f"\n  Output: {output_path}")
    print("Done!")


if __name__ == "__main__":
    main()
