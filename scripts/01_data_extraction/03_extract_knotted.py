#!/usr/bin/env python3
# -- Project root setup --
import os as _os
from pathlib import Path as _Path
_os.chdir(_Path(__file__).resolve().parents[2])
"""
Extract knotted protein sequences from official database
Combines extraction and filtering into a single script
Input: data/raw/_all_chains_knotted_N_C_sequence, data/raw/_nr_chains_knotted_N_C
Output: data/knotted_286_full_length.fasta, data/knotted_286_regions.json
"""
import os
import json

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ALL_SEQUENCES_FILE = os.path.join(BASE_DIR, 'data/raw/_all_chains_knotted_N_C_sequence')
NR_FILE = os.path.join(BASE_DIR, 'data/raw/_nr_chains_knotted_N_C')

OUTPUT_FASTA = os.path.join(BASE_DIR, 'data/knotted_286_full_length.fasta')
OUTPUT_JSON = os.path.join(BASE_DIR, 'data/knotted_286_regions.json')


# ============ Utility Functions ============

def write_fasta(records, output_file):
    """Write sequences to FASTA file"""
    with open(output_file, 'w') as f:
        for seq_id, sequence in records:
            f.write(f">{seq_id}\n")
            for i in range(0, len(sequence), 80):
                f.write(sequence[i:i+80] + '\n')


def write_json(data, output_file):
    """Write data to JSON file"""
    with open(output_file, 'w') as f:
        json.dump(data, f, indent=2)


def read_all_sequences():
    """Read all sequences file"""
    sequences = {}

    with open(ALL_SEQUENCES_FILE, 'r') as f:
        for line in f:
            if line.startswith('#') or not line.strip():
                continue

            parts = line.strip().split(';')
            if len(parts) >= 11:
                pdb = parts[0].strip()
                chain = parts[1].strip()
                length = int(parts[3].strip())
                knot_type = parts[6].strip()
                n_cut = parts[7].strip()
                c_cut = parts[8].strip()
                range_str = parts[9].strip()
                sequence = parts[10].strip()

                key = f"{pdb}_{chain}"

                # Keep only first PFAM domain if multiple exist
                if key not in sequences:
                    sequences[key] = {
                        'pdb': pdb,
                        'chain': chain,
                        'length': length,
                        'knot_type': knot_type,
                        'n_cut': n_cut,
                        'c_cut': c_cut,
                        'range': range_str,
                        'sequence': sequence
                    }

    return sequences


def read_nr_list():
    """Read nr dataset list"""
    nr_ids = set()

    with open(NR_FILE, 'r') as f:
        for line in f:
            if line.startswith('#') or not line.strip():
                continue

            parts = line.strip().split(';')
            if len(parts) >= 2:
                pdb = parts[0].strip()
                chain = parts[1].strip()
                key = f"{pdb}_{chain}"
                nr_ids.add(key)

    return nr_ids


def parse_knot_range(range_str):
    """Parse knot region range"""
    # Format: "89-132" or "7-263" etc
    if '-' in range_str:
        parts = range_str.split('-')
        if len(parts) == 2:
            try:
                start = int(parts[0])
                end = int(parts[1])
                return start, end
            except ValueError:
                return None, None
    return None, None


def main():
    print("=" * 70)
    print("Extract Knotted Proteins from Official Database")
    print("=" * 70)

    # Read all sequences
    print("\nStep 1: Reading all sequences...")
    all_sequences = read_all_sequences()
    print(f"  Total sequences: {len(all_sequences)}")

    # Read nr list
    print("\nStep 2: Reading NR list...")
    nr_ids = read_nr_list()
    print(f"  NR sequences: {len(nr_ids)}")

    # Extract nr sequences
    print("\nStep 3: Extracting NR sequences...")

    fasta_records = []
    json_records = []

    success_count = 0
    missing_count = 0
    region_error_count = 0

    for seq_id in sorted(nr_ids):
        if seq_id not in all_sequences:
            print(f"  ❌ {seq_id}: Not found in sequence file")
            missing_count += 1
            continue

        seq_data = all_sequences[seq_id]
        sequence = seq_data['sequence']
        seq_len = len(sequence)

        # Parse knot region
        start, end = parse_knot_range(seq_data['range'])

        if start is None or end is None:
            print(f"  ❌ {seq_id}: Cannot parse region range '{seq_data['range']}'")
            region_error_count += 1
            continue

        # Validate region is within sequence bounds
        if start < 1 or end > seq_len or start > end:
            print(f"  ⚠️  {seq_id}: Region {start}-{end} out of bounds for length {seq_len}")
            # Still keep it but mark it

        # Create FASTA record
        fasta_header = f"knotted|{seq_id}|knot:{start}-{end}|type:{seq_data['knot_type']}"
        fasta_records.append((fasta_header, sequence))

        # Create JSON record
        json_record = {
            'id': seq_id,
            'sequence': sequence,
            'length': seq_len,
            'knot_type': seq_data['knot_type'],
            'regions': [
                {
                    'type': 'knot',
                    'start': start,
                    'end': end
                }
            ]
        }
        json_records.append(json_record)

        success_count += 1

    print(f"\n  Successfully extracted: {success_count}")
    print(f"  Missing sequences: {missing_count}")
    print(f"  Region errors: {region_error_count}")

    # Filter for valid regions only
    print("\nStep 4: Filtering for valid region annotations...")

    valid_fasta = []
    valid_json = []

    stats = {
        'valid': 0,
        'start_gt_end': 0,
        'out_of_bounds': 0
    }

    for i, record in enumerate(json_records):
        seq_id = record['id']
        seq_len = record['length']
        is_valid = True

        for region in record['regions']:
            start = region['start']
            end = region['end']

            if start > end:
                stats['start_gt_end'] += 1
                is_valid = False
                break
            elif start < 1 or end > seq_len:
                stats['out_of_bounds'] += 1
                is_valid = False
                break

        if is_valid:
            valid_json.append(record)
            valid_fasta.append(fasta_records[i])
            stats['valid'] += 1

    print(f"  ✅ Valid region annotations: {stats['valid']} sequences")
    print(f"  ❌ start > end: {stats['start_gt_end']} sequences")
    print(f"  ❌ Out of bounds: {stats['out_of_bounds']} sequences")

    # Write output files
    print("\nStep 5: Writing output files...")
    write_fasta(valid_fasta, OUTPUT_FASTA)
    print(f"  FASTA: {OUTPUT_FASTA}")

    output_json = {
        'dataset': 'knotted_proteins',
        'source': 'official_sequence_file_filtered',
        'filter_criteria': 'region_valid (1 <= start <= end <= length)',
        'total_sequences': len(valid_json),
        'sequences': valid_json
    }
    write_json(output_json, OUTPUT_JSON)
    print(f"  JSON: {OUTPUT_JSON}")

    # Statistics
    print("\n" + "=" * 70)
    print("Dataset Statistics")
    print("=" * 70)

    lengths = [rec['length'] for rec in valid_json]
    print(f"Sequences: {len(valid_json)}")
    print(f"Length range: {min(lengths)} - {max(lengths)} aa")
    print(f"Mean length: {sum(lengths)/len(lengths):.1f} aa")

    # Knot region statistics
    knot_lengths = []
    for rec in valid_json:
        for region in rec['regions']:
            knot_len = region['end'] - region['start'] + 1
            knot_lengths.append(knot_len)

    print(f"\nKnot regions: {len(knot_lengths)}")
    print(f"Knot region length range: {min(knot_lengths)} - {max(knot_lengths)} aa")
    print(f"Knot region mean length: {sum(knot_lengths)/len(knot_lengths):.1f} aa")

    # Knot type distribution
    knot_types = {}
    for rec in valid_json:
        kt = rec['knot_type']
        knot_types[kt] = knot_types.get(kt, 0) + 1

    print(f"\nKnot type distribution:")
    for kt, count in sorted(knot_types.items(), key=lambda x: -x[1])[:10]:
        print(f"  {kt}: {count} sequences")

    print("\nDone!")


if __name__ == "__main__":
    main()
