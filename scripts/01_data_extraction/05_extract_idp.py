#!/usr/bin/env python3
# -- Project root setup --
import os as _os
from pathlib import Path as _Path
_os.chdir(_Path(__file__).resolve().parents[2])
"""
Download full-length sequences for IDP proteins
Input: data/raw/DisProt_release_2025_12.tsv
Output: data/idp_1000_full_length.fasta, data/idp_1000_regions.json
"""
import os
import time
import requests
import concurrent.futures
import random
import subprocess
import json
from tqdm import tqdm

# ============ Configuration ============
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TSV_FILE = os.path.join(BASE_DIR, 'data/raw/DisProt_release_2025_12.tsv')
OUTPUT_DIR = os.path.join(BASE_DIR, 'data')
OUTPUT_RAW_FASTA = os.path.join(OUTPUT_DIR, 'idp_full_length_raw.fasta')
OUTPUT_NR95_FASTA = os.path.join(OUTPUT_DIR, 'idp_full_length_nr95.fasta')
OUTPUT_1000_FASTA = os.path.join(OUTPUT_DIR, 'idp_1000_full_length.fasta')
OUTPUT_1000_JSON = os.path.join(OUTPUT_DIR, 'idp_1000_regions.json')

NUM_THREADS = 16
TARGET_COUNT = 1000
RANDOM_SEED = 42


# ============ Utility Functions ============

def download_uniprot_sequence(uniprot_id, max_retries=3, timeout=30):
    """Download sequence from UniProt REST API"""
    url = f"https://rest.uniprot.org/uniprotkb/{uniprot_id}.fasta"

    for attempt in range(max_retries):
        try:
            response = requests.get(url, timeout=timeout)
            response.raise_for_status()

            lines = response.text.strip().split('\n')
            if lines and lines[0].startswith('>'):
                sequence = ''.join(lines[1:])
                return sequence, True
            return None, False

        except requests.exceptions.RequestException:
            if attempt < max_retries - 1:
                time.sleep(1.5 ** attempt)
            else:
                return None, False
    return None, False


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


def parse_disprot_tsv(filepath):
    """
    Parse DisProt TSV file
    Returns: dict mapping UniProt ACC to list of disorder regions
    """
    protein_regions = {}

    with open(filepath, 'r') as f:
        header = f.readline()  # Skip header
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) < 3:
                continue

            uniprot_acc = parts[0]
            start = int(parts[1])
            end = int(parts[2])

            if uniprot_acc not in protein_regions:
                protein_regions[uniprot_acc] = []

            protein_regions[uniprot_acc].append((start, end))

    # Merge overlapping regions for each protein
    for uniprot_acc in protein_regions:
        regions = protein_regions[uniprot_acc]
        regions.sort()
        merged = []

        for start, end in regions:
            if merged and start <= merged[-1][1]:
                # Overlapping or adjacent, merge
                merged[-1] = (merged[-1][0], max(merged[-1][1], end))
            else:
                merged.append((start, end))

        protein_regions[uniprot_acc] = merged

    return protein_regions


def download_uniprot_worker(uniprot_id):
    """Worker function for parallel UniProt download"""
    sequence, success = download_uniprot_sequence(uniprot_id)
    return uniprot_id, sequence, success


def run_cdhit(input_fasta, output_fasta, identity=0.95, threads=8):
    """
    Run CD-HIT to remove redundancy
    """
    print(f"\nRunning CD-HIT at {identity*100}% identity...")

    cmd = [
        'cd-hit',
        '-i', input_fasta,
        '-o', output_fasta,
        '-c', str(identity),
        '-n', '5',  # word length
        '-M', '0',  # unlimited memory
        '-T', str(threads),
        '-d', '0'   # full description in output
    ]

    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(f"  CD-HIT completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"  CD-HIT failed: {e}")
        print(f"  stderr: {e.stderr}")
        return False


def main():
    print("=" * 70)
    print("IDP Full-Length Sequence Download")
    print("=" * 70)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Step 1: Parse DisProt TSV
    print("\nStep 1: Parsing DisProt TSV file...")
    protein_regions = parse_disprot_tsv(TSV_FILE)
    print(f"  Found {len(protein_regions)} unique UniProt accessions")

    total_regions = sum(len(regions) for regions in protein_regions.values())
    print(f"  Total disorder regions: {total_regions}")

    # Step 2: Download full-length sequences from UniProt
    print("\nStep 2: Downloading full-length sequences from UniProt...")

    uniprot_ids = list(protein_regions.keys())
    print(f"  UniProt IDs to download: {len(uniprot_ids)}")

    sequences = {}
    success_count = 0
    fail_count = 0

    with concurrent.futures.ThreadPoolExecutor(max_workers=NUM_THREADS) as executor:
        futures = [executor.submit(download_uniprot_worker, uid) for uid in uniprot_ids]

        for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="Downloading"):
            uniprot_id, sequence, success = future.result()

            if success and sequence:
                sequences[uniprot_id] = sequence
                success_count += 1
            else:
                fail_count += 1

    print(f"  Download complete: {success_count} success, {fail_count} failed")

    # Step 3: Write raw FASTA
    print("\nStep 3: Writing raw FASTA file...")
    raw_records = []
    for uniprot_id, sequence in sequences.items():
        regions = protein_regions[uniprot_id]
        region_str = ','.join([f"{s}-{e}" for s, e in regions])
        fasta_id = f"idp|{uniprot_id}|disorder:{region_str}"
        raw_records.append((fasta_id, sequence))

    write_fasta(raw_records, OUTPUT_RAW_FASTA)
    print(f"  Raw FASTA written: {OUTPUT_RAW_FASTA}")
    print(f"  Total sequences: {len(raw_records)}")

    # Step 4: CD-HIT redundancy removal (optional)
    print("\nStep 4: CD-HIT redundancy removal...")

    # Check if CD-HIT is available
    import shutil
    if shutil.which('cd-hit'):
        print("  CD-HIT found, running at 95% identity...")
        cdhit_success = run_cdhit(OUTPUT_RAW_FASTA, OUTPUT_NR95_FASTA, identity=0.95, threads=NUM_THREADS)
        if cdhit_success:
            nr95_fasta = OUTPUT_NR95_FASTA
        else:
            print("  CD-HIT failed, using raw sequences")
            nr95_fasta = OUTPUT_RAW_FASTA
    else:
        print("  CD-HIT not found, skipping redundancy removal")
        print("  Using raw sequences for sampling")
        nr95_fasta = OUTPUT_RAW_FASTA

    # Read non-redundant sequences
    nr_records = []
    with open(nr95_fasta, 'r') as f:
        current_id = None
        current_seq = []

        for line in f:
            line = line.strip()
            if line.startswith('>'):
                if current_id:
                    nr_records.append((current_id, ''.join(current_seq)))
                current_id = line[1:]
                current_seq = []
            else:
                current_seq.append(line)

        if current_id:
            nr_records.append((current_id, ''.join(current_seq)))

    print(f"  Non-redundant sequences: {len(nr_records)}")

    # Step 5: Random sampling
    print(f"\nStep 5: Random sampling {TARGET_COUNT} sequences...")

    random.seed(RANDOM_SEED)

    if len(nr_records) <= TARGET_COUNT:
        print(f"  Warning: Only {len(nr_records)} sequences available, using all")
        sampled_records = nr_records
    else:
        sampled_records = random.sample(nr_records, TARGET_COUNT)

    print(f"  Sampled {len(sampled_records)} sequences")

    # Step 6: Write output files
    print("\nStep 6: Writing output files...")

    # Write FASTA
    write_fasta(sampled_records, OUTPUT_1000_FASTA)
    print(f"  FASTA written: {OUTPUT_1000_FASTA}")

    # Prepare JSON data
    json_data = []
    for fasta_id, sequence in sampled_records:
        # Parse ID: "idp|P03265|disorder:294-334,454-464"
        parts = fasta_id.split('|')
        if len(parts) >= 3:
            uniprot_id = parts[1]
            region_str = parts[2].replace('disorder:', '')

            # Parse regions
            regions = []
            for region in region_str.split(','):
                if '-' in region:
                    start, end = region.split('-')
                    regions.append({
                        "type": "disorder",
                        "start": int(start),
                        "end": int(end)
                    })

            json_data.append({
                "id": uniprot_id,
                "sequence": sequence,
                "length": len(sequence),
                "regions": regions
            })

    output_json = {
        "dataset": "idp",
        "total_sequences": len(json_data),
        "sequences": json_data
    }
    write_json(output_json, OUTPUT_1000_JSON)
    print(f"  JSON written: {OUTPUT_1000_JSON}")

    # Summary statistics
    print("\n" + "=" * 70)
    print("Summary Statistics")
    print("=" * 70)
    print(f"  Total sequences: {len(sampled_records)}")
    if sampled_records:
        lengths = [len(seq) for _, seq in sampled_records]
        print(f"  Length range: {min(lengths)} - {max(lengths)} aa")
        print(f"  Mean length: {sum(lengths) / len(lengths):.1f} aa")

    print("\nDone!")


if __name__ == "__main__":
    main()
