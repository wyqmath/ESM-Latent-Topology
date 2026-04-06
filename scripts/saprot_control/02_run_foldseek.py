#!/usr/bin/env python3
"""
Step 2: Run Foldseek to generate 3Di structural alphabet sequences.
Input: saprot_control/data/single_chain_pdbs/
Output: saprot_control/data/3di_sequences.fasta
"""
import os, subprocess, csv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT)

FOLDSEEK = Path('saprot_control/foldseek_bin')
PDB_DIR = Path('saprot_control/data/single_chain_pdbs')
DB_DIR = Path('saprot_control/data/foldseek_db')
TMP_DIR = Path('saprot_control/data/foldseek_tmp')
OUTPUT_FASTA = Path('saprot_control/data/3di_sequences.fasta')


def main():
    print("=" * 70)
    print("Step 2: Run Foldseek → 3Di sequences")
    print("=" * 70)

    DB_DIR.mkdir(parents=True, exist_ok=True)
    TMP_DIR.mkdir(parents=True, exist_ok=True)

    if not FOLDSEEK.exists():
        print("ERROR: Foldseek not found. Run 00_setup.sh first.")
        return

    pdb_files = list(PDB_DIR.glob('*.pdb'))
    print(f"  Input PDBs: {len(pdb_files)}")

    # Load manifest to know expected count
    manifest = list(csv.DictReader(open('saprot_control/data/manifest.csv')))
    print(f"  Manifest entries: {len(manifest)}")

    # Create Foldseek database
    db_path = DB_DIR / 'structures'
    print("\n  Creating Foldseek database...")
    subprocess.run([
        str(FOLDSEEK), 'createdb', str(PDB_DIR), str(db_path),
        '--threads', '4'
    ], check=True, capture_output=True)

    # Convert to 3Di FASTA
    # Need to symlink header DB for the _ss database
    print("  Linking header DB for 3Di...")
    ss_h = DB_DIR / 'structures_ss_h'
    ss_h_idx = DB_DIR / 'structures_ss_h.index'
    ss_h_dbt = DB_DIR / 'structures_ss_h.dbtype'
    for link, target_name in [(ss_h, 'structures_h'), (ss_h_idx, 'structures_h.index'), (ss_h_dbt, 'structures_h.dbtype')]:
        link.unlink(missing_ok=True)
        link.symlink_to(target_name)

    print("  Converting to 3Di FASTA...")
    subprocess.run([
        str(FOLDSEEK), 'convert2fasta', str(db_path) + '_ss', str(OUTPUT_FASTA)
    ], check=True, capture_output=True)

    # Also extract AA sequences from Foldseek's view
    aa_fasta = Path('saprot_control/data/foldseek_aa.fasta')
    subprocess.run([
        str(FOLDSEEK), 'convert2fasta', str(db_path), str(aa_fasta)
    ], check=True, capture_output=True)

    # Count output sequences
    n_3di = sum(1 for line in open(OUTPUT_FASTA) if line.startswith('>'))
    n_aa = sum(1 for line in open(aa_fasta) if line.startswith('>'))
    print(f"\n  3Di sequences: {n_3di}")
    print(f"  AA sequences:  {n_aa}")

    if n_3di != len(manifest):
        print(f"  WARNING: 3Di count ({n_3di}) != manifest ({len(manifest)})")
        # Find missing
        fasta_ids = set()
        with open(OUTPUT_FASTA) as f:
            for line in f:
                if line.startswith('>'):
                    fasta_ids.add(line[1:].strip().split()[0].replace('.pdb', ''))
        manifest_ids = set(r['seq_id'] for r in manifest)
        missing = manifest_ids - fasta_ids
        if missing:
            print(f"  Missing from 3Di: {len(missing)}")
            for m in sorted(list(missing))[:10]:
                print(f"    {m}")
    else:
        print("  Count matches manifest!")

    print("\nDone!")


if __name__ == "__main__":
    main()
