#!/usr/bin/env python3
"""
Step 1: Prepare single-chain PDB files for all categories.
- Anchor (856): Extract chain from local data/pdb_files/
- Knotted (286): Download from RCSB + extract chain
- Fold-switching (84 pairs = up to 168 structures): Download from RCSB + extract chain
Output: saprot_control/data/single_chain_pdbs/, saprot_control/data/manifest.csv
"""
import os, sys, csv, json, time, urllib.request
from pathlib import Path

# Project root
ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT)

OUT_DIR = Path('saprot_control/data/single_chain_pdbs')
OUT_DIR.mkdir(parents=True, exist_ok=True)

MANIFEST = []  # (seq_id, pdb_file, chain, label)


def download_pdb(pdb4, out_path, max_retries=3, timeout=30):
    """Download PDB or CIF from RCSB. Returns format ('pdb' or 'cif')."""
    pdb4 = pdb4.lower()
    urls = [
        (f"https://files.rcsb.org/download/{pdb4}.pdb", 'pdb'),
        (f"https://files.rcsb.org/download/{pdb4}.cif", 'cif'),
    ]
    for url, fmt in urls:
        target = out_path.with_suffix(f'.{fmt}')
        if target.exists() and target.stat().st_size > 100:
            return fmt, target
        for attempt in range(max_retries):
            try:
                req = urllib.request.urlopen(url, timeout=timeout)
                with open(target, 'wb') as f:
                    f.write(req.read())
                if target.stat().st_size > 100:
                    return fmt, target
            except Exception:
                time.sleep(1 * (attempt + 1))
    return None, None


def extract_chain_pdb(pdb_path, chain_id, out_path):
    """Extract a single chain from a PDB file (simple ATOM line parser)."""
    lines = []
    with open(pdb_path) as f:
        for line in f:
            if line.startswith(('ATOM', 'HETATM')):
                if len(line) > 21 and line[21] == chain_id:
                    lines.append(line)
            elif line.startswith('END'):
                lines.append(line)
    if not lines:
        return False
    with open(out_path, 'w') as f:
        f.writelines(lines)
        if not lines[-1].startswith('END'):
            f.write('END\n')
    return True


def extract_chain_cif(cif_path, chain_id, out_path):
    """Extract chain from CIF by converting relevant ATOM lines to PDB format."""
    lines = []
    with open(cif_path) as f:
        in_atom = False
        headers = []
        for line in f:
            if line.startswith('_atom_site.'):
                in_atom = True
                headers.append(line.strip().split('.')[-1])
                continue
            if in_atom and line.startswith('#'):
                in_atom = False
                continue
            if in_atom and not line.startswith('_') and not line.startswith('loop_'):
                fields = line.split()
                if len(fields) < len(headers):
                    continue
                col_map = {h: fields[i] for i, h in enumerate(headers) if i < len(fields)}
                auth_chain = col_map.get('auth_asym_id', col_map.get('label_asym_id', ''))
                if auth_chain == chain_id and col_map.get('group_PDB', '') == 'ATOM':
                    # Build PDB ATOM line
                    serial = col_map.get('id', '1')
                    name = col_map.get('auth_atom_id', col_map.get('label_atom_id', 'CA'))
                    resname = col_map.get('auth_comp_id', col_map.get('label_comp_id', 'ALA'))
                    resseq = col_map.get('auth_seq_id', col_map.get('label_seq_id', '1'))
                    x = col_map.get('Cartn_x', '0.0')
                    y = col_map.get('Cartn_y', '0.0')
                    z = col_map.get('Cartn_z', '0.0')
                    occ = col_map.get('occupancy', '1.00')
                    bfac = col_map.get('B_iso_or_equiv', '0.00')
                    element = col_map.get('type_symbol', name[0])

                    name_padded = f" {name:<3s}" if len(name) < 4 else name[:4]
                    pdb_line = (
                        f"ATOM  {int(serial):>5d} {name_padded} {resname:>3s} {chain_id}"
                        f"{int(resseq):>4d}    "
                        f"{float(x):>8.3f}{float(y):>8.3f}{float(z):>8.3f}"
                        f"{float(occ):>6.2f}{float(bfac):>6.2f}"
                        f"          {element:>2s}  \n"
                    )
                    lines.append(pdb_line)

    if not lines:
        return False
    with open(out_path, 'w') as f:
        f.writelines(lines)
        f.write('END\n')
    return True


def process_structure(pdb4, chain, seq_id, label, pdb_cache_dir):
    """Download (if needed) and extract chain. Returns success bool."""
    out_path = OUT_DIR / f"{seq_id}.pdb"
    if out_path.exists() and out_path.stat().st_size > 100:
        return True

    # Check local pdb_files first
    local_pdb = ROOT / 'data' / 'pdb_files' / f"{pdb4}.pdb"
    if local_pdb.exists():
        ok = extract_chain_pdb(local_pdb, chain, out_path)
        if ok:
            return True

    # Download
    cache_path = pdb_cache_dir / pdb4
    fmt, dl_path = download_pdb(pdb4, cache_path)
    if fmt is None:
        return False

    if fmt == 'pdb':
        return extract_chain_pdb(dl_path, chain, out_path)
    else:
        return extract_chain_cif(dl_path, chain, out_path)


def main():
    print("=" * 70)
    print("Step 1: Prepare single-chain PDBs")
    print("=" * 70)

    pdb_cache = Path('saprot_control/data/pdb_cache')
    pdb_cache.mkdir(parents=True, exist_ok=True)

    # ── Anchor (856) ──
    print("\n[Anchor] Extracting chains from local PDB files...")
    meta = list(csv.DictReader(open('data/anchor_metadata.csv')))
    anchor_count = 0
    anchor_fail = 0
    for row in meta:
        source_id = row['source_id']  # e.g., '1UCSA'
        pdb4 = source_id[:4].lower()
        chain = source_id[4:]
        seq_id = f"anchor_{source_id}"

        ok = process_structure(pdb4, chain, seq_id, 'anchor', pdb_cache)
        if ok:
            MANIFEST.append((seq_id, f"single_chain_pdbs/{seq_id}.pdb", chain, 'anchor'))
            anchor_count += 1
        else:
            anchor_fail += 1
            if anchor_fail <= 5:
                print(f"  WARN: Failed {source_id}")

    print(f"  Anchor: {anchor_count} OK, {anchor_fail} failed")

    # ── Knotted (286) ──
    print("\n[Knotted] Downloading from RCSB...")
    knotted_count = 0
    knotted_fail = 0
    with open('data/knotted_286_full_length.fasta') as f:
        for line in f:
            if line.startswith('>'):
                # >knotted|1aja_A|knot:46-369|type:31
                parts = line[1:].strip().split('|')
                id_part = parts[1]  # e.g., '1aja_A'
                pdb4, chain = id_part.split('_')
                pdb4 = pdb4.lower()
                seq_id = f"knotted_{pdb4}_{chain}"

                ok = process_structure(pdb4, chain, seq_id, 'knotted', pdb_cache)
                if ok:
                    MANIFEST.append((seq_id, f"single_chain_pdbs/{seq_id}.pdb", chain, 'knotted'))
                    knotted_count += 1
                else:
                    knotted_fail += 1
                    if knotted_fail <= 5:
                        print(f"  WARN: Failed {id_part}")

    print(f"  Knotted: {knotted_count} OK, {knotted_fail} failed")

    # ── Fold-switching (67 pairs = 134 sequences) ──
    print("\n[Fold-switching] Downloading from RCSB...")
    with open('data/fold_switching_pairs.json') as f:
        pairs_data = json.load(f)

    fs_count = 0
    fs_fail = 0
    seen_fs = set()  # avoid duplicates when partner_also_in_dataset
    for pair in pairs_data['pairs']:
        for side, prefix in [('our', 'conf1'), ('partner', 'conf2')]:
            pdb4 = pair[f'{side}_pdb4']
            chain = pair[f'{side}_chain']
            raw_id = pair[f'{side}_id']

            pdb4_norm = pdb4.lower()
            seq_id = f"fs_{pdb4_norm}_{chain}"

            if seq_id in seen_fs:
                continue
            seen_fs.add(seq_id)

            ok = process_structure(pdb4_norm, chain, seq_id, 'fold_switching', pdb_cache)
            if ok:
                conf_label = f'fold_switching_{prefix}'
                MANIFEST.append((seq_id, f"single_chain_pdbs/{seq_id}.pdb", chain, conf_label))
                fs_count += 1
            else:
                fs_fail += 1
                if fs_fail <= 10:
                    print(f"  WARN: Failed {raw_id} ({pdb4_norm} chain {chain})")

    print(f"  Fold-switching: {fs_count} OK, {fs_fail} failed")

    # ── Write manifest ──
    manifest_path = Path('saprot_control/data/manifest.csv')
    with open(manifest_path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['seq_id', 'pdb_file', 'chain', 'label'])
        for row in MANIFEST:
            w.writerow(row)

    print(f"\n  Manifest: {manifest_path} ({len(MANIFEST)} entries)")
    for label in ['anchor', 'knotted', 'fold_switching_conf1', 'fold_switching_conf2']:
        n = sum(1 for r in MANIFEST if r[3] == label)
        if n > 0:
            print(f"    {label}: {n}")

    print("\nDone!")


if __name__ == "__main__":
    main()
