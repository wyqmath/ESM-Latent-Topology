#!/usr/bin/env python3
"""Read-only verification of the posthoc repair integrity lock, never evaluate models."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import yaml


def verify(root, lock):
    document = yaml.safe_load(lock.read_text())
    if document.get('kind') != 'posthoc_correction_integrity' or document.get('gate_approval') is not False:
        raise ValueError('Unexpected lock kind or gate approval')
    seen, errors = set(), []
    for record in document['files']:
        rel = Path(record['path'])
        if rel.is_absolute() or '..' in rel.parts or str(rel) in seen:
            raise ValueError(f'Unsafe or duplicate lock path: {rel}')
        seen.add(str(rel))
        path = root / rel
        if not path.is_file():
            errors.append({'path': str(rel), 'reason': 'missing'})
            continue
        if not path.resolve().is_relative_to(root.resolve()):
            raise ValueError(f'Path outside root: {rel}')
        raw = path.read_bytes()
        if len(raw) != record['bytes'] or hashlib.sha256(raw).hexdigest() != record['sha256']:
            errors.append({'path': str(rel), 'reason': 'hash_or_size_mismatch'})
    return {'status': 'PASS' if not errors else 'FAIL', 'checked_files': len(seen),
            'kind': document['kind'], 'gate_approval': False, 'errors': errors}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--lock', type=Path)
    args = parser.parse_args()
    lock = args.lock or args.root / 'configs/p5_correction_lock_20261001.yaml'
    result = verify(args.root, lock)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result['status'] == 'PASS' else 1


if __name__ == '__main__':
    sys.exit(main())
