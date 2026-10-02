#!/usr/bin/env python3
"""Validate a prospective split against audited historical-use component closure.

No split is generated or overwritten. New samples require a new complete graph and
use audit first. A passing guard does not establish task label adequacy.
"""
import argparse
import csv
import gzip
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
AUDIT = ROOT / 'results/repairs/20261001/exposure'
SPLITS = {'development', 'confirmation', 'final_holdout'}


def read_table(path):
    opener = gzip.open if str(path).endswith('.gz') else open
    with opener(path, 'rt', newline='') as f:
        return list(csv.DictReader(f, delimiter='\t'))


def unique_index(rows, key):
    result = {r[key]: r for r in rows}
    if len(result) != len(rows):
        raise ValueError(f'Duplicate {key}')
    return result


def validate(candidate, components, policy, background=None, audited_background=None):
    cmap = unique_index(components, 'sample_id')
    pmap = unique_index(policy, 'component_id')
    proposal = unique_index(candidate, 'sample_id')
    if set(proposal) != set(cmap):
        raise ValueError('Candidate sample universe differs from audited graph; rebuild full graph/use audit')
    violations = []
    assignments = defaultdict(set)

    def check(cid, split, identity):
        if cid not in pmap:
            raise ValueError(f'Unknown audited component {cid}')
        if split not in SPLITS:
            raise ValueError(f'Unknown split {split}')
        assignments[cid].add(split)
        if pmap[cid]['force_future_development'] not in ('0', '1'):
            raise ValueError(f'Invalid historical-use policy for {cid}')
        if pmap[cid]['force_future_development'] == '1' and split != 'development':
            violations.append({'id': identity, 'component_id': cid, 'reason': 'historical_use_or_read_migrated'})

    for sid, r in proposal.items():
        if r.get('task_area') != cmap[sid]['task_area']:
            raise ValueError(f'Task identity differs from audited graph for {sid}')
        if r.get('component_id', cmap[sid]['component_id']) != cmap[sid]['component_id']:
            raise ValueError(f'Incorrect native component for {sid}')
        check(cmap[sid]['component_id'], r['split'], sid)
    if background is not None:
        if audited_background is None:
            raise ValueError('Audited background accession-to-component map required')
        bmap = unique_index(audited_background, 'uniprot_accession')
        bproposal = unique_index(background, 'uniprot_accession')
        if set(bproposal) != set(bmap):
            raise ValueError('Candidate background differs from audited universe')
        for acc, row in bproposal.items():
            cid = bmap[acc]['component_id']
            if row.get('component_id', cid) != cid:
                raise ValueError(f'Incorrect background component for {acc}')
            check(cid, row['split'], acc)
        if set(assignments) != set(pmap):
            raise ValueError('Candidate maps do not cover every audited global component')
    for cid, splits in assignments.items():
        if len(splits) > 1:
            violations.append({'component_id': cid, 'reason': 'component_split', 'splits': sorted(splits)})
    return {'passed': not violations, 'n_samples': len(proposal), 'violations': violations,
            'scope': 'historical-use and binding-component guard; task eligibility checked separately',
            'background_checked': background is not None,
            'ready_for_global_split': not violations and background is not None}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--candidate', type=Path, required=True)
    ap.add_argument('--component-map', type=Path, default=AUDIT/'component_map.tsv')
    ap.add_argument('--policy', type=Path, default=AUDIT/'component_future_policy.tsv')
    ap.add_argument('--background-candidate', type=Path)
    ap.add_argument('--background-map', type=Path, default=AUDIT/'background_component_map.tsv.gz')
    ap.add_argument('--output', type=Path)
    args = ap.parse_args()
    if args.output and args.output.exists():
        ap.error('Refuse to overwrite QC')
    qc = validate(read_table(args.candidate), read_table(args.component_map), read_table(args.policy),
                  read_table(args.background_candidate) if args.background_candidate else None,
                  read_table(args.background_map) if args.background_candidate else None)
    qc['created_at_utc'] = datetime.now(timezone.utc).isoformat()
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(qc, indent=2)+'\n')
    print(json.dumps({'passed': qc['passed'], 'ready_for_global_split': qc['ready_for_global_split'],
                      'n_samples': qc['n_samples'], 'n_violations': len(qc['violations'])}))
    raise SystemExit(0 if qc['ready_for_global_split'] else 1)


if __name__ == '__main__':
    main()
