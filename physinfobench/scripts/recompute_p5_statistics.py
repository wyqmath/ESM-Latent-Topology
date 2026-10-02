#!/usr/bin/env python3
"""Statistical correction from immutable saved scores; never rerun PLM evaluation."""
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import sys
import numpy as np
import sklearn
import yaml
from sklearn.linear_model import LogisticRegression
from cluster_bootstrap import bootstrap_auroc, component_ids, weighted_auroc

ROOT = Path(__file__).resolve().parents[1]


def rows(path):
    with open(path, newline='') as f:
        return list(csv.DictReader(f, delimiter='\t'))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def unique_map(table, key):
    result = {r[key]: r for r in table}
    if len(result) != len(table):
        raise ValueError(f'Duplicate {key}')
    return result


def length_comparator():
    # Matches original frozen runners exactly; no validation labels used in fitting.
    seq_rows = rows(ROOT/'data/curated/knots_sequences.tsv')
    seq = {r['record_id'].lower(): r for r in seq_rows}
    if len(seq) != len(seq_rows):
        raise ValueError('Sequence case-normalization collision')
    knot_rows = rows(ROOT/'data/curated/knots.tsv')
    knot = {r['record_id'].lower(): r for r in knot_rows}
    if len(knot) != len(knot_rows):
        raise ValueError('Knot case-normalization collision')
    units = []
    for r in rows(ROOT/'data/splits/split_manifest.tsv'):
        if r['task_area'] != 'knot' or r['split'] != 'development':
            continue
        rec = r['sample_id'].split(':', 1)[1].lower()
        s = seq.get(rec)
        if s is None or s['presence_target'] not in ('0', '1'):
            continue
        k = knot.get(rec)
        if k is None or k['presence_mask'] != '1' or k['presence_target'] != s['presence_target']:
            raise ValueError(f'Development label inconsistency {rec}')
        units.append((r['sample_id'], int(s['len']), int(s['presence_target'])))
    if len(units) != 750 or sum(u[2] for u in units) != 127:
        raise ValueError('Frozen development universe differs from 750 (127+/623-)')
    clf = LogisticRegression(C=1, class_weight='balanced', solver='liblinear', max_iter=1000, random_state=2026)
    clf.fit(np.log1p(np.array([[u[1]] for u in units], dtype=float)), np.array([u[2] for u in units]))
    return clf, seq, units


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--component-map', type=Path, required=True)
    ap.add_argument('--output', type=Path, default=ROOT/'results/repairs/20261001/statistics')
    ap.add_argument('--posthoc-exposure', type=Path, help='Optional audited eligible subset; descriptive sensitivity only')
    args = ap.parse_args()
    protocol = yaml.safe_load((ROOT/'configs/evaluation_protocol.yaml').read_text())['aggregation_and_uncertainty']
    if (protocol['resampling_B'], protocol['confidence_level'], protocol['seeds']) != (2000, .95, [13,42,2026]):
        raise ValueError('Frozen statistical protocol changed; do not silently use old constants')
    eligible = None
    if args.posthoc_exposure:
        eligible = {r['sample_id'] for r in rows(args.posthoc_exposure)
                    if r['cohort'] in ('confirmation_knot','holdout_knot') and r['posthoc_sensitivity_eligible'] == '1'}
        if not eligible:
            raise ValueError('Posthoc subset is empty')
    if args.output.exists() and list(args.output.iterdir()):
        ap.error('Output exists and is nonempty; refuse overwrite')
    args.output.mkdir(parents=True, exist_ok=True)
    comp_rows = rows(args.component_map)
    comp = unique_map(comp_rows, 'sample_id')
    mapping = {sid: r['component_id'] for sid, r in comp.items()}
    clf_len, seq, dev = length_comparator()
    inputs = [args.component_map, ROOT/'data/splits/split_manifest.tsv', ROOT/'data/curated/knots.tsv',
              ROOT/'data/curated/knots_sequences.tsv', Path(__file__), ROOT/'scripts/cluster_bootstrap.py',
              ROOT/'configs/evaluation_protocol.yaml']
    summary = {'created_at_utc': datetime.now(timezone.utc).isoformat(),
               'kind': ('posthoc descriptive unexposed-component sensitivity; not fresh/preregistered' if eligible is not None else 'statistical correction of saved predictions; no model rerun'),
               'software': {'python': platform.python_version(), 'numpy': np.__version__, 'sklearn': sklearn.__version__},
               'parameters': {'B': 2000, 'confidence': .95, 'seeds': [13, 42, 2026]},
               'independence_unit': 'full binding component; preserves chain-weighted AUROC estimand',
               'limitations': ['Bootstrap conditions on the frozen trained reader and saved evaluation set.',
                              'Component resampling does not remove historical development exposure.',
                              'Min-TM component binding does not guarantee max-TM or family independence.',
                              'Saved PLM scores have six significant digits; report point precision accordingly.',
                              'Length scores reconstructed from frozen development-only sklearn fit; original per-chain comparator scores were not archived.'],
               'length_reconstruction': {'development_n': len(dev), 'development_pos': sum(u[2] for u in dev),
                                         'coef': clf_len.coef_.tolist(), 'intercept': clf_len.intercept_.tolist()}, 'sets': {}}
    if args.posthoc_exposure:
        inputs.append(args.posthoc_exposure)
    table = []
    for split, rel in [('confirmation', 'results/confirmation/CONF-KNOT-PRESENCE-P502-v1'),
                       ('final_holdout', 'results/holdout/HOLD-KNOT-PRESENCE-P505-v1')]:
        score_path = ROOT/rel/'scores.tsv'
        metrics_path = ROOT/rel/'metrics.json'
        inputs.extend([score_path, metrics_path])
        full_rows = rows(score_path)
        sr = full_rows if eligible is None else [r for r in full_rows if r['sample_id'] in eligible]
        ids = [r['sample_id'] for r in sr]
        y = np.array([int(r['label']) for r in sr])
        scores = np.array([float(r['score']) for r in sr])
        groups = component_ids(ids, mapping)
        if any(comp[sid]['current_split'] != split for sid in ids):
            raise ValueError(f'{split}: component split mismatch')
        orig = json.loads(metrics_path.read_text())
        point = weighted_auroc(y, scores)
        if eligible is None and abs(point-orig['auroc']) > 5e-7:
            raise ValueError('Saved-score point estimate differs from frozen metric')
        lengths = []
        for sid in ids:
            rec = sid.split(':',1)[1].lower()
            if rec not in seq:
                raise ValueError(f'Missing length {sid}')
            lengths.append(int(seq[rec]['len']))
        length_scores = clf_len.predict_proba(np.log1p(np.array(lengths, dtype=float)[:, None]))[:,1]
        length_auc = weighted_auroc(y, length_scores)
        if eligible is None and abs(length_auc-orig['length_control_auroc']) > 5e-7:
            raise ValueError('Length reconstruction AUROC does not match frozen metric')
        with open(args.output/f'{split}_length_scores.tsv', 'w', newline='') as f:
            w=csv.writer(f,delimiter='\t'); w.writerow(['sample_id','length','length_score','component_id'])
            w.writerows(zip(ids,lengths,length_scores,groups))
        record = {'n_chains': len(ids), 'n_pos': int(y.sum()), 'n_neg': int((1-y).sum()),
                  'n_components': len(set(groups)), 'original_metrics': orig,
                  'saved_score_auroc': point, 'reconstructed_length_auroc': length_auc,
                  'primary_component': {}, 'secondary_chain_legacy_comparability': {}}
        for unit, group_ids in [('primary_component',groups),('secondary_chain_legacy_comparability',ids)]:
            for seed in [13,42,2026]:
                stats, draws, deltas = bootstrap_auroc(y,scores,group_ids,seed=seed,comparison_scores=length_scores)
                record[unit][str(seed)] = stats
                with open(args.output/f'{split}_{unit}_seed{seed}_draws.tsv','w',newline='') as f:
                    w=csv.writer(f,delimiter='\t'); w.writerow(['draw','plm_auroc','plm_minus_length'])
                    w.writerows((i,float(a),float(d)) for i,(a,d) in enumerate(zip(draws,deltas)))
                table.append([split,unit,seed,stats['n_units'],point,stats['ci95_low'],stats['ci95_high'],
                              stats['degenerate_discarded'],stats['paired_difference']['point'],
                              stats['paired_difference']['ci95_low'],stats['paired_difference']['ci95_high']])
        summary['sets'][split]=record
    summary['input_sha256']={str(p.relative_to(ROOT)) if p.is_relative_to(ROOT) else str(p):sha(p) for p in inputs}
    with open(args.output/'summary.tsv','w',newline='') as f:
        w=csv.writer(f,delimiter='\t'); w.writerow(['split','resampling_unit','seed','n_units','auroc','ci95_low','ci95_high','degenerate','plm_minus_length','delta_ci95_low','delta_ci95_high']); w.writerows(table)
    (args.output/'statistics_correction.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False)+'\n')
    print(json.dumps({'output':str(args.output),'sets':{k:{'n':v['n_chains'],'components':v['n_components'],'auroc':v['saved_score_auroc']} for k,v in summary['sets'].items()}},indent=2))

if __name__ == '__main__':
    main()
