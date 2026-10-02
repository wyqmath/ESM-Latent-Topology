"""Pre-upload review: synthetic fixtures only; no checkpoint/GPU/network use.

Known defects are asserted as reproductions, not as successful acceptance tests.
Run from any directory with this project's .venv_local/bin/python.
"""
import ast
import contextlib
import csv
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch
from sklearn.metrics import average_precision_score

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
findings = {}


def module(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / 'scripts' / (name + '.py'))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def ast_function(path, name, namespace, parent=None):
    tree = ast.parse((ROOT / path).read_text())
    if parent:
        tree = next(x for x in tree.body if isinstance(x, ast.FunctionDef) and x.name == parent)
    fn = next(x for x in ast.walk(tree) if isinstance(x, ast.FunctionDef) and x.name == name)
    exec(compile(ast.Module(body=[fn], type_ignores=[]), str(ROOT / path), 'exec'), namespace)
    return namespace[name]


metrics = module('metrics')
observed = [metrics.average_precision([.5, .5], y) for y in ([1, 0], [0, 1])]
reference = [average_precision_score(y, [.5, .5]) for y in ([1, 0], [0, 1])]
assert observed == [1., .5] and reference == [.5, .5]
findings['ap_ties'] = {'observed': observed, 'sklearn_reference': reference}
findings['rank_direction'] = {'baseline_best': metrics.positive_rank_percentile(list('abcd'), ['a']),
                              'plm_formula_best': 1 / 4}
baseline_rows = list(csv.DictReader(open(ROOT / 'results/baselines/l2_development.tsv'), delimiter='\t'))
stage = json.loads((ROOT / 'results/probes/T-FS-L2-PU-RANK.stageB.metrics.json').read_text())
findings['baseline_rank_conversion'] = {
    'N': stage['universe'],
    'formula': 'mean_rank/N = 1 - old_percentile + 1/N',
    'rows': [{'baseline': r['baseline'], 'old': float(r['positive_rank_percentile']),
              'mean_rank_over_N': 1 - float(r['positive_rank_percentile']) + 1 / stage['universe']}
             for r in baseline_rows if r['k'] == 'percentile']}

extract = module('extract_representations')
loads = []
forwards = []


class FakeTokenizer:
    def __call__(self, seqs, **kwargs):
        width = max(map(len, seqs)) + 2
        mask = torch.zeros((len(seqs), width), dtype=torch.long)
        for i, seq in enumerate(seqs):
            mask[i, :len(seq) + 2] = 1
        return {'input_ids': mask.clone(), 'attention_mask': mask}


class FakeModel:
    def half(self):
        return self

    def cuda(self):
        return self

    def eval(self):
        return self

    def __call__(self, input_ids, attention_mask):
        forwards.append(True)
        b, t = input_ids.shape
        return SimpleNamespace(hidden_states=tuple(torch.ones((b, t, 2)) for _ in range(34)))


def fake_tokenizer(*args, **kwargs):
    loads.append({'kind': 'tokenizer', 'args': args, 'kwargs': kwargs})
    return FakeTokenizer()


def fake_model(*args, **kwargs):
    loads.append({'kind': 'model', 'args': args, 'kwargs': kwargs})
    return FakeModel()


def run_extract(fasta, dest, extra):
    capture = io.StringIO()
    with patch.object(extract.AutoTokenizer, 'from_pretrained', side_effect=fake_tokenizer), \
         patch.object(extract.AutoModel, 'from_pretrained', side_effect=fake_model), \
         patch.dict(os.environ, {'ESM_REVISION_SNAPSHOT': 'audit-requested-revision'}), \
         patch.object(sys, 'argv', ['extract', '--fasta', str(fasta), '--out-dir', str(dest), *extra]), \
         patch.object(torch.cuda, 'is_available', return_value=True), \
         patch.object(torch.Tensor, 'cuda', lambda self: self), \
         contextlib.redirect_stdout(capture), contextlib.redirect_stderr(capture):
        extract.main()
    return list(csv.DictReader(open(dest / 'extract_manifest.tsv'), delimiter='\t'))


with tempfile.TemporaryDirectory(prefix='plm-code-audit-') as temp:
    temp = Path(temp)
    fasta = temp / 'fixture.fa'
    fasta.write_text('>fixture\nACD\n')
    default = temp / 'default'
    rows = run_extract(fasta, default, [])
    with np.load(default / (rows[0]['key'] + '.npz'), allow_pickle=False) as z:
        findings['default_extractor'] = {'saved_arrays': z.files, 'declared_layer_set': rows[0]['layer_set']}
        assert 'mean_layers' not in z.files
    paired = temp / 'paired'
    cpu = run_extract(fasta, paired, ['--mean-layers', '33'])
    n_forward = len(forwards)
    gpu = run_extract(fasta, paired, ['--mean-layers', '33', '--device', 'cuda'])
    assert cpu[0]['key'] == gpu[0]['key'] and gpu[0]['cache'] == 'hit' and len(forwards) == n_forward
    findings['precision_cache_collision'] = {'cpu_key': cpu[0]['key'], 'cuda_key': gpu[0]['key'],
                                             'cuda_cache': gpu[0]['cache'], 'mock_only': True}
    assert all('revision' not in x['kwargs'] for x in loads)
    findings['checkpoint_loading'] = {'requested_revision': 'audit-requested-revision', 'calls': loads}
    cached = paired / (cpu[0]['key'] + '.npz')
    with np.load(cached, allow_pickle=False) as z:
        items = {k: z[k] for k in z.files}
    meta = json.loads(str(items['meta']))
    meta['len_used'] = 999
    items['meta'] = np.array(json.dumps(meta))
    np.savez_compressed(cached, **items)
    bad_hash = hashlib.sha256(cached.read_bytes()).hexdigest()
    rejected = run_extract(fasta, paired, ['--mean-layers', '33'])
    assert [r['cache'] for r in rejected] == ['reject', 'miss']
    assert bad_hash != hashlib.sha256(cached.read_bytes()).hexdigest()
    findings['cache_rejection_overwrite'] = {'manifest_states': [r['cache'] for r in rejected],
                                             'same_key_overwritten': True}

    build = module('p507_build_final_universe')
    sandbox = temp / 'universe'
    interim = sandbox / 'data/interim'
    (interim / 'p507_isolate').mkdir(parents=True)
    curated = sandbox / 'data/curated'
    curated.mkdir(parents=True)
    splits = sandbox / 'data/splits'
    splits.mkdir(parents=True)
    (interim / 'p507_type_frozen_table.tsv').write_text('chain\tstatus\tfrozen_type\nNEW_A\tfrozen\t3_1\n')
    (interim / 'p507_isolate/seq_clusters_rep.tsv').write_text('')
    (curated / 'knots.tsv').write_text('record_id\ttype_task_tier\tc2_primary\n')
    (curated / 'knots_sequences.tsv').write_text('record_id\n')
    (splits / 'split_manifest.tsv').write_text('task_area\tsample_id\tsplit\tdev_fold\n')
    log = io.StringIO()
    with patch.object(build, 'ROOT', str(sandbox)), contextlib.redirect_stdout(log):
        build.main()
    output = json.loads((interim / 'p507_final_universe.json').read_text())
    assert 'p507:NEW_A' in output['nodes']
    findings['missing_structure_files'] = {'generated_nodes': list(output['nodes']), 'log': log.getvalue()}

dual = 'TM-score= 0.82000 (if normalized by length of Structure_1)\nTM-score= 0.40000 (if normalized by length of Structure_2)'
modern = dual.replace('Structure_1', 'Chain_1').replace('Structure_2', 'Chain_2')
response = SimpleNamespace(stdout=dual, returncode=0)
work = ast_function('scripts/p507_isolate_cluster.py', 'work',
                    {'subprocess': SimpleNamespace(run=lambda *a, **kw: response), 'OUT': '/fixture', 'USALIGN': 'fake'},
                    parent='stage5_usalign')
one_direction = work(('A', 'B'))[2]
response.stdout = modern
modern_parsed = work(('A', 'B'))[2]
assert one_direction == .82 and modern_parsed is None
findings['usalign_parser'] = {'parsed_Structure_1': one_direction, 'true_min': .4,
                              'parsed_Chain_format': modern_parsed}

captured = []
ns = {'np': np, 'os': os, 'ROOT': '/fixture', 'EMB': '/fixture', 'QC': {'tasks': {}}, 'SUMMARY': [],
      'load_emb': lambda _: {'porter_9_A': {'mean_layers': np.array([[1., 2.]])}},
      'rd': lambda _: [{'pair_id': 'porter_9', 'target': '1', 'valid_mask': '1'}],
      'write_selection': lambda name, rows: captured.extend(rows)}
ast_function('scripts/run_probes_p303.py', 't_fs_l1', ns)()
assert captured[0]['cosine'] == 1.
findings['missing_FS_endpoint'] = {'missing': 'porter_9_B', 'reported_cosine': captured[0]['cosine']}
masked = subprocess.run(['bash', '-c', 'set -e; (printf "[probe] partial\\n"; exit 7) | grep -E "probe"; printf "CONTINUED\\n"'],
                        capture_output=True, text=True)
assert masked.returncode == 0 and 'CONTINUED' in masked.stdout
findings['pipeline_exit_masked'] = {'returncode': masked.returncode, 'stdout': masked.stdout}

syntax = {'python': [], 'shell': []}
for base in ('scripts', 'tests'):
    for p in sorted((ROOT / base).rglob('*.py')):
        ast.parse(p.read_text(), filename=str(p))
        syntax['python'].append(str(p.relative_to(ROOT)))
for p in sorted((ROOT / 'scripts').rglob('*.sh')):
    subprocess.run(['bash', '-n', str(p)], check=True, capture_output=True)
    syntax['shell'].append(str(p.relative_to(ROOT)))
findings['syntax'] = {'python_count': len(syntax['python']), 'shell_count': len(syntax['shell']), 'files': syntax}
findings['environment_lock_missing_sklearn'] = not any(
    l.lower().startswith('scikit-learn') for l in (ROOT / 'environment.lock').read_text().splitlines())
findings['source_sha256'] = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                            for p in sorted((ROOT / 'scripts').rglob('*')) if p.is_file() and p.suffix in ('.py', '.sh')}
findings['head'] = subprocess.check_output(['git', '-C', str(ROOT), 'rev-parse', 'HEAD'], text=True).strip()
(OUT / 'audit_evidence.json').write_text(json.dumps(findings, indent=2, ensure_ascii=False) + '\n')
print(f'Reproduced {len([k for k in findings if k not in ("syntax", "source_sha256", "head")])} evidence categories; synthetic fixtures only.')
print(f'Syntax passed: {len(syntax["python"])} Python, {len(syntax["shell"])} shell files.')
print(OUT / 'audit_evidence.json')
