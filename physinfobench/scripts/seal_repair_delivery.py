#!/usr/bin/env python3
"""Seal posthoc correction provenance or package read-only logs and results."""
import argparse
import csv
from datetime import datetime
import hashlib
import json
from pathlib import Path
import subprocess
import zipfile
import yaml
from verify_correction_lock import verify

ROOT = Path(__file__).resolve().parents[1]
LOCK = ROOT / 'configs/p5_correction_lock_20261001.yaml'
SCIENCE_REPORTS = ['reports/confirmation.md', 'reports/generalization.md',
                   'reports/claim_evidence_matrix.md', 'reports/aggregation_diagnosis.md',
                   'reports/tasks/P5.07.md', 'reports/tasks/P5.08.md']


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def selected_paths():
    chosen = set()
    for directory in ['scripts', 'configs', 'tests', 'reports', 'logs', 'results',
                      'data/curated', 'data/splits', 'data/manifests']:
        for path in (ROOT / directory).rglob('*'):
            if path.is_file() and not path.is_symlink() and '__pycache__' not in path.parts:
                if path.suffix in {'.npz', '.npy', '.pt', '.pth', '.ckpt', '.safetensors', '.h5', '.pkl', '.bin'} and path.name != 'frozen_reader_coefficients.npz':
                    continue
                chosen.add(path)
    for name in ['README.md', 'TODO.md', '.gitattributes', '.gitignore']:
        chosen.add(ROOT / name)
    # Include the existing lightweight source manifests used by the exposure audit.
    with (ROOT / 'results/repairs/20261001/exposure/input_checksums.tsv').open() as f:
        for row in csv.DictReader(f, delimiter='\t'):
            relative = Path(row['path'])
            if not relative.is_absolute():
                path = ROOT / relative
                if not path.is_file():
                    raise ValueError(f'Missing audited input {path}')
                chosen.add(path)
    return sorted(chosen)


def write_lock():
    if LOCK.exists():
        raise FileExistsError(f'Refuse to replace existing lock: {LOCK}')
    files = []
    for path in selected_paths():
        rel = path.relative_to(ROOT).as_posix()
        # Final substantive repair reviews are locked; the seal's own report is not self-referential.
        excluded_reports = {'repair_summary.md', 'delivery_seal_independent_review.md',
                            'p508_lock_independent_review.json'}
        repair_report = rel.startswith('reports/repairs/20261001/') and path.name not in excluded_reports
        if not (rel.startswith(('scripts/', 'configs/', 'tests/', 'data/', 'results/'))
                or rel in SCIENCE_REPORTS or repair_report or rel.startswith('logs/reviews/R5.')):
            continue
        if path == LOCK or rel.endswith(('correction_lock_verification.json', 'seal_independent_review.json')):
            continue
        raw = path.read_bytes()
        files.append({'path': rel, 'bytes': len(raw), 'sha256': sha(raw)})
    document = {
        'kind': 'posthoc_correction_integrity', 'created_at': datetime.now().strftime('%Y-%m-%d %H:%M'),
        'timezone': 'Asia/Shanghai', 'baseline_commit': '18c89ea11491004d01ebbe227dc16c28dc87a67f',
        'gate_approval': False, 'preregistration': False, 'new_independent_evaluation': False,
        'meaning': 'Byte integrity for completed correction inputs, code and outputs; no scientific gate approval.',
        'current_claim_levels': {'knot_presence': 'D: historical development exposure',
                                 'fold_switch_ranking': 'D: independent strict positives unavailable',
                                 'external_ca_nonobservation': 'X: posthoc proxy-label correction'},
        'external_dependencies': [
            {'manifest': 'results/repairs/20261001/exposure/input_checksums.tsv',
             'scope': 'Original project endpoint table remains outside repository; original path and SHA recorded.'},
            {'manifest': 'results/repairs/20261001/p508/h33_exact_metrics/replay_provenance.json',
             'scope': 'Cluster embedding caches and raw CIF inputs retained remotely; manifest/hash metadata included, cache bytes excluded.'}],
        'files': files}
    LOCK.write_text(yaml.safe_dump(document, allow_unicode=True, sort_keys=False))
    print(json.dumps({'lock': str(LOCK), 'files': len(files)}, indent=2))


def package(destination):
    if destination.exists():
        raise FileExistsError(f'Refuse to replace delivery: {destination}')
    check = verify(ROOT, LOCK)
    if check['status'] != 'PASS':
        raise ValueError(check)
    chosen = selected_paths()
    git_head = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    meta = {'created_at': datetime.now().strftime('%Y-%m-%d %H:%M'), 'timezone': 'Asia/Shanghai',
            'git_head': git_head, 'correction_lock': 'configs/p5_correction_lock_20261001.yaml',
            'correction_lock_verification': check, 'scope': 'Repair and historical evidence; no raw structures or embedding caches.'}
    extra = {'DELIVERY_METADATA.json': (json.dumps(meta, ensure_ascii=False, indent=2)+'\n').encode(),
             'DELIVERY_README.md': (
                 '# 本轮修复日志与结果包\n\n'
                 '先读 reports/repairs/20261001/repair_summary.md，再读 reports/claim_evidence_matrix.md。'
                 'TODO.md 已登记修复进度；logs/reviews/R5.*.md 保存独立审核。原版结果和纠错结果目录分开。\n\n'
                 '包内含当前源码、配置、标签/划分、保存预测、逐次bootstrap、审核日志、旧基线、运行输入元数据。'
                 '原始CIF、论文、PLM权重和embedding缓存沿manifest在原本地/集群路径维护。'
                 '统计复算使用包内保存预测；P5.08完整读取器重放需要集群缓存。\n\n'
                 '校验全包文件：shasum -a 256 -c MANIFEST.sha256。安装现有Python环境所需numpy、scikit-learn、PyYAML后，'
                 '运行 python scripts/verify_correction_lock.py 核对纠错锁。复跑命令在 repair_summary.md；输出必须采用新目录。\n\n'
                 'G5等待确认，Phase6尚未启动；历史已曝光数据没有恢复未见测试资格。\n').encode()}
    records = []
    destination.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(destination, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=6) as bundle:
        for path in chosen:
            rel = path.relative_to(ROOT).as_posix()
            raw = path.read_bytes()
            bundle.writestr(rel, raw)
            records.append((rel, sha(raw)))
        for rel, raw in extra.items():
            bundle.writestr(rel, raw)
            records.append((rel, sha(raw)))
        manifest = ''.join(f'{digest}  {rel}\n' for rel, digest in sorted(records))
        bundle.writestr('MANIFEST.sha256', manifest)
    with zipfile.ZipFile(destination) as bundle:
        if bundle.testzip() is not None:
            raise ValueError('ZIP CRC check failed')
        members = bundle.namelist()
        if len(members) != len(set(members)) or set(members) != {r[0] for r in records} | {'MANIFEST.sha256'}:
            raise ValueError('ZIP duplicate member or manifest coverage mismatch')
        for rel, digest in records:
            if sha(bundle.read(rel)) != digest:
                raise ValueError(f'ZIP hash mismatch {rel}')
    checksum = sha(destination.read_bytes())
    destination.with_suffix('.zip.sha256').write_text(f'{checksum}  {destination.name}\n')
    qc = {'status': 'PASS', 'zip_path': str(destination), 'bytes': destination.stat().st_size,
          'sha256': checksum, 'verified_payload_files': len(records), 'zip_members': len(records)+1,
          'CRC_all_pass': True, 'manifest_SHA_all_pass': True, 'git_head': git_head,
          'correction_lock_check': check}
    destination.with_suffix('.zip.qc.json').write_text(json.dumps(qc, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps(qc, ensure_ascii=False, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--create-lock', action='store_true')
    mode.add_argument('--package', type=Path)
    args = parser.parse_args()
    write_lock() if args.create_lock else package(args.package)


if __name__ == '__main__':
    main()
