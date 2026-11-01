"""只读核查正常适配的来源、HIL 版本与缓存；仅向新报告文件写结果。

python scripts/research/audit_adaptation_prerequisites.py --self-test
python scripts/research/audit_adaptation_prerequisites.py --output <new-report.json>
python scripts/research/audit_adaptation_prerequisites.py --draft-from <report.json> --output <draft.json>

缓存数值检查是固定样本抽查，不证明全部张量或严格因果预处理正确。
成员草案不是正式协议，不修改原清单，也不用于训练。
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA = Path('D:/UAV-Data')
MANIFEST = DATA / 'splits/rflymad_case_split_v2/rflymad_case_level_split_manifest.csv'
AUDIT = ROOT / 'docs/research/2026-10-08/data_feasibility/full_case_audit.csv'
SPLITS = {'train': 'train.pt', 'val': 'val.pt', 'test': 'test.pt', 'external_test': 'real_holdout_test.pt'}

def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()

def read_csv(path):
    with Path(path).open(encoding='utf-8-sig', newline='') as stream:
        return list(csv.DictReader(stream))

def duplicate_groups(records, field):
    groups = defaultdict(set)
    for row in records:
        for value in row.get(field, []):
            if value:
                groups[value].add(row['case_uid'])
    return [{'key': key, 'case_uids': sorted(uids)} for key, uids in sorted(groups.items()) if len(uids) > 1]

def probe_positions(size):
    if size <= 0:
        raise ValueError('empty case cannot be probed')
    return sorted({0, size // 2, size - 1})

def audit_sources(rows):
    from openpyxl import load_workbook
    records, errors = [], []
    for row in rows:
        if row['domain'].upper() != 'REAL':
            continue
        source = Path(row['source_path'])
        entry = {key: row[key] for key in ['case_uid', 'flight_status', 'source_disposition', 'level2_label']}
        entry['source_exists'] = source.is_dir()
        entry['raw_log_keys'], entry['capture_dates'], entry['testinfo_case_id'] = [], [], []
        if not source.is_dir():
            errors.append({'case_uid': row['case_uid'], 'error': 'source directory missing'})
        else:
            # Filenames identify potential shared source logs, not proven device IDs.
            raw = sorted(p for p in source.glob('**/*') if p.is_file() and p.suffix.lower() in {'.ulg', '.bag', '.tlog'})
            entry['raw_log_keys'] = sorted({p.suffix.lower() + ':' + p.name.lower() for p in raw})
            entry['capture_dates'] = sorted({p.name[:10] for p in raw if p.suffix.lower() == '.tlog'})
            infos = sorted(source.glob('TestInfo*.xlsx'))
            if len(infos) != 1:
                errors.append({'case_uid': row['case_uid'], 'error': 'expected one TestInfo workbook', 'count': len(infos)})
            else:
                book = load_workbook(infos[0], read_only=True, data_only=False)
                try:
                    pairs = list(book.worksheets[0].iter_rows(values_only=True))
                    ids = [str(pair[1]).strip() for pair in pairs if len(pair) >= 2 and str(pair[0]).strip() == 'Real Flight case ID' and pair[1] is not None]
                    entry['testinfo_case_id'] = ids
                    if len(ids) != 1:
                        errors.append({'case_uid': row['case_uid'], 'error': 'missing or multiple Real Flight case ID'})
                finally:
                    book.close()
        records.append(entry)
    return {'case_count': len(records), 'shared_raw_log_name_candidates': duplicate_groups(records, 'raw_log_keys'), 'shared_testinfo_case_id_candidates': duplicate_groups(records, 'testinfo_case_id'), 'normal_capture_dates': dict(Counter(date for r in records if r['source_disposition'] == 'retain_normal' for date in r['capture_dates'])), 'errors': errors, 'records': records, 'limitation': 'No repeated filenames does not prove independent flights or devices; same-day capture is recorded but is not automatically an indivisible flight group.'}

def audit_versions(rows):
    prior = {r['case_uid']: r for r in read_csv(AUDIT)}
    hil_log = {r['output_filename']: r for r in read_csv(DATA / 'processing_logs/hil-50hz-20260813-110229/case_results.csv')}
    selected = [r for r in rows if r['domain'] == 'HIL' or (r['domain'].upper() == 'REAL' and r['level2_label'] == 'normal')]
    records = []
    for r in selected:
        path = Path(r['final_output_path'])
        digest = sha256(path) if path.is_file() else None
        records.append({'case_uid': r['case_uid'], 'domain': r['domain'], 'source_disposition': r['source_disposition'], 'path': str(path), 'sha256': digest, 'matches_manifest': digest == r['output_sha256'], 'matches_prior_audit': digest is not None and digest == prior.get(r['case_uid'], {}).get('sha256'), 'matches_hil_reprocessing': digest is not None and digest == hil_log.get(path.name, {}).get('output_sha256') if r['domain'] == 'HIL' else None})
    summary = {}
    for domain in sorted({r['domain'] for r in records}):
        group = [r for r in records if r['domain'] == domain]
        summary[domain] = {'files': len(group), 'missing': sum(r['sha256'] is None for r in group), 'manifest_hash_mismatches': sum(not r['matches_manifest'] for r in group), 'prior_audit_hash_mismatches': sum(not r['matches_prior_audit'] for r in group), 'hil_reprocessing_hash_mismatches': sum(not r['matches_hil_reprocessing'] for r in group) if domain == 'HIL' else None}
    return {'summary': summary, 'records': records}

def audit_cache(rows):
    import numpy as np
    import pandas as pd
    import torch
    sys.path.insert(0, str(ROOT / 'scripts/data'))
    from rfly_feature_contract import load_contract, read_case_features, window_batch
    torch.set_num_threads(2)
    by_uid = {r['case_uid']: r for r in rows}
    strata = defaultdict(list)
    for r in rows:
        if r['split'] in SPLITS and r['level2_label'] in {'normal', 'motor', 'sensor'}:
            strata[(r['split'], r['domain'], r['level2_label'])].append(r['case_uid'])
    selected = {uids[i] for uids in strata.values() for i in probe_positions(len(uids))}
    counts, spans = {}, defaultdict(list)
    usecols = ['case_uid', 'split', 'level2_label', 'window_start_row', 'window_end_row']
    for chunk in pd.read_csv(DATA / 'window_manifest_v1/window_manifest.csv', usecols=usecols, chunksize=200000):
        part = chunk[chunk['split'].isin(SPLITS) & chunk['level2_label'].isin(['normal', 'motor', 'sensor'])]
        for uid, count in part.groupby('case_uid', sort=False).size().items():
            counts[uid] = counts.get(uid, 0) + int(count)
        for rec in part[part['case_uid'].isin(selected)].to_dict(orient='records'):
            spans[rec['case_uid']].append((int(rec['window_start_row']), int(rec['window_end_row']), 0 if rec['level2_label'] == 'normal' else 1))
    cache = DATA / 'online_cache_v2'
    summary = json.loads((cache / 'build_summary.json').read_text(encoding='utf-8-sig'))
    cap = int(summary['max_windows_per_case'])
    if cap <= 0:
        raise ValueError('this audit expects a positive per-case cap')
    contract = load_contract()
    records, split_checks = [], []
    for split, filename in SPLITS.items():
        # weights_only avoids arbitrary object loading; mmap avoids loading every tensor into RAM.
        payload = torch.load(cache / filename, map_location='cpu', weights_only=True, mmap=True)
        x, y = payload['x'], payload['y']
        offset = 0
        for uid, n in counts.items():
            row = by_uid[uid]
            if row['split'] != split:
                continue
            size = min(n, cap)
            if uid in selected:
                all_spans = sorted(spans[uid])
                if len(all_spans) != n:
                    raise ValueError('window count mismatch: ' + uid)
                picks = np.unique(np.linspace(0, n - 1, cap).round().astype(int)) if n > cap else np.arange(n)
                positions = probe_positions(size)
                chosen = [all_spans[int(picks[i])] for i in positions]
                matrix, columns = read_case_features(row['final_output_path'], row['domain'], contract)
                expected = window_batch(matrix, columns, [(s, e) for s, e, label in chosen])
                actual = x[[offset + i for i in positions]].numpy()
                labels = y[[offset + i for i in positions]].numpy()
                delta = np.max(np.abs(expected.astype(np.float64) - actual.astype(np.float64)), axis=(0, 2))
                records.append({'case_uid': uid, 'split': split, 'domain': row['domain'], 'label': row['level2_label'], 'cache_positions': [offset + i for i in positions], 'windows_checked': len(positions), 'features_match': bool(np.allclose(expected, actual, rtol=1e-6, atol=1e-5, equal_nan=False)), 'labels_match': bool(np.array_equal(labels, np.array([label for s, e, label in chosen]))), 'max_absolute_difference': float(delta.max()), 'max_difference_channel': columns[int(delta.argmax())]})
            offset += size
        expected_base = summary['splits'][split].get('manifest_windows', summary['splits'][split]['windows'])
        split_checks.append({'split': split, 'manifest_samples': offset, 'expected_manifest_samples': expected_base, 'tensor_shape': list(x.shape), 'summary_samples': summary['splits'][split]['windows'], 'counts_match': offset == expected_base and len(x) == summary['splits'][split]['windows'] and len(y) == len(x)})
        del payload, x, y
    return {'selected_cases': len(selected), 'windows_checked': sum(r['windows_checked'] for r in records), 'case_probes': records, 'split_checks': split_checks, 'all_probes_match': all(r['features_match'] and r['labels_match'] for r in records) and all(r['counts_match'] for r in split_checks), 'limitation': 'Sampled current CSV-to-cache checks only; extra RflySim training windows and full tensor integrity are not revalidated.'}

def select_normal_roles(records, seed):
    """保守绑定 TestInfo 实验案例编号候选关联组，按固定哈希排序；不读模型分数。"""
    strata = defaultdict(lambda: defaultdict(list))
    for row in records:
        if row['source_disposition'] != 'retain_normal':
            continue
        ids = row['testinfo_case_id']
        if len(ids) != 1:
            raise ValueError('ambiguous raw case identity: ' + row['case_uid'])
        # A repeated TestInfo ID is an association candidate, not proof of the same flight or device.
        group = 'testinfo_case_id_candidate:' + ids[0]
        strata[row['flight_status']][group].append(row['case_uid'])
    assignments = {}
    for status, groups in sorted(strata.items()):
        ordered = sorted(groups, key=lambda key: (hashlib.sha256((seed + '|' + status + '|' + key).encode()).hexdigest(), key))
        used = set()
        for role, budget in [('adapt_normal', 2), ('calibrate_normal', 1)]:
            remaining = budget
            for group in ordered:
                if group not in used and len(groups[group]) <= remaining:
                    for uid in groups[group]:
                        assignments[uid] = {'role': role, 'association_group': group}
                    remaining -= len(groups[group])
                    used.add(group)
                if remaining == 0:
                    break
            if remaining:
                raise ValueError('normal budget cannot respect groups: ' + status + '/' + role)
        for group in ordered:
            if group not in used:
                for uid in groups[group]:
                    assignments[uid] = {'role': 'test_normal', 'association_group': group}
    return assignments

def membership_draft(report, source_path):
    sources = report['sources']
    if sources['errors'] or sources['shared_raw_log_name_candidates']:
        raise ValueError('resolve source metadata errors/shared raw logs before drafting')
    seed = 'rflymad-real-normal-adapt-draft-v1-20261029'
    assignments = select_normal_roles(sources['records'], seed)
    members = []
    for row in sources['records']:
        uid = row['case_uid']
        if uid in assignments:
            assignment = assignments[uid]
        elif row['source_disposition'] == 'retain_normal_only':
            assignment = {'role': 'supplemental_normal_test', 'association_group': 'testinfo_case_id_candidate:' + row['testinfo_case_id'][0]}
        else:
            assignment = {'role': 'test_fault', 'association_group': 'testinfo_case_id_candidate:' + row['testinfo_case_id'][0]}
        members.append({key: row[key] for key in ['case_uid', 'flight_status', 'source_disposition', 'level2_label']} | assignment)
    roles = defaultdict(set)
    for member in members:
        role = member['role']
        # Supplemental healthy portions stay under a test role for leakage checking.
        roles[member['association_group']].add('test' if role.startswith('test') or role == 'supplemental_normal_test' else role)
    if any(len(values) > 1 for values in roles.values()):
        raise ValueError('TestInfo case ID candidate group crosses adaptation/calibration/test')
    counts = dict(Counter(row['role'] for row in members))
    if counts != {'adapt_normal': 10, 'calibrate_normal': 5, 'test_normal': 33, 'supplemental_normal_test': 25, 'test_fault': 392}:
        raise ValueError('unexpected membership counts: ' + str(counts))
    return {'status': 'DRAFT_NOT_FROZEN', 'created_at': datetime.now().astimezone().isoformat(), 'seed': seed, 'selection': 'Within flight status, sort TestInfo case-ID candidate groups by SHA256(seed|status|group); fill 2 adaptation cases then 1 calibration case without splitting a group; all others test. No prediction scores used.', 'source_audit_path': str(source_path), 'source_audit_sha256': sha256(source_path), 'role_counts': counts, 'members': members, 'limitations': ['No device-independent or date-held-out claim; the same TestInfo case ID is conservatively grouped as an association candidate, not proven to be one flight or device.', 'Prior Real test results were already viewed.', 'Budget and rules need confirmation; no production split, cache or model changed.']}

def freeze_protocol(report, report_path, roles_output, protocol_output):
    """Write the frozen Real role overlay and protocol metadata without changing base data."""
    roles_output = Path(roles_output)
    protocol_output = Path(protocol_output)
    if roles_output.exists() or protocol_output.exists():
        raise FileExistsError('refusing to overwrite frozen protocol outputs')
    draft = membership_draft(report, report_path)
    manifest_rows = {row['case_uid']: row for row in read_csv(MANIFEST)}
    fieldnames = [
        'case_uid', 'domain', 'flight_status', 'level2_label',
        'source_disposition', 'role', 'association_group', 'testinfo_case_id',
        'source_csv', 'final_output_path', 'output_sha256', 'row_count',
    ]
    rows = []
    for member in sorted(draft['members'], key=lambda item: item['case_uid']):
        manifest_row = manifest_rows[member['case_uid']]
        source = next(item for item in report['sources']['records'] if item['case_uid'] == member['case_uid'])
        rows.append({
            'case_uid': member['case_uid'],
            'domain': manifest_row['domain'],
            'flight_status': member['flight_status'],
            'level2_label': member['level2_label'],
            'source_disposition': member['source_disposition'],
            'role': member['role'],
            'association_group': member['association_group'],
            'testinfo_case_id': source['testinfo_case_id'][0],
            'source_csv': manifest_row['final_output_path'],
            'final_output_path': manifest_row['final_output_path'],
            'output_sha256': manifest_row['output_sha256'],
            'row_count': manifest_row['row_count'],
        })
    role_counts = dict(Counter(row['role'] for row in rows))
    if len(rows) != 465 or role_counts != {
        'adapt_normal': 10,
        'calibrate_normal': 5,
        'test_normal': 33,
        'supplemental_normal_test': 25,
        'test_fault': 392,
    }:
        raise ValueError('unexpected frozen role counts: ' + str(role_counts))
    roles_output.parent.mkdir(parents=True, exist_ok=True)
    with roles_output.open('x', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    cache_identity_path = ROOT / 'artifacts/normal_adaptation_audit_20261029/cache_identity.json'
    cache_identity = json.loads(cache_identity_path.read_text(encoding='utf-8-sig')) if cache_identity_path.exists() else None
    protocol = {
        'protocol_name': 'rflymad_online_normal_adapt_v1',
        'status': 'FROZEN',
        'frozen_at': datetime.now().astimezone().isoformat(),
        'base_case_split': 'rflymad_case_split_v2',
        'base_manifest_sha256': report['manifest_sha256'],
        'role_manifest': str(roles_output),
        'role_manifest_sha256': sha256(roles_output),
        'source_audit_report': str(report_path),
        'source_audit_sha256': sha256(report_path),
        'cache_identity_report': str(cache_identity_path) if cache_identity is not None else None,
        'cache_identity': cache_identity,
        'role_counts': role_counts,
        'role_rules': {
            'adapt_normal': '10 retain_normal cases: 2 per flight_status, selected by fixed SHA256 ordering of TestInfo case-ID candidate groups',
            'calibrate_normal': '5 retain_normal cases: 1 per flight_status, selected by the same ordering after adaptation groups',
            'test_normal': 'remaining retain_normal cases',
            'test_fault': 'all 392 Real motor/sensor cases',
            'supplemental_normal_test': 'all 25 retain_normal_only cases, reported separately',
            'association_rule': 'same TestInfo Real Flight case ID candidate group cannot cross adaptation, calibration and test roles; this does not assert same flight or device',
        },
        'source_training': {
            'cache': 'online_cache_v2',
            'source_domains': ['SIL', 'HIL'],
            'source_fault_classes': ['motor', 'sensor'],
            'rflysim_normal_train_sessions': 32,
            'rflysim_normal_holdout_sessions': 8,
            'real_fault_or_normal_training': False,
            'input_norm_stats': 'source training statistics only',
        },
        'input_contract': {
            'feature_schema': 'rfly_seq_columns_v2',
            'feature_count': 26,
            'target_hz': 50.0,
            'window_size_frames': 150,
            'window_seconds': 3.0,
            'manifest_stride_frames': 5,
            'max_windows_per_case': 64,
            'window_label': 'causal endpoint normal/fault label',
        },
        'model_arms': {
            'A0_no_target_adaptation': 'source best checkpoint, source normalization, no Real reads',
            'A1_normal_bn_recalibration': {
                'source_checkpoint_same_as_A0': True,
                'real_role': 'adapt_normal',
                'gradient_updates': False,
                'parameters_frozen': True,
                'reset_batchnorm_running_stats': True,
                'batch_size': 64,
                'num_workers': 0,
                'shuffle': False,
                'window_order': 'case_uid then window_end_row',
                'dropout_training': False,
                'input_norm_stats': 'source training statistics only',
                'only_for_models_with_batchnorm': True,
            },
        },
        'threshold_arms': {
            'T0_fixed': {'onset_threshold': 0.60, 'clear_threshold': 0.40, 'k_on': 3, 'k_off': 3, 'min_event_s': 0.5, 'merge_gap_s': 0.5, 'inference_every_frames': 10},
            'T1_normal_calibrated': {'calibration_role': 'calibrate_normal', 'quantile': 0.995, 'quantile_method': 'linear', 'onset_threshold': 'clip(max(0.60, q), 0.60, 0.95)', 'clear_threshold': 0.40, 'k_on': 3, 'k_off': 3, 'min_event_s': 0.5, 'merge_gap_s': 0.5, 'inference_every_frames': 10},
        },
        'evaluation': {
            'main_real_test_roles': ['test_normal', 'test_fault'],
            'supplemental_real_test_roles': ['supplemental_normal_test'],
            'primary_metrics': ['fault_case_event_recall', 'motor_case_recall', 'sensor_case_recall', 'normal_case_event_false_alarm_rate', 'false_alarm_events_per_hour', 'event_detection_latency_distribution'],
            'secondary_metrics': ['window_accuracy', 'window_macro_f1', 'window_fault_recall'],
            'pre_onset_alarm': 'report separately; never convert to negative latency',
            'latency': 'event_start after ground_truth_onset minus ground_truth_onset; missing when no post-onset event',
            'real_replay_label': 'retrospective streaming replay until causal preprocessing is separately verified',
            'rflysim_holdout': 'report separately from RflyMAD Real',
        },
        'frozen_boundaries': [
            'Real fault cases are never used for training, adaptation, threshold calibration, early stopping or model selection.',
            'Base case split and source cache are not overwritten.',
            'Any change to members, adaptation method, threshold rule or preprocessing requires protocol v2.',
        ],
    }
    protocol_output.parent.mkdir(parents=True, exist_ok=True)
    with protocol_output.open('x', encoding='utf-8') as stream:
        json.dump(protocol, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
    return {'roles': str(roles_output), 'protocol': str(protocol_output), 'role_manifest_sha256': protocol['role_manifest_sha256'], 'role_counts': role_counts}

def self_test():
    assert probe_positions(1) == [0]
    assert probe_positions(6) == [0, 3, 5]
    try:
        probe_positions(0)
    except ValueError:
        pass
    else:
        raise AssertionError('empty case accepted')
    records = [{'case_uid': 'normal/a', 'logs': ['shared', 'shared']}, {'case_uid': 'fault/b', 'logs': ['shared']}, {'case_uid': 'other/c', 'logs': ['unique']}]
    assert duplicate_groups(records, 'logs') == [{'key': 'shared', 'case_uids': ['fault/b', 'normal/a']}]
    normal = [{'case_uid': 'n' + str(i), 'flight_status': 'hover', 'source_disposition': 'retain_normal', 'testinfo_case_id': [str(i // 2) if i < 2 else str(i)]} for i in range(8)]
    picked = select_normal_roles(normal, 'check')
    assert picked == select_normal_roles(list(reversed(normal)), 'check')
    assert Counter(r['role'] for r in picked.values()) == {'adapt_normal': 2, 'calibrate_normal': 1, 'test_normal': 5}
    assert picked['n0']['role'] == picked['n1']['role']
    try:
        select_normal_roles(normal[:2], 'check')
    except ValueError:
        pass
    else:
        raise AssertionError('infeasible group budget accepted')
    print('PASS: boundary sampling, cross-case groups, deterministic roles and infeasible budgets')

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--self-test', action='store_true')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--draft-from', type=Path, help='Use an existing audit to write a review-only membership draft')
    parser.add_argument('--freeze-from', type=Path, help='Freeze the formal role overlay and protocol from an audit report')
    parser.add_argument('--roles-output', type=Path)
    parser.add_argument('--protocol-output', type=Path)
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return
    if args.freeze_from:
        if args.roles_output is None or args.protocol_output is None:
            parser.error('--freeze-from requires --roles-output and --protocol-output')
        report = json.loads(args.freeze_from.read_text(encoding='utf-8-sig'))
        result = freeze_protocol(report, args.freeze_from, args.roles_output, args.protocol_output)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return
    if args.output is None:
        parser.error('--output is required; existing report files are never overwritten')
    if args.output.exists():
        raise FileExistsError(args.output)
    if args.draft_from:
        report = json.loads(args.draft_from.read_text(encoding='utf-8-sig'))
        draft = membership_draft(report, args.draft_from)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open('x', encoding='utf-8') as stream:
            json.dump(draft, stream, ensure_ascii=False, indent=2)
        print(json.dumps({'status': draft['status'], 'output': str(args.output), 'role_counts': draft['role_counts']}, ensure_ascii=False, indent=2))
        return
    rows = read_csv(MANIFEST)
    print('Auditing raw source metadata...', flush=True)
    sources = audit_sources(rows)
    print('Hashing HIL and Real-normal current CSVs...', flush=True)
    versions = audit_versions(rows)
    print('Probing cached windows against current features...', flush=True)
    cache = audit_cache(rows)
    result = {'checked_at': datetime.now().astimezone().isoformat(), 'script_sha256': sha256(__file__), 'manifest_sha256': sha256(MANIFEST), 'sources': sources, 'versions': versions, 'cache': cache, 'mutations': 'Only this new JSON report; no source, cache, split, model or remote changes.'}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x', encoding='utf-8') as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
    print(json.dumps({'report': str(args.output), 'source_metadata_errors': len(sources['errors']), 'shared_raw_candidates': len(sources['shared_raw_log_name_candidates']), 'shared_case_id_candidates': len(sources['shared_testinfo_case_id_candidates']), 'versions': versions['summary'], 'cache_probes_match': cache['all_probes_match'], 'windows_checked': cache['windows_checked']}, ensure_ascii=False, indent=2))

if __name__ == '__main__':
    main()
