"""Finish fixed-crop label repair for the frozen SIL/HIL internal test only.

No models, scores, tuning, training, crop or feature changes. Every original
cache index is retained in a mask-aware evidence bundle, not generic y/test.pt.
"""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import sys

import numpy as np
import pandas as pd
import torch
from pyulog import ULog

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts/research'))
from audit_source_event_boundaries import MANIFEST, EXPECTED_MANIFEST, read_gtd
from validate_fixed_crop_labels import microseconds
from audit_rflymad_raw_sources import active_ids, digest
from audit_real_labeltime import hold_integer
from repair_fixed_crop_labeltime import repair_csv
from rfly_feature_contract import read_case_features, source_columns_for_domain, window_batch

ART = ROOT / 'artifacts/internal_test_labeltime_v1_20261031'
OUT = Path('D:/UAV-Data/internal_test_labeltime_v1')
OLD = Path('D:/UAV-Data/online_cache_v2')
WINDOWS = Path('D:/UAV-Data/window_manifest_v1/window_manifest.csv')
CONTRACT = Path('D:/UAV-Data/feature_contract_v1.json')
VERSION = 'rflymad_internal_test_labeltime_v1'
STATE = 'UAVState_data_fault_state'


def save(path, data):
    with Path(path).open('x', encoding='utf-8') as stream:
        json.dump(data, stream, ensure_ascii=False, indent=2, allow_nan=False)


def cases():
    if digest(MANIFEST) != EXPECTED_MANIFEST:
        raise ValueError('Frozen case split identity changed')
    with MANIFEST.open(encoding='utf-8-sig', newline='') as stream:
        rows = [r for r in csv.DictReader(stream) if r['domain'] in ['SIL', 'HIL']
                and r['split'] == 'test' and r['level2_label'] in ['normal', 'motor', 'sensor']]
    if len(rows) != 553 or Counter(r['domain'] for r in rows) != {'SIL': 276, 'HIL': 277}:
        raise ValueError('Frozen test scope changed')
    # Existing documented HIL version reconciliation, not a new reprocessing.
    with (ROOT / 'docs/research/2026-10-08/data_feasibility/full_case_audit.csv').open(encoding='utf-8-sig', newline='') as f:
        prior = {r['case_uid']: r for r in csv.DictReader(f)}
    with Path('D:/UAV-Data/processing_logs/hil-50hz-20260813-110229/case_results.csv').open(encoding='utf-8-sig', newline='') as f:
        reprocessed = {r['output_filename']: r for r in csv.DictReader(f)}
    for row in rows:
        row['split_manifest_output_sha256'] = row['output_sha256']
        if row['domain'] == 'HIL':
            expected = reprocessed[Path(row['final_output_path']).name]['output_sha256']
            if expected != prior[row['case_uid']]['sha256']:
                raise ValueError('Previously reconciled HIL identities disagree')
            row['output_sha256'] = expected
            row['processed_identity_basis'] = 'verified_20260813_reprocessing_and_20261008_audit'
        else:
            row['processed_identity_basis'] = 'frozen_split_manifest'
    return rows


def eligible(reference, command, covered, command_covered, clock_review, input_finite):
    for x in [reference, command, covered, command_covered, input_finite]:
        if x.shape != reference.shape:
            raise ValueError('Reference/eligibility shape mismatch')
    agree = covered & command_covered & (reference == command)
    label = np.where(agree, reference, -1).astype(np.int8)
    valid = agree & ~np.full(len(agree), bool(clock_review), dtype=bool) & input_finite
    return label, valid


def phase_info(state, start, end):
    seq = state[start:end]
    onset_rows = np.flatnonzero(state == 1)
    onset = int(onset_rows[0]) if len(onset_rows) else -1
    if (seq < 0).any():
        return 'unknown_reference', None, onset
    if np.all(seq == 0):
        phase = 'normal_flight' if onset < 0 else ('pre_fault' if end <= onset else 'post_fault_normal')
    elif np.all(seq == 1):
        phase = 'stable_fault'
    else:
        phase = 'transition'
    return phase, float((seq == 1).mean()), onset


def check(path, expected):
    if digest(path) != expected:
        raise ValueError('Source identity changed: ' + str(path))


def initialize():
    if OUT.exists():
        raise FileExistsError('Use independent new output; existing output never overwritten')
    rows = cases()
    if shutil.disk_usage(OUT.parent).free < 8_000_000_000:
        raise ValueError('Insufficient space for independent repair')
    identities = {str(p): digest(p) for p in [MANIFEST, CONTRACT, OLD / 'norm_stats.json', OLD / 'test.pt',
        ART / 'repair_plan.json', ROOT / 'scripts/data/repair_fixed_crop_labeltime.py']}
    OUT.mkdir(parents=True, exist_ok=False)
    for name in ['cases', 'references', 'parts']:
        (OUT / name).mkdir()
    shutil.copyfile(OLD / 'norm_stats.json', OUT / 'norm_stats.json')
    pd.DataFrame(rows).to_csv(OUT / 'historical_case_membership.csv', index=False)
    save(OUT / 'RUNNING.json', {'version': VERSION, 'started_at': datetime.now(timezone.utc).isoformat(),
        'script_sha256': digest(__file__), 'identities': identities, 'scope': '553 internal test cases only; never fit or tune on final test'})
    print('INITIALIZED_INTERNAL_TEST_REPAIR', len(rows), flush=True)


def build_window_metadata(row, q, old, ref, cmd, row_valid, finite, st, raw_sensor_bad, reversal, target):
    uid = row['case_uid']
    original = Path(row['final_output_path'])
    gaps = np.flatnonzero(np.diff(st) > 100000)
    gap_left, gap_right = st[gaps], st[gaps + 1]
    count = (len(q) - 150) // 5 + 1
    starts = np.arange(count, dtype=int) * 5
    if count > 64:
        starts = starts[np.unique(np.linspace(0, count - 1, 64).round().astype(int))]
    if len(starts) != 64:
        raise ValueError('Expected original cap64 per frozen test case')
    windows = []
    for i, s in enumerate(starts):
        s = int(s); e = s + 150; ix = e - 1
        has_bad = bool(((st >= q[s]) & (st <= q[ix]) & raw_sensor_bad).any())
        whole_input_finite = bool(finite[s:e].all())
        valid = bool(row_valid[ix] and whole_input_finite and not has_bad)
        reasons = []
        if ref[ix] != cmd[ix]: reasons.append('GTD_ULog_endpoint_state_disagreement')
        if reversal: reasons.append('original_inertial_clock_review')
        if not whole_input_finite or has_bad: reasons.append('required_input_nonfinite_review')
        phase, fraction, onset = phase_info(ref, s, e)
        windows.append({'case_uid': uid, 'domain': row['domain'], 'split': 'test', 'case_level2_label': row['level2_label'],
            'case_sample_index': i, 'window_start_row': s, 'window_end_row': e, 'existing_label': int(old[ix]),
            'gtd_reference_label': int(ref[ix]), 'command_reference_label': int(cmd[ix]),
            'qualified_label': int(ref[ix]) if valid else -1, 'endpoint_evaluation_valid': valid,
            'source_references_agree': bool(ref[ix] == cmd[ix]), 'input_clock_review': reversal,
            'required_input_finite': whole_input_finite and not has_bad,
            'sensor_gap_over_0_1s_intersects_window': bool(((gap_left < q[ix]) & (gap_right > q[s])).any()),
            'review_reason': '|'.join(reasons), 'phase': phase, 'fault_fraction': fraction, 'fault_onset_row': onset,
            'whole_window_label_valid': bool(row_valid[s:e].all()), 'source_csv': str(target),
            'historical_source_csv': str(original)})
    return windows


def prepare_case(row):
    uid = row['case_uid']; raw = Path(row['source_path']); original = Path(row['final_output_path'])
    check(original, row['output_sha256'])
    source_stat = (original.stat().st_size, original.stat().st_mtime_ns)
    gtd, meta = raw / 'TrueData/UAVState_data.xlsx', raw / 'TestInfo.csv'
    logs = list((raw / 'Log').glob('*.ulg'))
    if len(logs) != 1:
        raise ValueError('Expected one original ULog: ' + uid)
    identities = {str(p): digest(p) for p in [gtd, meta, logs[0]]}
    frame = pd.read_csv(original, usecols=['Timestamp', STATE])
    stamps, previous = frame.Timestamp.to_numpy(dtype=float), frame[STATE].to_numpy(dtype=float)
    if not np.isfinite(stamps).all() or not np.equal(stamps, np.rint(stamps)).all():
        raise ValueError('Invalid original timestamp')
    q = stamps.astype(np.int64); old = previous.astype(np.int8)
    if len(q) != int(row['row_count']) or not np.all(np.diff(q) == 20000) or not np.isin(previous, [0, 1]).all():
        raise ValueError('Original crop/label invalid')
    gt, gs = read_gtd(gtd)
    gu = microseconds(gt)
    ref, gc = hold_integer(gu, gs, q)
    log = ULog(str(logs[0]), message_name_filter_list=['rfly_ctrl_lxl', 'sensor_combined'])
    commands = [d.data for d in log.data_list if d.name == 'rfly_ctrl_lxl' and d.multi_id == 0]
    sensors = [d.data for d in log.data_list if d.name == 'sensor_combined' and d.multi_id == 0]
    if len(commands) != 1 or len(sensors) != 1:
        raise ValueError('Missing or ambiguous required original topics: ' + uid)
    ct = commands[0]['timestamp'].astype(np.int64)
    cmd, cc = hold_integer(ct, active_ids(commands[0]['id']), q)
    st = sensors[0]['timestamp'].astype(np.int64)
    if len(st) < 2:
        raise ValueError('Missing original inertial clock')
    reversal = bool((np.diff(st) < 0).any())
    gap_at = np.flatnonzero(np.diff(st) > 100000)
    gap_left, gap_right = st[gap_at], st[gap_at + 1]
    raw_sensor_bad = np.zeros(len(st), dtype=bool)
    for name in ['gyro_rad[0]', 'gyro_rad[1]', 'gyro_rad[2]', 'accelerometer_m_s2[0]', 'accelerometer_m_s2[1]', 'accelerometer_m_s2[2]']:
        if name not in sensors[0]:
            raise ValueError('Missing required original inertial field: ' + name)
        raw_sensor_bad |= ~np.isfinite(sensors[0][name])
    mapping = source_columns_for_domain(json.loads(CONTRACT.read_text(encoding='utf-8-sig')), row['domain'])
    inputs = pd.read_csv(original, usecols=sorted(set(mapping.values())))
    finite = np.isfinite(inputs.apply(pd.to_numeric, errors='raise').to_numpy(dtype=float)).all(axis=1)
    reference_label, row_valid = eligible(ref, cmd, gc, cc, reversal, finite)
    # Same full-coverage condition as the completed source repair; fail rather
    # than silently invent a new extrapolation convention for final tests.
    if not gc.all() or not cc.all():
        raise ValueError('Incomplete original reference coverage requires explicit review: ' + uid)
    if row['source_fault_type'] == 'no_fault' and ((gs == 1).any() or active_ids(commands[0]['id']).any()):
        raise ValueError('Frozen normal family contains recorded original injection: ' + uid)
    sidecar = OUT / 'references' / (uid.replace('/', '__') + '.npz')
    np.savez_compressed(sidecar, timestamp_us=q, existing_state=old, gtd_state_reference=ref,
        command_state_reference=cmd, gtd_covered=gc, command_covered=cc, source_references_agree=gc & cc & (ref == cmd),
        qualified_reference=reference_label, row_label_valid=row_valid, required_input_row_finite=finite,
        original_sensor_timestamp_us=st, original_sensor_required_field_nonfinite=raw_sensor_bad)
    target = OUT / 'cases' / row['domain'] / original.name
    info = repair_csv(original, target, sidecar, {'processed_csv_sha256': row['output_sha256'],
        'sidecar_sha256': digest(sidecar), 'source_rows': len(q)}, reversal)
    windows = build_window_metadata(row, q, old, ref, cmd, row_valid, finite, st, raw_sensor_bad, reversal, target)
    if (original.stat().st_size, original.stat().st_mtime_ns) != source_stat:
        raise ValueError('Source changed during read')
    record = dict(row, historical_final_output_path=str(original), final_output_path=str(target),
        original_identities=identities, reference_file=str(sidecar), sidecar_sha256=digest(sidecar),
        original_inertial_clock_review=reversal, sensor_nonfinite_samples=int(raw_sensor_bad.sum()),
        changed_endpoint_labels=sum(w['endpoint_evaluation_valid'] and w['existing_label'] != w['qualified_label'] for w in windows),
        diagnostic_changed_endpoints=sum(w['existing_label'] != w['gtd_reference_label'] for w in windows),
        uncertain_endpoints=sum(not w['endpoint_evaluation_valid'] for w in windows),
        raw_onset_after_crop=bool(np.any(gs) and gu[np.flatnonzero(gs)[0]] > q[-1]), **info)
    return record, windows


def prepare_part(part):
    running = json.loads((OUT / 'RUNNING.json').read_text(encoding='utf-8'))
    execution = json.loads((OUT / 'EXECUTION_REVISION.json').read_text(encoding='utf-8')) if (OUT / 'EXECUTION_REVISION.json').exists() else running
    if digest(__file__) != execution['script_sha256']:
        raise ValueError('Script changed since plan execution')
    rows = cases(); size = 111
    selected = rows[(part - 1) * size:part * size]
    if not selected:
        raise ValueError('Empty test partition')
    target = OUT / 'parts' / ('part' + str(part) + '.json')
    if target.exists():
        raise FileExistsError('Partition already completed')
    records, windows = [], []
    ledger = OUT / 'parts' / ('part' + str(part) + '.jsonl')
    if ledger.exists():
        records = [json.loads(s) for s in ledger.read_text(encoding='utf-8').splitlines()]
        done = {r['case_uid']: r for r in records}
        members = {r['case_uid']: r for r in selected}
        if len(done) != len(records) or not set(done).issubset(members):
            raise ValueError('Interrupted partition membership mismatch')
        for r in records:
            check(Path(r['final_output_path']), r['new_csv_sha256'])
            check(Path(r['reference_file']), r['sidecar_sha256'])
            check(Path(r['historical_final_output_path']), r['output_sha256'])
            for path, expected in r['original_identities'].items(): check(Path(path), expected)
            with np.load(r['reference_file'], allow_pickle=False) as d:
                windows.extend(build_window_metadata(members[r['case_uid']], d['timestamp_us'], d['existing_state'],
                    d['gtd_state_reference'], d['command_state_reference'], d['row_label_valid'], d['required_input_row_finite'],
                    d['original_sensor_timestamp_us'], d['original_sensor_required_field_nonfinite'],
                    r['original_inertial_clock_review'], Path(r['final_output_path'])))
        selected = [r for r in selected if r['case_uid'] not in done]
        for row in selected:
            for leftover in [OUT / 'cases' / row['domain'] / Path(row['final_output_path']).name,
                             OUT / 'references' / (row['case_uid'].replace('/', '__') + '.npz')]:
                if leftover.exists():
                    saved = OUT / 'interrupted_files' / leftover.relative_to(OUT)
                    saved.parent.mkdir(parents=True, exist_ok=True)
                    if saved.exists(): raise FileExistsError('Previous interruption snapshot exists')
                    leftover.replace(saved)
    with ledger.open('a' if ledger.exists() else 'x', encoding='utf-8') as stream:
        with ThreadPoolExecutor(max_workers=3) as pool:
            for i, (r, w) in enumerate(pool.map(prepare_case, selected), len(records) + 1):
                records.append(r); windows.extend(w)
                stream.write(json.dumps(r, ensure_ascii=False, allow_nan=False) + chr(10)); stream.flush()
                if i % 30 == 0:
                    print('INTERNAL_TEST_REPAIRED_PART', part, i, '/', len(selected), flush=True)
    pd.DataFrame(windows).to_csv(OUT / 'parts' / ('part' + str(part) + '_windows.csv'), index=False)
    summary = {'cases': len(records), 'windows': len(windows), 'changed_rows': sum(r['changed_state_rows'] for r in records),
        'diagnostic_changed_endpoints': sum(r['diagnostic_changed_endpoints'] for r in records),
        'corrected_qualified_endpoints': sum(r['changed_endpoint_labels'] for r in records),
        'uncertain_endpoints': sum(r['uncertain_endpoints'] for r in records),
        'clock_review_cases': sum(r['original_inertial_clock_review'] for r in records)}
    save(target, summary); print(json.dumps(summary, ensure_ascii=False), flush=True)


def cache():
    if (OUT / 'repair_manifest.json').exists():
        raise FileExistsError('Completed version must not be overwritten')
    running = json.loads((OUT / 'RUNNING.json').read_text(encoding='utf-8'))
    execution = json.loads((OUT / 'EXECUTION_REVISION.json').read_text(encoding='utf-8')) if (OUT / 'EXECUTION_REVISION.json').exists() else running
    check(Path(__file__), execution['script_sha256'])
    for path, expected in running['identities'].items(): check(Path(path), expected)
    records, candidates = [], []
    for part in range(1, 6):
        json.loads((OUT / 'parts' / ('part' + str(part) + '.json')).read_text(encoding='utf-8'))
        records.extend(json.loads(s) for s in (OUT / 'parts' / ('part' + str(part) + '.jsonl')).read_text(encoding='utf-8').splitlines())
        candidates.append(pd.read_csv(OUT / 'parts' / ('part' + str(part) + '_windows.csv'), keep_default_na=False))
    candidates = pd.concat(candidates, ignore_index=True)
    by_uid = {r['case_uid']: r for r in records}
    if len(records) != 553 or set(by_uid) != {r['case_uid'] for r in cases()}:
        raise ValueError('Incomplete test membership')
    use = ['case_uid', 'domain', 'split', 'level2_label', 'window_start_row', 'window_end_row', 'source_csv']
    manifest_hash = digest(WINDOWS); frames = []
    for chunk in pd.read_csv(WINDOWS, usecols=use, chunksize=200000):
        selected = chunk[(chunk.split == 'test') & chunk.domain.isin(['SIL', 'HIL']) & chunk.level2_label.isin(['normal', 'motor', 'sensor'])]
        if len(selected): frames.append(selected)
    full = pd.concat(frames, ignore_index=True); parts = []
    for uid, group in full.groupby('case_uid', sort=False):
        group = group.sort_values('window_start_row'); n = int(by_uid[uid]['row_count'])
        if group.window_start_row.tolist() != list(range(0, n - 149, 5)) or not np.all(group.window_end_row - group.window_start_row == 150):
            raise ValueError('Original manifest grid mismatch')
        picks = np.unique(np.linspace(0, len(group) - 1, 64).round().astype(int))
        group = group.iloc[picks].copy(); group['case_sample_index'] = np.arange(len(group)); parts.append(group)
    original = pd.concat(parts, ignore_index=True)
    original = original.rename(columns={'level2_label': 'historical_level2_label', 'source_csv': 'manifest_historical_source_csv'})
    original.insert(0, 'source_cache_index', np.arange(len(original)))
    meta = original.merge(candidates.drop(columns=['domain', 'split']),
        on=['case_uid', 'case_sample_index', 'window_start_row', 'window_end_row'], how='left', validate='one_to_one', indicator=True, sort=False)
    if len(meta) != 35392 or len(meta) != len(candidates) or not (meta['_merge'] == 'both').all():
        raise ValueError('Original selected index mapping differs')
    meta = meta.drop(columns='_merge').sort_values('source_cache_index').reset_index(drop=True)
    torch.set_num_threads(4)
    old = torch.load(OLD / 'test.pt', map_location='cpu', weights_only=True, mmap=True)
    if old['x'].shape != (35392, 26, 150) or old['x'].dtype != torch.float32:
        raise ValueError('Unexpected old test X')
    if not np.array_equal(old['y'].numpy(), meta.existing_label) or not np.array_equal((meta.historical_level2_label != 'normal').astype(int), meta.existing_label):
        raise ValueError('Historical test y/order mismatch')
    contract = json.loads(CONTRACT.read_text(encoding='utf-8-sig')); norm = json.loads((OUT / 'norm_stats.json').read_text(encoding='utf-8'))
    for i, (uid, group) in enumerate(meta.groupby('case_uid', sort=False), 1):
        r = by_uid[uid]
        check(Path(r['final_output_path']), r['new_csv_sha256']); check(Path(r['historical_final_output_path']), r['output_sha256'])
        check(Path(r['reference_file']), r['sidecar_sha256'])
        matrix, columns = read_case_features(r['historical_final_output_path'], r['domain'], contract)
        if columns != norm['columns']: raise ValueError('Channel order mismatch')
        spans = list(zip(group.window_start_row.astype(int), group.window_end_row.astype(int)))
        expected = torch.from_numpy(window_batch(matrix, columns, spans))
        ids = group.source_cache_index.to_numpy(dtype=np.int64)
        if not torch.equal(expected, old['x'][ids]): raise ValueError('Original X mapping mismatch: ' + uid)
        if i % 100 == 0: print('INTERNAL_TEST_X_IDENTITIES', i, '/553', flush=True)
    valid = meta.endpoint_evaluation_valid.to_numpy(dtype=bool)
    qualified = meta.qualified_label.to_numpy(dtype=np.int64)
    if not np.array_equal(qualified, np.where(valid, meta.gtd_reference_label, -1)):
        raise ValueError('Mask/label mismatch')
    bundle = {'x': old['x'].clone(), 'historical_y': old['y'].clone(),
        'gtd_reference_y': torch.from_numpy(meta.gtd_reference_label.to_numpy(dtype=np.int64)),
        'command_reference_y': torch.from_numpy(meta.command_reference_label.to_numpy(dtype=np.int64)),
        'qualified_reference_y': torch.from_numpy(qualified), 'endpoint_evaluation_valid': torch.from_numpy(valid),
        'source_cache_index': torch.arange(len(meta), dtype=torch.int64)}
    target = OUT / 'internal_test_evidence.pt'; partial = OUT / 'internal_test_evidence.pt.partial'
    torch.save(bundle, partial); del bundle; partial.replace(target)
    actual = torch.load(target, map_location='cpu', weights_only=True, mmap=True)
    for s in range(0, len(meta), 512):
        if not torch.equal(actual['x'][s:s + 512], old['x'][s:s + 512]): raise ValueError('Persisted X differs')
    if not torch.equal(actual['qualified_reference_y'], torch.from_numpy(qualified)): raise ValueError('Persisted targets differ')
    meta['level2_label'] = np.where(qualified < 0, 'uncertain', np.where(qualified == 0, 'normal', meta.case_level2_label))
    meta.to_csv(OUT / 'test_window_links.csv', index=False)
    meta[~meta.endpoint_evaluation_valid].to_csv(OUT / 'evaluation_ineligible_window_ledger.csv', index=False)
    pd.DataFrame([{k: v for k, v in r.items() if k != 'original_identities'} for r in records]).to_csv(OUT / 'case_manifest.csv', index=False)
    for path, expected in running['identities'].items(): check(Path(path), expected)
    check(WINDOWS, manifest_hash); check(OUT / 'norm_stats.json', digest(OLD / 'norm_stats.json'))
    summary = {'status': 'COMPLETE_INTERNAL_TEST_LABELTIME_REPAIR_WITH_EXPLICIT_MASKS', 'version': VERSION,
        'completed_at': datetime.now(timezone.utc).isoformat(), 'script_sha256': digest(__file__), 'cases': 553, 'windows': len(meta),
        'changed_state_rows': sum(r['changed_state_rows'] for r in records),
        'diagnostic_changed_endpoint_labels': int((meta.existing_label != meta.gtd_reference_label).sum()),
        'corrected_qualified_endpoint_labels': int(((meta.existing_label != qualified) & valid).sum()),
        'eligible_endpoints': int(valid.sum()), 'ineligible_endpoints': int((~valid).sum()),
        'review_reasons': dict(Counter(str(x) for x in meta.loc[~meta.endpoint_evaluation_valid, 'review_reason'])),
        'clock_review_cases': sum(r['original_inertial_clock_review'] for r in records),
        'gap_flagged_windows_retained': int(meta.sensor_gap_over_0_1s_intersects_window.sum()),
        'after_crop_activation_cases': sum(r['raw_onset_after_crop'] for r in records),
        'all_original_X_preserved': True, 'input_identities': {**running['identities'], str(WINDOWS): manifest_hash},
        'outputs_sha256': {name: digest(OUT / name) for name in ['internal_test_evidence.pt', 'test_window_links.csv',
            'case_manifest.csv', 'evaluation_ineligible_window_ledger.csv', 'norm_stats.json']},
        'models_or_training_used': False, 'event_scoring_ready': False,
        'scope_limit': 'Final-test label correction only. No physical-onset, full input-quality or online causality certification. Unknowns retained, no generic y/test.pt or historical score update.'}
    save(OUT / 'repair_manifest.json', summary); print(json.dumps({k:v for k,v in summary.items() if k not in ['input_identities','outputs_sha256']}, ensure_ascii=False, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=['init', 'cases', 'cache'], required=True)
    parser.add_argument('--part', type=int, choices=range(1, 6), default=1)
    args = parser.parse_args()
    if args.stage == 'init': initialize()
    elif args.stage == 'cases': prepare_part(args.part)
    else: cache()


if __name__ == '__main__': main()
