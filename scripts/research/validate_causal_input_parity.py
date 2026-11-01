"""Source-only small-scale input-rule fit and batch/stream parity verification.

No models, training, final-test/Real reads, old-cache changes or crop changes.
Timing recipe and case selection are fixed in the saved plan before this runs.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
import hashlib
import heapq
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from pyulog import ULog

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'packages/uav_fault'))
from uav_fault.causal_pipeline import (GROUPS, TopicTrace, CausalInputStream, batch_frames, batch_windows, STALE)
from uav_fault.causal_input import canonical_window
from audit_rflymad_raw_sources import digest

OUT = ROOT / 'artifacts/causal_input_parity_v1_20261031'
MANIFEST = Path('D:/UAV-Data/splits/rflymad_case_split_v2/rflymad_case_level_split_manifest.csv')
FEATURE = Path('D:/UAV-Data/feature_contract_v1.json')
PRIOR = ROOT / 'artifacts/fixed_crop_label_validation_20261031/case_results.jsonl'
GAP_CASES = ['HIL/hover/motor/TestCase_181_2000000180', 'HIL/circling/motor/TestCase_118_2300000117']


def save(path, value):
    with Path(path).open('x', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)


def choose():
    with MANIFEST.open(encoding='utf-8-sig', newline='') as stream:
        all_rows = list(csv.DictReader(stream))
    source = {r['case_uid']: r for r in all_rows if r['domain'] in ['SIL', 'HIL'] and r['split'] in ['train', 'val']}
    fit = []
    for domain in ['SIL', 'HIL']:
        for mode in ['hover', 'waypoint', 'circling', 'velocity', 'acce', 'dece']:
            candidates = [r for r in source.values() if r['domain'] == domain and r['split'] == 'train'
                          and r['flight_status'] == mode and r['source_fault_type'] == 'no_fault']
            fit.append(min(candidates, key=lambda r: r['case_uid']))
    validation = []
    for domain in ['SIL', 'HIL']:
        validation.append(min([r for r in source.values() if r['domain'] == domain and r['split'] == 'val'
            and r['source_fault_type'] == 'no_fault'], key=lambda r: r['case_uid']))
    validation.extend(source[uid] for uid in GAP_CASES)
    assert len({r['case_uid'] for r in fit + validation}) == 16
    return fit, validation


def integer_field(data, field):
    values = np.asarray(data[field])
    if values.ndim != 1 or not np.issubdtype(values.dtype, np.integer):
        raise ValueError('Missing/noninteger original clock: ' + field)
    return values.astype(np.int64)


def load_case(row, prior):
    uid = row['case_uid']; record = prior[uid]
    csvpath = Path(row['final_output_path']); raw = Path(row['source_path'])
    logs = list((raw / 'Log').glob('*.ulg'))
    if len(logs) != 1 or digest(logs[0]) != record['original_ulog_sha256'] or digest(csvpath) != record['processed_csv_sha256']:
        raise ValueError('Previously verified source identity changed')
    frame = pd.read_csv(csvpath, usecols=['Timestamp'])
    t = frame.Timestamp.to_numpy(dtype=float)
    if not np.isfinite(t).all() or not np.equal(t, np.rint(t)).all():
        raise ValueError('Original crop time not integer')
    grid = t.astype(np.int64)
    if len(grid) != int(row['row_count']) or not np.all(np.diff(grid) == 20000):
        raise ValueError('Original crop changed')
    topics = sorted(set(item[0] for item in GROUPS.values()))
    log = ULog(str(logs[0]), message_name_filter_list=topics)
    datasets = {}
    for topic in topics:
        found = [d.data for d in log.data_list if d.name == topic and d.multi_id == 0]
        if len(found) != 1: raise ValueError('Missing/ambiguous original topic: ' + topic)
        datasets[topic] = found[0]
    traces, timing = {}, {}
    for group, (topic, clock, fields) in GROUPS.items():
        d = datasets[topic]; pub = integer_field(d, 'timestamp')
        if (pub[1:] < pub[:-1]).any(): raise ValueError('Publication clock reversal: ' + uid + '/' + group)
        # Retain last already-published message before crop and all crop arrivals.
        start = max(int(np.searchsorted(pub, grid[0], side='right')) - 1, 0)
        end = int(np.searchsorted(pub, grid[-1], side='right'))
        avail = pub[start:end]
        if not len(avail): raise ValueError('No original topic coverage: ' + group)
        if clock == 'timestamp':
            obs = avail.copy()
        elif clock == 'accelerometer_timestamp_relative':
            relative = integer_field(d, clock)[start:end]
            if (relative == 2147483647).any():
                raise ValueError('Invalid accelerometer clock sentinel; no silent fallback')
            obs = avail + relative
        else:
            obs = integer_field(d, clock)[start:end]
        if (obs <= 0).any() or (obs > avail).any() or (obs[1:] < obs[:-1]).any():
            raise ValueError('Invalid/reversed observation clock: ' + uid + '/' + group)
        values = np.column_stack([d[field][start:end] for _, field in fields]).astype(np.float64)
        traces[group] = TopicTrace(obs, avail, values)
        inside = (avail >= grid[0]) & (avail <= grid[-1]); in_t = avail[inside]
        dt = np.diff(in_t); positive = dt[dt > 0]
        if not len(positive): raise ValueError('Insufficient timing observations')
        lag = avail[inside] - obs[inside]
        timing[group] = {'topic': topic, 'observation_clock': clock, 'rows': len(in_t),
            'median_period_us': float(np.median(positive)), 'p99_period_us': float(np.quantile(positive, .99)),
            'max_gap_us': int(positive.max()), 'median_source_lag_us': float(np.median(lag)),
            'max_source_lag_us': int(lag.max()), 'nonfinite_value_rows': int((~np.isfinite(values).all(axis=1)).sum())}
    return {'row': row, 'traces': traces, 'grid': grid, 'timing': timing,
        'identities': {str(csvpath): record['processed_csv_sha256'], str(logs[0]): record['original_ulog_sha256']}}


def run():
    if (OUT / 'input_policy.json').exists() or (OUT / 'summary.json').exists():
        raise FileExistsError('Never overwrite a frozen pilot run')
    plan = json.loads((OUT / 'plan.json').read_text(encoding='utf-8'))
    fit_rows, validation_rows = choose()
    save(OUT / 'selection.json', {'fit': [r['case_uid'] for r in fit_rows],
        'validation': [r['case_uid'] for r in validation_rows], 'manifest_sha256': digest(MANIFEST),
        'plan_sha256': digest(OUT / 'plan.json'), 'script_sha256': digest(__file__),
        'feature_contract_sha256': digest(FEATURE),
        'case_usage': 'timing/statistics fitted on source train anchors only; val/gap cases fixed before fitting, no final-test/Real/model access'})
    prior = {r['case_uid']: r for r in [json.loads(s) for s in PRIOR.read_text(encoding='utf-8').splitlines()]}
    fit = [load_case(r, prior) for r in fit_rows]
    features = json.loads(FEATURE.read_text(encoding='utf-8-sig'))
    columns = [item['canonical_name'] for item in features['features']]
    budgets, basis = {}, {}
    for group in GROUPS:
        period = max(c['timing'][group]['median_period_us'] for c in fit)
        lag = max(c['timing'][group]['median_source_lag_us'] for c in fit)
        limit = int(np.ceil((period + lag + 20000) / 20000) * 20000)
        budgets[group] = limit
        basis[group] = {'largest_train_case_median_period_us': period, 'largest_train_case_median_lag_us': lag,
                        'grid_allowance_us': 20000, 'ceil_grid_tick_us': 20000, 'max_age_us': limit}
    policy = {'version': 'causal_input_v1_source_publication_replay_pilot',
        'status': 'RULES_FROZEN_FOR_SMALL_SCALE_REPLAY_ONLY', 'columns': columns,
        'target_hz': 50, 'window_frames': 150, 'stride_frames': 5, 'max_age_us': budgets, 'budget_basis': basis,
        'clock': 'Same PX4 microsecond boot clock, ULog message timestamp as replay availability proxy. Not proven actual publish/write/transport receive time. timestamp_sample or IMU relative clock for age where provided, otherwise declared publication proxy.',
        'normalization': 'Causal valid cap64 source-fit windows only; same frozen diagnostic stats for both paths. Never old stats/val/test/Real fitted.',
        'selection': 'No future interpolation; last available message, final duplicate arrival; observation<=availability<=grid. Latest invalid/stale does not revert to older good record.',
        'invalid': 'Preserve all grid rows and original window starts, NaN and masks for missing/stale/nonfinite. No model on any input-invalid frame/window; labels separate.',
        'live_input_accepted': False, 'training_enabled': False, 'deployment_enabled': False,
        'limit_note': 'Engineering pilot one-update-plus-grid-and-lag budget, not a learned physical tolerance. Fixed before validation and never widened for coverage or model scores.'}
    save(OUT / 'input_policy.json', policy)
    save(OUT / 'fit_timing.json', {c['row']['case_uid']: c['timing'] for c in fit})
    fit_frames = {}; sums = np.zeros(26); squares = np.zeros(26); samples = 0; fit_counts = {}
    for case in fit:
        matrix, valid, provenance = batch_frames(case['traces'], case['grid'], columns, budgets)
        fit_frames[case['row']['case_uid']] = (matrix, valid, provenance)
        starts = np.arange(0, len(matrix) - 149, 5, dtype=np.int64)
        if len(starts) > 64: starts = starts[np.unique(np.linspace(0, len(starts) - 1, 64).round().astype(int))]
        valid_windows, windows = batch_windows(matrix, valid, starts, columns, mean=np.zeros(26), std=np.ones(26))
        fit_counts[case['row']['case_uid']] = {'original_cap64_windows': len(starts), 'valid_cap64_windows': int(valid_windows.sum())}
        for value in windows.values():
            x = value.T.astype(np.float64); sums += x.sum(axis=0); squares += (x*x).sum(axis=0); samples += len(x)
    if not samples: raise ValueError('No valid fit windows; do not widen policy to force success')
    mean = sums / samples; variance = np.maximum(squares / samples - mean*mean, 0); std = np.sqrt(variance)
    constant = std == 0; std[constant] = 1
    norm = {'columns': columns, 'mean': mean.tolist(), 'std': std.tolist(), 'frames_counted': samples,
        'window_weighting': 'all frames in eligible unchanged cap64 source fit windows, overlap intentionally counted',
        'zero_variance_unit_scale_columns': [columns[i] for i in np.flatnonzero(constant)], 'fit_cases': fit_counts,
        'purpose': 'Diagnostic parity only; not production fitted statistics', 'input_policy_sha256': digest(OUT / 'input_policy.json')}
    save(OUT / 'pilot_norm_stats.json', norm)
    validation = [load_case(r, prior) for r in validation_rows]
    results = []
    with (OUT / 'case_results.jsonl').open('x', encoding='utf-8') as stream:
        for case in fit + validation:
            uid = case['row']['case_uid']; grid = case['grid']; traces = case['traces']
            matrix, valid, provenance = fit_frames.get(uid, (None, None, None))
            if matrix is None: matrix, valid, provenance = batch_frames(traces, grid, columns, budgets)
            starts = np.arange(0, len(grid) - 149, 5, dtype=np.int64)
            wvalid, windows = batch_windows(matrix, valid, starts, columns, mean=mean, std=std)
            runner = CausalInputStream(columns, budgets, origin_us=int(grid[0]), mean=mean, std=std)
            events = []
            for order, (group, trace) in enumerate(traces.items()):
                for i, avail in enumerate(trace.available_us): events.append((int(avail), order, i, group))
            events.sort()  # Cross-topic merge only; each topic was validated monotonic before merge.
            cursor = 0; stream_valid = []; stream_matrix = []; matched_windows = 0
            for i, tick in enumerate(grid):
                while cursor < len(events) and events[cursor][0] <= tick:
                    available, _, j, group = events[cursor]; trace = traces[group]
                    runner.push(group, int(trace.observation_us[j]), available, trace.values[j]); cursor += 1
                emitted = runner.emit(int(tick)); stream_valid.append(emitted['input_valid']); stream_matrix.append(emitted['values'])
                if emitted['input_valid'] != bool(valid[i]): raise AssertionError('Frame mask mismatch')
                if not np.array_equal(emitted['values'], matrix[i], equal_nan=True): raise AssertionError('Canonical row mismatch')
                for group, p in provenance.items():
                    for key in ['source_index', 'observation_us', 'available_us', 'age_us', 'reason', 'valid']:
                        if emitted['provenance'][group][key] != p[key][i]: raise AssertionError('Provenance mismatch: '+group+'/'+key)
                if emitted['window_due']:
                    start = emitted['window_start_row']; j = start // 5
                    if emitted['window_input_valid'] != bool(wvalid[j]): raise AssertionError('Window eligibility mismatch')
                    if emitted['window_input_valid']:
                        if not np.array_equal(emitted['window'], windows[start]): raise AssertionError('Normalized input tensor mismatch')
                        matched_windows += 1
            if matched_windows != len(windows): raise AssertionError('Missing parity windows')
            # Future truncation check over the original gap, including source IDs
            # and normalized input prefix, not merely equal values from one source.
            cutoff = 149508661 if uid == GAP_CASES[1] else (891885220 if uid == GAP_CASES[0] else int(grid[len(grid)//2]))
            prefix_grid = grid[grid <= cutoff]
            prefix = {k: TopicTrace(t.observation_us[t.available_us<=cutoff], t.available_us[t.available_us<=cutoff], t.values[t.available_us<=cutoff]) for k,t in traces.items()}
            pm, pv, pp = batch_frames(prefix, prefix_grid, columns, budgets)
            assert np.array_equal(pm, matrix[:len(prefix_grid)], equal_nan=True) and np.array_equal(pv, valid[:len(prefix_grid)])
            for group in GROUPS:
                assert np.array_equal(pp[group]['source_index'], provenance[group]['source_index'][:len(prefix_grid)])
            group_counts = {g: {'invalid_frames': int((~p['valid']).sum()), 'missing': int(((p['reason']&1)>0).sum()),
                'stale': int(((p['reason']&2)>0).sum()), 'nonfinite': int(((p['reason']&4)>0).sum())} for g,p in provenance.items()}
            gap_result = None
            if uid in GAP_CASES:
                ix = np.flatnonzero(grid == cutoff); assert len(ix) == 1; ix = int(ix[0]); p = provenance['gyro']
                assert not p['valid'][ix] and (int(p['reason'][ix]) & STALE)
                gap_result = {'endpoint_us': cutoff, 'selected_observation_us': int(p['observation_us'][ix]),
                    'age_us': int(p['age_us'][ix]), 'limit_us': budgets['gyro'], 'reason': 'stale',
                    'row_retained_invalid': True, 'uses_future_observation': False}
            filename = uid.replace('/', '__') + '.npz'
            np.savez_compressed(OUT / filename, grid_us=grid, canonical_rows=matrix, row_input_valid=valid,
                window_start_row=starts, window_input_valid=wvalid)
            record = {'case_uid': uid, 'split': case['row']['split'], 'phase': 'fit' if uid in fit_frames else 'validation',
                'rows': len(grid), 'valid_rows': int(valid.sum()), 'original_stride5_windows': len(starts),
                'valid_windows': int(wvalid.sum()), 'ineligible_windows_retained': int((~wvalid).sum()),
                'normalized_windows_exact_equal': matched_windows, 'all_source_indices_clocks_masks_rows_equal': True,
                'future_prefix_invariance': True, 'group_invalid_counts': group_counts, 'gap_endpoint': gap_result,
                'identities': case['identities'], 'output_file': filename, 'output_sha256': digest(OUT / filename)}
            results.append(record); stream.write(json.dumps(record, ensure_ascii=False) + chr(10)); stream.flush()
            print('CAUSAL_BATCH_STREAM_PARITY', uid, int(wvalid.sum()), '/', len(starts), flush=True)
    # Check only intended original inputs and source identity, never read a test cache.
    for case in fit + validation:
        for path, expected in case['identities'].items():
            if digest(path) != expected: raise ValueError('Original input changed during pilot')
    summary = {'status': 'PASS_SMALL_SCALE_CAUSAL_REPLAY_INPUT_PARITY_NOT_LIVE_CLEARANCE',
        'completed_at': datetime.now(timezone.utc).isoformat(), 'fit_cases': len(fit), 'validation_cases': len(validation),
        'all_canonical_rows': sum(r['rows'] for r in results), 'all_original_windows': sum(r['original_stride5_windows'] for r in results),
        'valid_normalized_windows_exact_equal': sum(r['valid_windows'] for r in results),
        'ineligible_windows_retained': sum(r['ineligible_windows_retained'] for r in results),
        'source_indices_timestamps_ages_reasons_masks_equal': True, 'all_future_prefix_checks_passed': True,
        'gap_examples_rejected_as_stale_without_deleting_rows': 2, 'max_age_us': budgets,
        'script_sha256': digest(__file__), 'pipeline_sha256': digest(ROOT / 'packages/uav_fault/uav_fault/causal_pipeline.py'),
        'policy_sha256': digest(OUT / 'input_policy.json'), 'normalization_sha256': digest(OUT / 'pilot_norm_stats.json'),
        'live_transport_tested': False, 'old_runtime_service_modified': False, 'training_run': False,
        'model_forward_or_scores_used': False, 'original_dataset_or_cache_rebuilt': False,
        'remaining': 'Live per-topic observation/receive-clock provenance and adapter integration not validated; pilot budgets/stats not deployment or full-dataset training approval.'}
    save(OUT / 'summary.json', summary); print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    run()
