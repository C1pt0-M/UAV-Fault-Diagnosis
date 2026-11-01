"""Source-only M1 sidecars; continuous200ms native history, original150 windows.

Independent output, resumable verified case receipts, no old data mutation.
"""
from __future__ import annotations
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
import time

os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'packages/uav_fault'))
sys.path.insert(0, str(ROOT / 'scripts/research'))
sys.path.insert(0, str(ROOT / 'scripts/data'))
import numpy as np
import pandas as pd
from pyulog import ULog
from uav_fault.causal_pipeline import GROUPS, batch_frames, TopicTrace
from build_full_causal_input import extract_group
from continuous_kinematic_residuals import (Trace, compute_continuous, continuous_window_masks)
from validate_kinematic_residuals import sha, PREFIX, TOPIC_INVALID, INPUT_INVALID, RESET

TOPICS = {'baro': 'vehicle_air_data', 'velocity': 'vehicle_local_position',
    'attitude': 'vehicle_attitude', 'gyro': 'sensor_combined'}
MARKERS = {'baro': ['baro_device_id'], 'velocity': ['z_reset_counter', 'vz_reset_counter'],
    'attitude': ['quat_reset_counter'], 'gyro': []}
WIDTHS = {'baro': ['baro_alt_relative'], 'velocity': ['vel_z'],
    'attitude': ['att_q_' + k for k in 'wxyz'], 'gyro': ['gyro_' + k for k in 'xyz']}


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def save(path, record):
    with path.open('x', encoding='utf-8') as stream:
        json.dump(record, stream, ensure_ascii=False, indent=2, allow_nan=False)


def load_native(record, payload, policy):
    """Read200ms before original crop; leave original crop/input population intact."""
    grid = payload['timestamp_us']
    ext_grid = np.r_[grid[0] - np.arange(10, 0, -1, dtype=np.int64) * 20000, grid]
    log = ULog(record['raw_ulog'], message_name_filter_list=sorted({topic for topic, _, _ in GROUPS.values()}))
    native, offsets, errors, raw_data = {}, {}, {}, {}
    for group, (topic, _, fields) in GROUPS.items():
        try:
            found = [d.data for d in log.data_list if d.name == topic and d.multi_id == 0]
            if len(found) != 1:
                raise ValueError('missing_or_ambiguous_native_topic:' + topic)
            raw_data[group] = found[0]
            native[group], offsets[group] = extract_group(found[0], group, ext_grid)
        except (ValueError, KeyError, IndexError) as exc:
            errors[group] = str(exc)
            native[group] = TopicTrace(np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64), np.empty((0, len(fields))))
            offsets[group] = 0
    # Only reconstruct the extra10 base frames; do not rebuild/replace26 data.
    _, pre_valid, _ = batch_frames(native, ext_grid[:10], policy['columns'], policy['max_age_us'])
    base_valid = np.r_[pre_valid, payload['row_input_valid']]
    traces, arrays, counter_info = {}, {}, {}
    for group in TOPICS:
        if group in errors or len(native[group].available_us) == 0:
            raise ValueError('Unavailable required native support:' + group + ':' + errors.get(group, 'empty'))
        trace, raw = native[group], raw_data[group]
        offset = offsets[group]; n = len(trace.available_us)
        vals = trace.values[:, 2:3] if group == 'velocity' else trace.values
        counters = {name: raw[name][offset:offset + n].copy() for name in MARKERS[group] if name in raw}
        valid = np.ones(n, dtype=bool)
        if group == 'velocity' and 'v_z_valid' in raw:
            valid &= raw['v_z_valid'][offset:offset + n] == 1
        traces[group] = Trace(trace.observation_us, trace.available_us, vals, int(policy['max_age_us'][group]), counters, valid)
        idx = np.searchsorted(trace.available_us, grid, side='right') - 1
        for field, source in [('observation_us', trace.observation_us), ('available_us', trace.available_us)]:
            # Source base-valid positions must exactly reuse the prior provenance.
            compare = payload[group + '__valid']
            if (idx[compare] < 0).any() or not np.array_equal(source[idx[compare]], payload[group + '__' + field][compare]):
                raise RuntimeError('Native provenance differs from frozen base:' + group)
        good = payload[group + '__valid']; column_ids = [policy['columns'].index(k) for k in WIDTHS[group]]
        if not np.array_equal(vals[idx[good]], payload['canonical_rows'][good][:, column_ids]):
            raise RuntimeError('Native values differ from frozen base:' + group)
        arrays[group + '__native_observation_us'] = trace.observation_us
        arrays[group + '__native_available_us'] = trace.available_us
        arrays[group + '__native_values'] = vals
        arrays[group + '__native_valid'] = valid
        arrays[group + '__native_source_index'] = np.arange(offset, offset + n, dtype=np.int64)
        for name, values in counters.items():
            arrays[group + '__native_' + name] = values
        counter_info[group] = {'available_fields': list(counters),
            'missing_fields': [name for name in MARKERS[group] if name not in counters],
            'changes': {name: int(np.count_nonzero(np.diff(v.astype(np.int64)))) for name, v in counters.items()}}
    return ext_grid, base_valid, traces, arrays, counter_info, errors


def process(item):
    task, plan, out = item
    out = Path(out); uid = task['case_uid']; key = uid.replace('/', '__')
    receipt = out / 'case_records' / (key + '.json'); dest = out / 'cases' / (key + '.npz')
    identity = hashlib.sha256(json.dumps(task, sort_keys=True).encode()).hexdigest()
    if receipt.exists():
        old = read(receipt)
        if old['task_identity'] != identity or sha(dest) != old['sidecar_sha256']:
            raise ValueError('Resume identity mismatch:' + uid)
        return old
    if dest.exists():
        raise FileExistsError('Unreceipted output, inspect manually:' + str(dest))
    started = time.perf_counter()
    for name, expected in [(task['record_file'], task['record_sha256']), (task['case_file'], task['case_sha256'])]:
        if sha(name) != expected:
            raise ValueError('Frozen source identity mismatch:' + name)
    record = read(task['record_file']); policy = read(Path(plan['source_root']) / 'input_contract.json')
    with np.load(task['case_file'], allow_pickle=False) as data:
        payload = {name: data[name] for name in data.files}
    n = len(payload['timestamp_us']); arrays = {}
    counters, errors, failure = {}, {}, None
    if task['originally_admitted'] > 0:
        if sha(task['raw_ulog']) != task['raw_sha256']:
            raise ValueError('Original ULog identity mismatch:' + uid)
        rawstat = (Path(task['raw_ulog']).stat().st_size, Path(task['raw_ulog']).stat().st_mtime_ns)
        try:
            ext, base_valid, traces, native, counters, errors = load_native(record, payload, policy)
            residual = compute_continuous(ext.astype(np.int64), base_valid.astype(bool), traces)
            values = np.column_stack([residual['vertical_m'][10:], residual['attitude_rad'][10:]])
            reason = residual['reason'][10:]; valid = np.column_stack([residual['vertical_valid'][10:], np.repeat(residual['attitude_valid'][10:, None], 3, axis=1)])
            arrays.update(native); arrays['endpoint_age_us'] = residual['endpoint_age_us'][10:]
            arrays['prehistory_timestamp_us'] = ext[:10]; arrays['prehistory_input_valid'] = base_valid[:10]
        except (ValueError, KeyError, IndexError) as exc:
            # Preserve explicit native support failure; never fabricate residuals.
            failure = type(exc).__name__ + ':' + str(exc)
            values = np.full((n, 4), np.nan); valid = np.zeros((n, 4), dtype=bool); reason = np.full((n, 2), TOPIC_INVALID, dtype=np.uint16)
        if rawstat != (Path(task['raw_ulog']).stat().st_size, Path(task['raw_ulog']).stat().st_mtime_ns):
            raise ValueError('Original ULog changed while read:' + uid)
    else:
        values = np.full((n, 4), np.nan); valid = np.zeros((n, 4), dtype=bool); reason = np.full((n, 2), INPUT_INVALID, dtype=np.uint16)
        failure = 'original_case_has_no_admitted_windows_not_rescued'
    assert values.shape == (n, 4) and np.isfinite(values[valid]).all()
    joint = valid.all(axis=1); starts = payload['window_start_row']; ids = payload['original_cache_index']
    masks = continuous_window_masks(joint, starts)
    included = payload['supervised_valid'] & masks.all(axis=1)
    whole_normal = np.array([bool(payload['row_label_valid'][s:s+150].all() and (payload['reference_state'][s:s+150] == 0).all()) for s in starts])
    arrays.update(timestamp_us=payload['timestamp_us'], residual=values, valid=valid, reason=reason,
        original_row_input_valid=payload['row_input_valid'], original_row_label_valid=payload['row_label_valid'],
        original_reference_state=payload['reference_state'], window_start_row=starts,
        window_end_row=payload['window_end_row'], original_cache_index=ids,
        original_supervised_valid=payload['supervised_valid'],
        original_reference_y=payload['reference_y'], window_residual_valid=masks,
        eligible=included, whole_normal=whole_normal)
    partial = dest.with_suffix('.partial.npz')
    np.savez_compressed(partial, **arrays); partial.replace(dest)
    excluded = payload['supervised_valid'] & ~included
    reason_counts = {str(flag): int(np.count_nonzero((reason & flag) != 0)) for flag in [PREFIX, INPUT_INVALID, TOPIC_INVALID, RESET, 16]}
    r = {'case_uid': uid, 'split': task['split'], 'task_identity': identity,
        'source_record_sha256': task['record_sha256'], 'source_case_sha256': task['case_sha256'],
        'raw_ulog_sha256': task['raw_sha256'], 'sidecar_file': dest.name, 'sidecar_sha256': sha(dest),
        'rows': n, 'original_windows': len(ids), 'originally_admitted_windows': int(payload['supervised_valid'].sum()),
        'eligible_windows': int(included.sum()), 'newly_excluded_windows': int(excluded.sum()),
        'train_normal_fit_windows': int((included & whole_normal).sum()) if task['split'] == 'train' else 0,
        'reason_bit_occurrences_nonexclusive': reason_counts, 'native_counter_info': counters, 'native_errors': errors,
        'whole_case_failure': failure, 'elapsed_s': time.perf_counter() - started}
    save(receipt, r)
    return r


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--limit', type=int)
    args = parser.parse_args(); plan = read(args.plan); out = Path(plan['output_root'])
    for name, expected in plan['source_code_sha256'].items():
        if sha(name) != expected:raise ValueError('Pinned source changed:' + name)
    for path, expected in plan['source_data_identities'].items():
        if path not in ['cache/train.pt', 'cache/val.pt'] and sha(Path(plan['source_root']) / path) != expected:
            raise ValueError('Pinned metadata changed:' + path)
    out.mkdir(exist_ok=True)
    for name in ['cases', 'case_records', 'ledgers', 'cache']:(out / name).mkdir(exist_ok=True)
    tasks = plan['tasks'][:args.limit] if args.limit else plan['tasks']
    save_name = out / 'generation_plan.json'
    if save_name.exists():
        if sha(save_name) != sha(args.plan):raise ValueError('Different dataset plan already present')
    else:save_name.write_bytes(args.plan.read_bytes())
    status = out / 'generation_status.json'
    def update(record):
        tmp = status.with_suffix('.partial.json');tmp.write_text(json.dumps(record,ensure_ascii=False,indent=2),encoding='utf-8')
        # Windows readers briefly hold files without delete-sharing. Retry only
        # status publication; never recompute/alter data to recover this lock.
        for attempt in range(100):
            try:
                tmp.replace(status);break
            except PermissionError:
                if attempt==99:raise
                time.sleep(.05)
    started = time.perf_counter(); completed = []; update({'status':'RUNNING_SOURCE_ONLY_CONTINUOUS_FEATURES','cases_total':len(tasks),'cases_complete':0,'workers':args.workers})
    try:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(process, (task, plan, str(out))):task for task in tasks}
            for future in as_completed(futures):
                r=future.result();completed.append(r)
                info={'status':'RUNNING_SOURCE_ONLY_CONTINUOUS_FEATURES','cases_total':len(tasks),'cases_complete':len(completed),
                    'original_admitted_complete':sum(x['originally_admitted_windows'] for x in completed),
                    'M1_eligible_complete':sum(x['eligible_windows'] for x in completed),'newly_excluded_complete':sum(x['newly_excluded_windows'] for x in completed),
                    'last_case':r['case_uid'],'elapsed_s':time.perf_counter()-started}
                update(info)
                if len(completed)%10==0:print(json.dumps(info),flush=True)
        update(dict(info,status='COMPLETE_CASE_FEATURES_AWAITING_DATASET_FINALIZATION'))
        print('COMPLETE_CASE_FEATURES',len(completed),time.perf_counter()-started,flush=True)
    except Exception as exc:
        update({'status':'FAILED_STOPPED_NO_QUALITY_RELAXATION','cases_complete':len(completed),'error':repr(exc)})
        raise


if __name__ == '__main__':
    main()
