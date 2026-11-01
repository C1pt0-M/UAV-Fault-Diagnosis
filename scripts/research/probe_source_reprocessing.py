"""Candidate source-only reprocessing probe; NOT an approved training builder.

Uses existing 26-channel contract, original ULog publication timestamps and GTD.
Writes diagnostic NPZs in a fresh artifacts directory; never changes source data.
No model inference, fitting, normalization, Real or final-test access.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import sys

import numpy as np
import pandas as pd
from pyulog import ULog

from audit_source_event_boundaries import read_gtd, digest, source_tasks, hold
from audit_rflymad_raw_sources import active_ids

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts/data'))
from rfly_feature_contract import contract_columns, window_batch

CONTRACT = Path('D:/UAV-Data/feature_contract_v1.json')
CONTRACT_SHA = 'bdc39e51d728203bb3053d3ea94c2afc844e81022242b7856bc59eb80ff564fe'
TOPICS = ['actuator_controls_0', 'actuator_outputs', 'sensor_combined',
          'vehicle_air_data', 'vehicle_attitude', 'vehicle_local_position',
          'vehicle_magnetometer']


def coalesce(times, values):
    raw_times = np.asarray(times)
    if not np.isfinite(raw_times).all() or not np.equal(raw_times, np.round(raw_times)).all():
        raise ValueError('Nonfinite or fractional publication timestamp')
    times = raw_times.astype(np.int64)
    values = np.asarray(values)
    if times.ndim != 1 or not len(times) or values.shape[0] != len(times):
        raise ValueError('Empty or mismatched original topic')
    if np.any(np.diff(times) < 0):
        raise ValueError('Original topic timestamp reversal; no implicit sorting')
    keep = np.r_[times[1:] != times[:-1], True]
    return times[keep], values[keep]


def sample_published(times, values, query, age_cap_us):
    times, values = coalesce(times, values)
    if values.ndim == 1:
        values = values[:, None]
    indices = np.searchsorted(times, query, side='right') - 1
    picked = np.maximum(indices, 0)
    age = query - times[picked]
    out = values[picked].astype(float)
    valid = ((indices >= 0) & (query <= times[-1]) & (age >= 0)
             & (age <= age_cap_us) & np.isfinite(out).all(axis=1))
    out[~valid] = np.nan
    return out, valid, age


def flight_bounds(times, landed, contact, settle_us=2000000, age_cap_us=2000000):
    times, states = coalesce(times, np.column_stack([landed, contact]))
    if not np.isin(states, [0, 1]).all():
        raise ValueError('Nonbinary land detector flags')
    airborne = (states[:, 0] == 0) & (states[:, 1] == 0)
    hits = np.flatnonzero(airborne)
    if not len(hits):
        raise ValueError('No observed airborne state')
    first = int(hits[0])
    left_censored = first == 0
    # Earliest observed airborne episode, independent of fault identity/time.
    landing_start = None
    end = int(times[-1])
    confirmed = False
    for i in range(first + 1, len(times)):
        if times[i] - times[i - 1] > age_cap_us:
            landing_start = None
        if states[i, 0] == 1:
            if landing_start is None:
                landing_start = i
            if times[i] - times[landing_start] >= settle_us:
                end = int(times[landing_start])
                confirmed = True
                break
        else:
            landing_start = None
    if end <= times[first]:
        raise ValueError('Empty airborne interval')
    later_airborne = bool(confirmed and np.any(airborne & (times > end)))
    return {'start_us': int(times[first]), 'end_us': end,
            'left_censored': left_censored, 'right_censored': not confirmed,
            'landing_confirmed': confirmed, 'later_airborne_observed': later_airborne}


def feature_sources(contract, domain):
    result = []
    for item in contract['features']:
        if item['canonical_name'] == 'acc_norm':
            result.append(None)
            continue
        source = item['source_columns'][domain]
        match = None
        for topic in TOPICS:
            prefix = '_' + topic + '_0_'
            if source.startswith(prefix):
                match = (topic, source[len(prefix):])
                break
        if match is None:
            raise ValueError('Unmapped raw source: ' + source)
        result.append(match)
    return result


def choose_tasks():
    tasks = source_tasks()
    by = {t[0]['case_uid']: t for t in tasks}
    review = ROOT / 'artifacts/source_event_boundaries_review_20261031/case_review_inventory.csv'
    with review.open(encoding='utf-8-sig', newline='') as f:
        rows = list(csv.DictReader(f))
    selected = {r['case_uid'] for r in rows
                if r['raw_onset_after_existing_crop'] == 'True' or r['inertial_clock_review_required'] == 'True'}
    strata = defaultdict(list)
    normals = defaultdict(list)
    for task in tasks:
        row = task[0]
        strata[(row['domain'], row['flight_status'], row['source_fault_type'])].append(row['case_uid'])
        if row['source_fault_type'] == 'no_fault':
            normals[(row['domain'], row['flight_status'])].append(row['case_uid'])
    for values in strata.values():
        selected.add(sorted(values)[0])
    for values in normals.values():
        selected.add(sorted(values)[-1])
    return [by[uid] for uid in sorted(selected)]


def probe_case(task, contract, out, policy):
    row, old, _ = task
    result = {k: row[k] for k in ['case_uid', 'domain', 'split', 'flight_status', 'source_fault_type']}
    result.update(status='review_required', issues=[])
    try:
        raw = Path(row['source_path'])
        logs = list((raw / 'Log').glob('*.ulg'))
        if len(logs) != 1:
            raise ValueError('Expected one original ULog')
        logpath = logs[0]
        gtdpath = raw / 'TrueData/UAVState_data.xlsx'
        paths = [logpath, gtdpath, raw / 'TestInfo.csv', Path(row['final_output_path'])]
        before = {str(p): (p.stat().st_size, p.stat().st_mtime_ns) for p in paths}
        if digest(paths[2]) != old['testinfo_sha256'] or digest(paths[3]) != old['csv_sha256']:
            raise ValueError('Prior metadata/processed identity changed')
        old_times = pd.read_csv(paths[3], usecols=['Timestamp']).Timestamp.to_numpy(dtype=float) / 1e6
        if not len(old_times) or not np.isfinite(old_times).all() or (np.diff(old_times) <= 0).any():
            raise ValueError('Invalid existing processed timeline')
        log = ULog(str(logpath), message_name_filter_list=TOPICS + ['vehicle_land_detected', 'rfly_ctrl_lxl'])
        datasets = {}
        for name in TOPICS + ['vehicle_land_detected', 'rfly_ctrl_lxl']:
            found = [d.data for d in log.data_list if d.name == name and d.multi_id == 0]
            if len(found) != 1:
                raise ValueError('Missing/ambiguous original topic: ' + name)
            datasets[name] = found[0]
        land = datasets['vehicle_land_detected']
        bounds = flight_bounds(land['timestamp'], land['landed'], land['ground_contact'],
                               policy['landing_confirmation_us'], policy['land_max_age_us'])
        query = np.arange(bounds['start_us'], bounds['end_us'], policy['grid_step_us'], dtype=np.int64)
        if not len(query):
            raise ValueError('Empty candidate grid')
        columns = contract_columns(contract)
        sources = feature_sources(contract, row['domain'])
        matrix = np.full((len(query), len(columns)), np.nan)
        topic_valid = np.ones(len(query), dtype=bool)
        per_topic = {}
        for name in TOPICS:
            mapping = [(i, field) for i, pair in enumerate(sources) if pair is not None
                       for topic, field in [pair] if topic == name]
            data = datasets[name]
            for _, field in mapping:
                if field not in data:
                    raise ValueError('Missing original field: ' + name + '/' + field)
            values = np.column_stack([data[field] for _, field in mapping])
            sampled, valid, age = sample_published(data['timestamp'], values, query, policy['feature_max_age_us'])
            matrix[:, [i for i, _ in mapping]] = sampled
            topic_valid &= valid
            per_topic[name] = {'source_rows': len(data['timestamp']), 'invalid_candidate_frames': int((~valid).sum()),
                               'max_selected_age_us': int(age[valid].max()) if valid.any() else None}
        acc = matrix[:, [columns.index(n) for n in ['accel_x', 'accel_y', 'accel_z']]]
        matrix[:, columns.index('acc_norm')] = np.sqrt(np.sum(acc * acc, axis=1))
        land_sampled, land_valid, _ = sample_published(
            land['timestamp'], np.column_stack([land['landed'], land['ground_contact']]),
            query, policy['land_max_age_us'])
        airborne = land_valid & (land_sampled[:, 0] == 0) & (land_sampled[:, 1] == 0)
        gt, flags = read_gtd(gtdpath)
        reference, coverage = hold(gt, flags, query / 1e6)
        command = datasets['rfly_ctrl_lxl']
        command_reference, command_covered = hold(command['timestamp'] / 1e6, active_ids(command['id']), query / 1e6)
        full_valid = topic_valid & airborne & coverage & np.isfinite(matrix).all(axis=1)
        starts = np.arange(0, max(0, len(query) - 150 + 1), 5, dtype=int)
        invalid_prefix = np.r_[0, np.cumsum(~full_valid)]
        window_valid = invalid_prefix[starts + 150] == invalid_prefix[starts]
        spans = [(int(s), int(s + 150)) for s in starts[window_valid]]
        # Reuse existing relative-altitude transformation, with no zero filling.
        windows = window_batch(matrix, columns, spans)
        if not np.isfinite(windows).all():
            raise ValueError('Nonfinite value passed candidate window validity gate')
        result.update(status='candidate_probe_complete', bounds=bounds,
                      candidate_frames=len(query), input_and_reference_valid_frames=int(full_valid.sum()),
                      candidate_windows=len(starts), input_and_reference_valid_windows=len(spans),
                      valid_window_recorded_active_endpoints=int(sum(reference[e - 1] for _, e in spans)),
                      valid_window_unactivated_endpoints=int(sum(not reference[e - 1] for _, e in spans)),
                      old_crop_start_s=float(old_times[0]), old_crop_end_s=float(old_times[-1]),
                      topic_diagnostics=per_topic, raw_gtd_sha256=digest(gtdpath), ulog_sha256=digest(logpath),
                      label_reference_disagreements=int(((reference != command_reference) & coverage & command_covered).sum()),
                      first_gtd_activation_s=float(gt[np.flatnonzero(flags)[0]]) if flags.any() else None,
                      candidate_end_s=bounds['end_us'] / 1e6, source_stat_unchanged=True)
        if any(before[str(p)] != (p.stat().st_size, p.stat().st_mtime_ns) for p in paths):
            raise ValueError('Source changed during read')
        # Raw-altitude column is intentionally NOT yet window-relative here.
        # Invalid rows remain NaN with explicit masks; this is not a training cache.
        np.savez_compressed(out / (str(result['case_uid']).replace('/', '__') + '.npz'),
                            timestamp_us=query, raw_feature_values=matrix, columns=np.array(columns),
                            input_reference_valid=full_valid, gtd_reference=reference, gtd_covered=coverage,
                            command_reference=command_reference, command_covered=command_covered,
                            candidate_window_starts=starts, candidate_window_valid=window_valid)
    except Exception as exc:
        result['status'] = 'review_required'
        result['issues'].append(type(exc).__name__ + ': ' + str(exc))
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output', type=Path, required=True)
    args = ap.parse_args()
    out = args.output.resolve()
    for root in [Path('D:/RflyMAD').resolve(), Path('D:/UAV-Data').resolve()]:
        if out == root or root in out.parents:
            raise ValueError('Never write under raw/processed dataset roots')
    if digest(CONTRACT) != CONTRACT_SHA:
        raise ValueError('Existing feature contract changed')
    contract = json.loads(CONTRACT.read_text(encoding='utf-8'))
    if len(contract_columns(contract)) != 26:
        raise ValueError('Expected existing 26-channel contract')
    tasks = choose_tasks()
    policy = {'status': 'candidate_development_only_not_training_protocol',
              'grid_step_us': 20000, 'feature_max_age_us': 200000,
              'land_max_age_us': 2000000, 'landing_confirmation_us': 2000000,
              'notes': ['0.2s per-input freshness and 2s landing confirmation are provisional declared engineering limits, not validated physical thresholds.',
                        'Use original publication timestamp, last duplicate, no future samples, no extrapolation past last record and no interpolation.',
                        'Causal relative to recorded publication clock only; physical delivery latency and actual online transport not certified.',
                        'Earliest airborne episode from land/contact flags; no fault labels or low-throttle termination used.',
                        'Right/left censoring and later observed airborne state reported, not silently treated as a fully verified flight.',
                        'NPZ has raw baro altitude despite canonical column name; window-relative transform is tested in memory before normalization. No training ready x/y/cache produced.']}
    out.mkdir(parents=True, exist_ok=False)
    (out / 'samples').mkdir()
    plan = {'created_at': datetime.now(timezone.utc).isoformat(), 'policy': policy,
            'selection': 'All 40 after-crop activation cases, all 3 inertial-clock exceptions, first source case in each domain/mode/subtype and last normal case per domain/mode.',
            'cases': [t[0]['case_uid'] for t in tasks], 'contract_sha256': CONTRACT_SHA,
            'script_sha256': digest(__file__), 'scope': 'Source train/val only; diagnostic candidate outputs, no final tests, Real, models, fitting or source overwrite.'}
    (out / 'plan.json').write_text(json.dumps(plan, ensure_ascii=False, indent=2), encoding='utf-8')
    records = []
    with (out / 'case_results.jsonl').open('x', encoding='utf-8') as f:
        for i, task in enumerate(tasks, 1):
            record = probe_case(task, contract, out / 'samples', policy)
            records.append(record)
            f.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + chr(10)); f.flush()
            if i % 20 == 0:
                print('PROBE', i, '/', len(tasks), flush=True)
    summary = {'cases': len(records), 'status_counts': dict(Counter(r['status'] for r in records)),
               'issues': dict(Counter(s for r in records for s in r['issues'])),
               'candidate_frames': sum(r.get('candidate_frames', 0) for r in records),
               'input_reference_valid_frames': sum(r.get('input_and_reference_valid_frames', 0) for r in records),
               'candidate_windows': sum(r.get('candidate_windows', 0) for r in records),
               'input_reference_valid_windows': sum(r.get('input_and_reference_valid_windows', 0) for r in records),
               'scope': plan['scope'], 'policy_status': policy['status']}
    (out / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == '__main__':
    main()
