"""Bounded CPU normal-stage stability and source fault response diagnostics.

Reuse fixed residual computation; no classifier, GPU, thresholds, test/Real or
source mutation. Larger-score AUCs are descriptive, with no sign optimization.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
import sys
import time
from types import SimpleNamespace

import numpy as np
import pandas as pd
from scipy.stats import rankdata
import torch

from validate_kinematic_residuals import (
    ROOT, Trace, compute_residuals, window_residual_mask, summarize, sha, read, save,
)
sys.path.insert(0, str(ROOT / 'packages/uav_fault'))
sys.path.insert(0, str(ROOT / 'scripts/data'))
from uav_fault.causal_input import canonical_window
from uav_fault.models.cnn.lstm import OnlineKGMSTCA

NEW = ['rh_signed_m', 'rR_x_rad', 'rR_y_rad', 'rR_z_rad', 'rh_abs_m', 'rR_norm_rad']
OLD = ['old_output_std', 'old_output_range', 'old_ctrl_output_gap_mean', 'old_ctrl_output_gap_std',
    'old_effort', 'old_acc_deviation', 'old_gyro_norm', 'old_vertical_consistency', 'old_attitude_energy']
FEATURES = NEW + OLD


def legacy_features(normalized, columns):
    """Execute the existing deterministic9-channel function, no model instance."""
    index = {name: i for i, name in enumerate(columns)}
    context = SimpleNamespace(_ctrl_idx=[index[f'actuator_ctrl_{i}'] for i in range(1, 5)],
        _out_idx=[index[f'actuator_out_{i}'] for i in range(1, 5)],
        _gyro_idx=[index['gyro_' + axis] for axis in 'xyz'],
        _quat_idx=[index['att_q_' + axis] for axis in 'xyz'],
        _acc_norm_idx=index['acc_norm'], _vel_z_idx=index['vel_z'],
        _alt_idx=index['baro_alt_relative'], _effort_idx=None)
    values = np.asarray(normalized)
    if values.dtype != np.float32 or values.shape != (26, 150) or not np.isfinite(values).all():
        raise ValueError('Exact finite normalized source window required')
    with torch.inference_mode():
        result = OnlineKGMSTCA._relation_features(context, torch.from_numpy(values).unsqueeze(0))
    result = result[0].numpy().astype(np.float64)
    if result.shape != (9, 150) or not np.isfinite(result).all():
        raise ValueError('Nonfinite existing9-channel heuristic features')
    return result


def fixed_auc(normal, fault):
    normal, fault = np.asarray(normal, dtype=float), np.asarray(fault, dtype=float)
    if not len(normal) or not len(fault):
        return None
    if not np.isfinite(normal).all() or not np.isfinite(fault).all():
        raise ValueError('Only explicitly finite admitted feature samples allowed')
    values = np.r_[normal, fault]
    ranks = rankdata(values, method='average')
    n0, n1 = len(normal), len(fault)
    return float((ranks[n0:].sum() - n1 * (n1 + 1) / 2) / (n0 * n1))


def quarter(end, rows):
    if not np.isfinite(end) or int(end) != end or not 150 <= end <= rows or rows < 150:
        raise ValueError('Original crop window-end position required')
    return min(3, (4 * (int(end) - 1)) // (rows - 1))


def whole_window_group(state, valid, phase):
    state, valid = np.asarray(state), np.asarray(valid)
    if state.shape != (150,) or valid.shape != (150,) or valid.dtype != bool:
        raise ValueError('Whole original150-frame label reference required')
    if not valid.all():
        return 'label_unknown_in_window'
    if np.all(state == 1):
        return 'active'
    if np.all(state == 0) and phase == 'pre_fault':
        return 'pre_fault'
    if np.all(state == 0):
        return 'normal_other_phase'
    return 'mixed_boundary'


def fit_diagnostic_reference(frame):
    source = frame[(frame['split'] == 'train') & frame.kind.eq('normal') & frame.common_eligible]
    if not len(source):
        raise ValueError('Selected source-train original-normal reference empty')
    result = {}
    for feature in FEATURES:
        x = source[feature].to_numpy(dtype=float)
        if not np.isfinite(x).all():
            raise ValueError('Invalid diagnostic reference feature')
        q1, median, q3 = np.quantile(x, [.25, .5, .75])
        scale = q3 - q1
        result[feature] = {'median': float(median), 'iqr': float(scale), 'windows': len(x),
            'usable_scale': bool(scale > 1e-12), 'fit_split': 'selected_source_train_original_normal_only'}
    return result


def describe(values):
    values = np.asarray(values, dtype=float)
    if not len(values):
        return {'windows': 0, 'median': None, 'p95': None}
    if not np.isfinite(values).all():
        raise ValueError('Invalid values reached descriptive statistics')
    return {'windows': len(values), 'median': float(np.median(values)), 'p95': float(np.quantile(values, .95))}


def phase_stability(frame, reference):
    rows = []
    normals = frame[frame.kind.eq('normal')]
    for uid, case in normals.groupby('case_uid', sort=True):
        metadata = case.iloc[0]
        for feature in FEATURES:
            early = case[case.common_eligible & case.crop_quarter.eq(0)][feature].to_numpy(dtype=float)
            late = case[case.common_eligible & case.crop_quarter.eq(3)][feature].to_numpy(dtype=float)
            a, b = describe(early), describe(late)
            shift = b['median'] - a['median'] if len(early) and len(late) else None
            scale = reference[feature]
            rows.append({'case_uid': uid, 'split': metadata['split'], 'domain': metadata.domain,
                'association_group': metadata.association_group, 'feature': feature,
                'early': a, 'late': b, 'late_minus_early_median': shift,
                'shift_in_train_normal_iqr': shift / scale['iqr'] if shift is not None and scale['usable_scale'] else None})
    return rows


def matched_fault_responses(frame, reference):
    rows = []
    normal = frame[frame.kind.eq('normal') & frame.common_eligible]
    faults = frame[frame.kind.eq('fault')]
    for uid, case in faults.groupby('case_uid', sort=True):
        metadata = case.iloc[0]
        active = case[case.common_eligible & case.window_group.eq('active')]
        pre = case[case.common_eligible & case.window_group.eq('pre_fault')]
        controls = normal[(normal['split'] == metadata['split']) & (normal.domain == metadata.domain)]
        for feature in FEATURES:
            a, b = describe(pre[feature]), describe(active[feature])
            details = []
            for q in range(4):
                control = controls[controls.crop_quarter.eq(q)][feature].to_numpy(dtype=float)
                positive = active[active.crop_quarter.eq(q)][feature].to_numpy(dtype=float)
                details.append({'crop_quarter': q, 'normal_windows': len(control), 'fault_windows': len(positive),
                    'auc_larger_fixed_feature_means_fault': fixed_auc(control, positive),
                    'normal': describe(control), 'fault': describe(positive)})
            matched = [row for row in details if row['auc_larger_fixed_feature_means_fault'] is not None]
            count = sum(row['fault_windows'] for row in matched)
            auc = sum(row['fault_windows'] * row['auc_larger_fixed_feature_means_fault'] for row in matched) / count if count else None
            shift = b['median'] - a['median'] if a['windows'] and b['windows'] else None
            scale = reference[feature]
            rows.append({'case_uid': uid, 'split': metadata['split'], 'domain': metadata.domain,
                'association_group': metadata.association_group, 'subtype': metadata.subtype,
                'feature': feature, 'pre_fault': a, 'active': b,
                'active_minus_pre_median': shift,
                'active_minus_pre_in_train_normal_iqr': shift / scale['iqr'] if shift is not None and scale['usable_scale'] else None,
                'matched_crop_quarter_auc': auc, 'matched_active_windows': count,
                'unmatched_active_windows': len(active) - count, 'quarter_details': details})
    return rows


def load_case_streams(record, payload, policy):
    from pyulog import ULog
    from build_full_causal_input import extract_group
    grid = payload['timestamp_us']
    log = ULog(record['raw_ulog'], message_name_filter_list=['vehicle_air_data', 'vehicle_local_position', 'vehicle_attitude', 'sensor_combined'])
    topics = {'baro': 'vehicle_air_data', 'velocity': 'vehicle_local_position', 'attitude': 'vehicle_attitude', 'gyro': 'sensor_combined'}
    markers = {'baro': ['baro_device_id'], 'velocity': ['z_reset_counter', 'vz_reset_counter'], 'attitude': ['quat_reset_counter'], 'gyro': []}
    columns = {'baro': ['baro_alt_relative'], 'velocity': ['vel_z'], 'attitude': ['att_q_' + x for x in 'wxyz'], 'gyro': ['gyro_' + x for x in 'xyz']}
    traces = {}; diagnostics = {}
    for group, topic in topics.items():
        found = [ds.data for ds in log.data_list if ds.name == topic and ds.multi_id == 0]
        if len(found) != 1:
            raise ValueError('Missing or ambiguous native topic: ' + topic)
        raw = found[0]; trace, offset = extract_group(raw, group, grid); n = len(trace.available_us)
        values = trace.values[:, 2:3] if group == 'velocity' else trace.values
        counters = {key: raw[key][offset:offset + n] for key in markers[group] if key in raw}
        valid = np.ones(n, dtype=bool)
        if group == 'velocity' and 'v_z_valid' in raw:
            valid &= raw['v_z_valid'][offset:offset + n] == 1
        traces[group] = Trace(trace.observation_us, trace.available_us, values, int(policy['max_age_us'][group]), counters, valid)
        idx = np.searchsorted(trace.available_us, grid, side='right') - 1
        if (idx < 0).any():
            raise ValueError('No preceding required native record')
        for key, values_clock in [('observation_us', trace.observation_us), ('available_us', trace.available_us)]:
            if not np.array_equal(values_clock[idx], payload[group + '__' + key]):
                raise ValueError('Native provenance differs from original build')
        mask = payload[group + '__valid']
        c = [policy['columns'].index(name) for name in columns[group]]
        if not np.array_equal(values[idx][mask], payload['canonical_rows'][mask][:, c]):
            raise ValueError('Physical values differ from original build')
        diagnostics[group] = {'counter_changes': {key: int(np.count_nonzero(np.diff(v.astype(np.int64)))) for key, v in counters.items()},
            'counter_fields_missing': [key for key in markers[group] if key not in raw]}
    return traces, diagnostics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, default=Path('D:/UAV-Data/causal_input_v1'))
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('New output root required, never overwrite prior diagnosis')
    started = time.perf_counter(); torch.set_num_threads(2)
    plan = read(args.plan); selected = plan['cases']
    if len(selected) != 36 or len({r['case_uid'] for r in selected}) != 36:
        raise ValueError('Fixed36 unique source cases required')
    if Counter(row['kind'] for row in selected) != {'normal': 12, 'fault': 24}:
        raise ValueError('Fixed normal/fault scope changed')
    if any(row['split'] not in ['train', 'val'] or row['case_uid'].split('/')[1] != 'hover' for row in selected):
        raise ValueError('Only frozen source hover train/val cases permitted')
    policy = read(args.data / 'input_contract.json'); norm = read(args.data / 'norm_stats.json')
    protocol = read(ROOT / 'artifacts/causal_source_longbudget_v1_20261101/corrected_longbudget_protocol.json')
    if norm['columns'] != policy['columns']:
        raise ValueError('Exact current source-normalization column schema required')
    inputs = {str(args.plan): sha(args.plan)}
    for relative in ['norm_stats.json', 'input_contract.json', 'ledgers/train_included_windows.csv', 'ledgers/val_included_windows.csv']:
        path = args.data / relative
        if sha(path) != protocol['data_identities'][relative]:
            raise ValueError('Source identity mismatch: ' + relative)
        inputs[str(path)] = sha(path)
    for relative in ['packages/uav_fault/uav_fault/models/cnn/lstm.py', 'packages/uav_fault/uav_fault/causal_input.py',
            'packages/uav_fault/uav_fault/causal_pipeline.py', 'scripts/data/build_full_causal_input.py', 'scripts/research/validate_kinematic_residuals.py']:
        path = ROOT / relative; inputs[str(path)] = sha(path)
    frames = {split: pd.read_csv(args.data / 'ledgers' / (split + '_included_windows.csv'), keep_default_na=False, low_memory=False) for split in ['train', 'val']}
    records = {}
    for selection in selected:
        uid = selection['case_uid']; key = uid.replace('/', '__')
        path = args.data / 'case_records' / (key + '.json'); record = read(path)
        if record['split'] != selection['split'] or record['role'] != selection['split'] or (record['family'] == 'normal') != (selection['kind'] == 'normal'):
            raise ValueError('Registered member identity differs')
        case_file = args.data / 'cases' / (key + '.npz')
        expected = next(h for name, h in record['output_identities'].items() if Path(name).name == case_file.name)
        if sha(case_file) != expected or sha(record['raw_ulog']) != record['raw_ulog_sha256']:
            raise ValueError('Registered native/case identity changed')
        for path, expected in [(path, sha(path)), (case_file, expected), (Path(record['raw_ulog']), record['raw_ulog_sha256'])]:
            inputs[str(path)] = expected
        records[uid] = (record, case_file)
    args.output.mkdir(parents=True, exist_ok=False)
    save(args.output / 'selection_before_analysis.json', {'plan_sha256': sha(args.plan), 'cases': selected,
        'inputs': inputs, 'script_sha256': sha(__file__), 'created_at': datetime.now(timezone.utc).isoformat()})
    all_rows = []; cases = {}
    for selection in selected:
        uid = selection['case_uid']; key = uid.replace('/', '__'); record, case_file = records[uid]
        with np.load(case_file, allow_pickle=False) as source:
            payload = {name: source[name].copy() for name in source.files}
        linked = frames[selection['split']]
        linked = linked[linked.case_uid.eq(uid)].copy()
        if not len(linked) or not linked.qualified_y.isin([0, 1]).all() or not linked.supervised_valid.all():
            raise ValueError('Selected source ledger population invalid')
        grid = payload['timestamp_us']; rows = len(grid)
        traces, diagnostic = load_case_streams(record, payload, policy)
        residual = compute_residuals(grid, payload['row_input_valid'], traces)
        starts = payload['window_start_row']; ids = payload['original_cache_index']
        window_mask = window_residual_mask(residual['joint_valid'], starts)
        window_mask[~payload['supervised_valid']] = False
        lookup = {int(index): i for i, index in enumerate(ids)}
        case_rows = []
        for item in linked.itertuples():
            original = int(item.original_cache_index); position = lookup[original]
            start = int(item.window_start_row); end = int(item.window_end_row)
            if item.window_start_row != start or item.window_end_row != end or start != int(starts[position]) or end - start != 150:
                raise ValueError('Original window identity mismatch')
            eligible = bool(window_mask[position, 10:].all())
            group = whole_window_group(payload['reference_state'][start:end], payload['row_label_valid'][start:end], item.phase)
            row = dict(selection, association_group=str(item.association_group), original_cache_index=original,
                window_start_row=start, window_end_row=end, crop_rows=rows, crop_quarter=quarter(end, rows),
                qualified_y=int(item.qualified_y), phase=item.phase, window_group=group, common_eligible=eligible,
                valid_residual_positions=int(window_mask[position].sum()),
                reason_vertical_bits=int(np.bitwise_or.reduce(residual['reason'][start + 10:end, 0])),
                reason_attitude_bits=int(np.bitwise_or.reduce(residual['reason'][start + 10:end, 1])))
            for feature in FEATURES:
                row[feature] = np.nan; row[feature + '__within_window_p95'] = np.nan
            row.update(vertical_observation_skew_median_us=np.nan, attitude_observation_skew_median_us=np.nan)
            if eligible:
                x = canonical_window(payload['canonical_rows'][start:end], policy['columns'], payload['row_input_valid'][start:end], mean=norm['mean'], std=norm['std'])
                old = legacy_features(x, policy['columns'])[:, 10:]
                rh = residual['vertical_m'][start + 10:end]; rR = residual['attitude_rad'][start + 10:end]
                values = np.vstack([rh[None, :], rR.T, np.abs(rh)[None, :], np.linalg.norm(rR, axis=1)[None, :], old])
                if values.shape != (15, 140) or not np.isfinite(values).all():
                    raise ValueError('Finite matched140-row feature contributions required')
                for feature, values_row in zip(FEATURES, values):
                    row[feature] = float(np.median(values_row))
                    row[feature + '__within_window_p95'] = float(np.quantile(values_row, .95))
                ages = residual['endpoint_age_us'][start + 10:end]
                row['vertical_observation_skew_median_us'] = float(np.median(np.abs(ages[:, 0] - ages[:, 1])))
                row['attitude_observation_skew_median_us'] = float(np.median(np.abs(ages[:, 2] - ages[:, 3])))
            case_rows.append(row)
        case_frame = pd.DataFrame(case_rows)
        case_frame.to_csv(args.output / (key + '_window_features.csv'), index=False)
        np.savez_compressed(args.output / (key + '_residuals.npz'), timestamp_us=grid, **residual,
            window_start_row=starts, original_cache_index=ids, original_supervised_valid=payload['supervised_valid'],
            window_residual_mask=window_mask)
        case_summary = {**selection, 'original_cap64_windows': len(starts), 'originally_admitted_windows': len(linked),
            'common_eligible_windows': int(case_frame.common_eligible.sum()),
            'eligible_window_groups': dict(Counter(case_frame.loc[case_frame.common_eligible, 'window_group'])),
            'all_original_input_qualifications_preserved': True, 'native_diagnostics': diagnostic,
            'residual_rows': summarize(residual)}
        cases[uid] = case_summary; all_rows.extend(case_rows)
        print('KINEMATIC_RESPONSE_CASE', selection['split'], uid, 'eligible', int(case_frame.common_eligible.sum()), '/', len(linked), flush=True)
    features = pd.DataFrame(all_rows)
    features.to_csv(args.output / 'window_features.csv', index=False)
    reference = fit_diagnostic_reference(features)
    save(args.output / 'diagnostic_reference_train_normal.json', reference)
    stages = phase_stability(features, reference)
    responses = matched_fault_responses(features, reference)
    save(args.output / 'normal_case_stage_stability.json', stages)
    save(args.output / 'matched_fault_responses.json', responses)
    def compact(rows, group_columns, value):
        f = pd.DataFrame([{**{key: row[key] for key in group_columns}, value: row[value]} for row in rows])
        result = []
        for key, g in f.groupby(group_columns, sort=True):
            key = key if isinstance(key, tuple) else (key,)
            x = g[value].dropna().to_numpy(dtype=float)
            result.append({**dict(zip(group_columns, key)), 'cases': len(g), 'cases_with_value': len(x),
                'case_value_median': float(np.median(x)) if len(x) else None,
                'case_value_min': float(np.min(x)) if len(x) else None,
                'case_value_max': float(np.max(x)) if len(x) else None})
        return result
    stage_summary = compact(stages, ['split', 'feature'], 'shift_in_train_normal_iqr')
    response_summary = compact(responses, ['split', 'subtype', 'feature'], 'matched_crop_quarter_auc')
    for path, expected in inputs.items():
        if sha(path) != expected:
            raise RuntimeError('Immutable input changed during CPU analysis: ' + path)
    summary = {'status': 'COMPLETE_FIXED36_SOURCE_HOVER_FEATURE_RESPONSE_DIAGNOSIS',
        'completed_at': datetime.now(timezone.utc).isoformat(), 'elapsed_s': time.perf_counter() - started,
        'case_count': len(selected), 'original_cap64_windows': sum(c['original_cap64_windows'] for c in cases.values()),
        'original_admitted_windows': len(features), 'common_eligible_windows': int(features.common_eligible.sum()),
        'cases': cases, 'normal_stability_case_summary': stage_summary, 'matched_fault_auc_case_summary': response_summary,
        'diagnostic_reference_only_from_selected_train_original_normal': True,
        'residual_training_normalization_fitted': False, 'model_weights_loaded': False,
        'classifier_training_or_inference': False, 'GPU_server_or_final_test_Real_used': False,
        'thresholds_signs_or_horizon_selected_by_results': False, 'existing_data_model_and_stats_modified': False,
        'inputs': inputs, 'script_sha256': sha(__file__),
        'limits': ['Hover-only small deterministic sample, not all fault severity/subtypes or physical onset proof.',
            'Source val is previously exposed development data, not blind final testing.',
            'Matched crop quarter is a coarse background adjustment, not matched physical states.',
            'SIL/HIL simulation pairs and overlapping windows are correlated; no sample-wise significance claims.',
            'Fixed-sign descriptive AUC does not validate a deployable threshold, multi-feature model or anomaly detector.']}
    save(args.output / 'summary.json', summary)
    print('COMPLETE_FIXED36_SOURCE_HOVER_FEATURE_RESPONSE_DIAGNOSIS', flush=True)
    print({key: summary[key] for key in ['elapsed_s', 'case_count', 'original_cap64_windows', 'original_admitted_windows', 'common_eligible_windows']}, flush=True)


if __name__ == '__main__':
    main()
