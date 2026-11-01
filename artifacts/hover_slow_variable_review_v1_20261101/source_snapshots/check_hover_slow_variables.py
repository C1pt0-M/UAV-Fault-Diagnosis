"""Descriptive check of existing hover inputs, time labels and saved val scores.

No model inference/training, raw-log scan, new normalization or final tests.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
import sys

import numpy as np
import pandas as pd

from validate_kinematic_residuals import ROOT, read, save, sha
sys.path.insert(0, str(ROOT / 'packages/uav_fault'))
from uav_fault.causal_input import canonical_window
from check_kinematic_feature_response import quarter


def label_time_counts(frame):
    hover = frame[frame.case_uid.str.split('/').str[1].eq('hover')].copy()
    # Original endpoints are exclusive row positions; last timestamp is end-1.
    hover['elapsed_s'] = (hover.window_end_row.to_numpy(dtype=float) - 1) * .02
    edges = [0, 10, 20, 30, 40, 60, float('inf')]
    result = []
    for origin, select in [('independent_normal', hover.case_level2_label.eq('normal')),
            ('fault_case', hover.case_level2_label.ne('normal'))]:
        for left, right in zip(edges[:-1], edges[1:]):
            part = hover[select & hover.elapsed_s.ge(left) & hover.elapsed_s.lt(right)]
            result.append({'origin': origin, 'elapsed_crop_seconds': [left, right if np.isfinite(right) else None],
                'windows': len(part), 'normal': int(part.qualified_y.eq(0).sum()), 'fault': int(part.qualified_y.eq(1).sum()),
                'cases': int(part.case_uid.nunique())})
    return result


def paired_channel_changes(frame, columns):
    result = []
    for uid, case in frame[frame.kind.eq('normal')].groupby('case_uid', sort=True):
        first = case.iloc[0]
        early = case[case.crop_quarter.eq(0)]; late = case[case.crop_quarter.eq(3)]
        for channel in columns:
            name = 'mean__' + channel
            a = float(early[name].median()) if len(early) else None
            b = float(late[name].median()) if len(late) else None
            result.append({'case_uid': uid, 'split': first['split'], 'domain': first.domain, 'channel': channel,
                'early_windows': len(early), 'late_windows': len(late), 'early_median_window_mean_z': a,
                'late_median_window_mean_z': b, 'late_minus_early_source_std': b - a if a is not None and b is not None else None,
                'early_median_within_window_std_z': float(early['std__' + channel].median()) if len(early) else None,
                'late_median_within_window_std_z': float(late['std__' + channel].median()) if len(late) else None})
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, default=Path('D:/UAV-Data/causal_input_v1'))
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('Fresh diagnosis output required')
    selection_path = ROOT / 'artifacts/kinematic_residual_discrimination_v1_20261101/plan.json'
    selection = read(selection_path)['cases']
    if len(selection) != 36 or any(c['split'] not in ['train', 'val'] for c in selection):
        raise ValueError('Reuse fixed36 source cases only')
    norm = read(args.data / 'norm_stats.json'); columns = norm['columns']
    if any(name in columns for name in ['pos_x', 'pos_y', 'pos_z', 'time_s', 'rate_err_x']):
        raise ValueError('Unexpected actual26-channel schema; do not assume missing fields')
    protocol = read(ROOT / 'artifacts/causal_source_longbudget_v1_20261101/corrected_longbudget_protocol.json')
    inputs = {str(selection_path): sha(selection_path), str(args.plan): sha(args.plan)}
    ledgers = {}
    for split in ['train', 'val']:
        relative = 'ledgers/' + split + '_included_windows.csv'
        path = args.data / relative
        if sha(path) != protocol['data_identities'][relative]:
            raise ValueError('Current source ledger identity mismatch')
        inputs[str(path)] = sha(path)
        ledgers[split] = pd.read_csv(path, keep_default_na=False, low_memory=False)
    for name in ['input_contract.json', 'norm_stats.json']:
        path = args.data / name
        if sha(path) != protocol['data_identities'][name]:
            raise ValueError('Source contract/stat identity mismatch')
        inputs[str(path)] = sha(path)
    args.output.mkdir(parents=True, exist_ok=False)
    save(args.output / 'selection_before_profiles.json', {'cases': selection, 'inputs': inputs,
        'script_sha256': sha(__file__), 'created_at': datetime.now(timezone.utc).isoformat()})
    labels = {split: label_time_counts(f) for split, f in ledgers.items()}
    profiles = []
    for member in selection:
        uid = member['case_uid']; key = uid.replace('/', '__')
        receipt_path = args.data / 'case_records' / (key + '.json'); receipt = read(receipt_path)
        if receipt['split'] != member['split'] or receipt['role'] != member['split']:
            raise ValueError('Selected source split mismatch')
        case_path = args.data / 'cases' / (key + '.npz')
        expected = next(h for name, h in receipt['output_identities'].items() if Path(name).name == case_path.name)
        if sha(case_path) != expected:
            raise ValueError('Source case identity mismatch')
        inputs[str(receipt_path)] = sha(receipt_path); inputs[str(case_path)] = expected
        f = ledgers[member['split']]; f = f[f.case_uid.eq(uid)]
        with np.load(case_path, allow_pickle=False) as payload:
            rows = payload['canonical_rows']; input_valid = payload['row_input_valid']
            if not np.all(np.diff(payload['timestamp_us']) == 20000):
                raise ValueError('Original20ms crop grid required')
            for item in f.itertuples():
                start, end = int(item.window_start_row), int(item.window_end_row)
                if end - start != 150 or not input_valid[start:end].all():
                    raise ValueError('Unqualified original window reached profile')
                x = canonical_window(rows[start:end], columns, input_valid[start:end], mean=norm['mean'], std=norm['std'])
                means = x.astype(np.float64).mean(axis=1); std = x.astype(np.float64).std(axis=1)
                row = {**member, 'original_cache_index': int(item.original_cache_index), 'qualified_y': int(item.qualified_y),
                    'crop_quarter': quarter(end, len(rows)), 'elapsed_crop_s': (end - 1) * .02}
                row.update({'mean__' + name: float(value) for name, value in zip(columns, means)})
                row.update({'std__' + name: float(value) for name, value in zip(columns, std)})
                # Confirms the channel is window-relative before fixed source normalization.
                altitude = columns.index('baro_alt_relative')
                expected_start = np.float32(-np.float32(norm['mean'][altitude]) / np.float32(norm['std'][altitude]))
                if x[altitude, 0] != expected_start:
                    raise ValueError('Barometric origin was not subtracted per window')
                profiles.append(row)
    frame = pd.DataFrame(profiles)
    frame.to_csv(args.output / 'all26_window_profiles.csv', index=False)
    shifts = paired_channel_changes(frame, columns)
    save(args.output / 'normal_case_channel_changes.json', shifts)
    predictions = {}
    for arm in protocol['arms']:
        path = ROOT / 'artifacts/causal_source_longbudget_v1_20261101/results/runs' / arm / 'source_validation_predictions.npz'
        inputs[str(path)] = sha(path)
        with np.load(path, allow_pickle=False) as saved:
            if not np.array_equal(saved['original_indices'], ledgers['val'].original_cache_index) or not np.array_equal(saved['labels'], ledgers['val'].qualified_y):
                raise ValueError('Saved current long-budget source-val prediction identity mismatch')
            positions = {int(original): i for i, original in enumerate(saved['original_indices'])}
            probabilities = saved['probabilities']; hard = probabilities.argmax(1)
            rows = []
            for uid, case in frame[frame.kind.eq('normal') & frame['split'].eq('val')].groupby('case_uid', sort=True):
                for q in [0, 3]:
                    part = case[case.crop_quarter.eq(q)]
                    indices = np.array([positions[int(v)] for v in part.original_cache_index], dtype=np.int64)
                    if not np.all(saved['labels'][indices] == 0):
                        raise ValueError('Normal-only descriptive score comparison required')
                    rows.append({'case_uid': uid, 'crop_quarter': q, 'windows': len(indices),
                        'false_positives': int(hard[indices].sum()),
                        'fault_score_median': float(np.median(probabilities[indices, 1])) if len(indices) else None})
            predictions[arm] = rows
    channel_summary = []
    for split in ['train', 'val']:
        for channel in columns:
            values = [row['late_minus_early_source_std'] for row in shifts if row['split'] == split and row['channel'] == channel and row['late_minus_early_source_std'] is not None]
            channel_summary.append({'split': split, 'channel': channel, 'cases_with_pair': len(values),
                'median_paired_shift_source_std': float(np.median(values)) if values else None,
                'median_absolute_paired_shift_source_std': float(np.median(np.abs(values))) if values else None})
    summary = {'status': 'COMPLETE_CURRENT_HOVER_SLOW_VARIABLE_DESCRIPTIVE_CHECK',
        'actual_columns': columns, 'profile_cases': frame.case_uid.nunique(), 'profile_windows': len(frame),
        'label_time_inventory_all_source_hover': labels, 'all26_case_paired_channel_summary': channel_summary,
        'saved_longbudget_source_val_normal_scores': predictions,
        'per_window_baro_origin_subtraction_verified_all_profile_windows': True,
        'inputs': inputs, 'script_sha256': sha(__file__), 'classifier_training_or_inference': False,
        'raw_logs_or_final_test_Real_read': False, 'normalization_or_features_changed': False,
        'limits': 'Passive profile/score co-variation and time-label association do not identify a causal shortcut or authorize deleting channels. Original crop position is not authenticated flight phase.'}
    for path, expected in inputs.items():
        if sha(path) != expected:
            raise ValueError('Read-only source identity changed')
    save(args.output / 'summary.json', summary)
    print(summary['status'], 'cases', summary['profile_cases'], 'windows', summary['profile_windows'])
    print('VAL_ALL26_PAIRED_SHIFTS', channel_summary[26:])
    print('SAVED_MODEL_NORMAL_SCORES', predictions)


if __name__ == '__main__':
    main()
