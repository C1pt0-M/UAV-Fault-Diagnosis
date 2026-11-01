"""CPU-only fixed normal-anchor kinematic residual feasibility check.

These are fixed-horizon *delivered-stream* consistency diagnostics: availability
change points, not fictitious synchronous50Hz observations. Real observation
ages/skew remain explicit. No model, training, cache rebuild or normalization fit.
"""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'packages/uav_fault'))
sys.path.insert(0, str(ROOT / 'scripts/data'))
HORIZON_US = 200000
PREFIX = 1
INPUT_INVALID = 2
TOPIC_INVALID = 4
RESET = 8
INVALID_QUATERNION = 16


@dataclass
class Trace:
    observation_us: np.ndarray
    available_us: np.ndarray
    values: np.ndarray
    max_age_us: int
    counters: dict[str, np.ndarray] = field(default_factory=dict)
    valid: np.ndarray | None = None

    def __post_init__(self):
        self.observation_us = np.asarray(self.observation_us)
        self.available_us = np.asarray(self.available_us)
        self.values = np.asarray(self.values, dtype=np.float64)
        a, o = self.available_us, self.observation_us
        if a.ndim != 1 or o.shape != a.shape or a.dtype.kind not in 'iu' or o.dtype.kind not in 'iu':
            raise ValueError('Explicit integer availability and observation clocks required')
        if not len(a) or self.values.ndim != 2 or len(self.values) != len(a):
            raise ValueError('Nonempty consistent trace layout required')
        if (o <= 0).any() or (o > a).any() or (np.diff(a.astype(np.int64)) < 0).any() or (np.diff(o.astype(np.int64)) < 0).any():
            raise ValueError('Clock invalid/reversed; no sorting or fallback')
        if not isinstance(self.max_age_us, int) or self.max_age_us < 0:
            raise ValueError('Explicit nonnegative age budget required')
        if self.valid is None:
            self.valid = np.ones(len(a), dtype=bool)
        else:
            self.valid = np.asarray(self.valid)
            if self.valid.dtype != bool or self.valid.shape != a.shape:
                raise ValueError('Trace validity must be explicit boolean rows')
        for name, values in self.counters.items():
            if np.asarray(values).shape != a.shape:
                raise ValueError('Reset/device marker layout mismatch: ' + name)

    def index(self, t):
        return int(np.searchsorted(self.available_us, t, side='right')) - 1

    def point(self, t):
        i = self.index(t)
        if i < 0:
            raise ValueError('not_yet_available')
        age = int(t) - int(self.observation_us[i])
        if age < 0 or age > self.max_age_us or not self.valid[i] or not np.isfinite(self.values[i]).all():
            raise ValueError('latest_missing_stale_nonfinite_or_flagged')
        return i

    def support(self, start, end):
        """Actual availability knots; no late-observation retroactive rewriting."""
        first, last = self.point(start), self.point(end)
        if end <= start:
            raise ValueError('Positive fixed interval required')
        middle = self.available_us[(self.available_us > start) & (self.available_us < end)]
        knots = np.r_[start, np.unique(middle), end].astype(np.int64)
        indices = np.searchsorted(self.available_us, knots[:-1], side='right') - 1
        durations = np.diff(knots) / 1e6
        if not self.valid[indices].all() or not np.isfinite(self.values[indices]).all():
            raise ValueError('interior_invalid_latest')
        # At a new arrival an invalid older row stops being latest; test its
        # age immediately before replacement as well as both exact endpoints.
        max_ages = knots[1:] - self.observation_us[indices]
        if (max_ages > self.max_age_us).any():
            raise ValueError('interior_stale_gap')
        support_indices = np.unique(np.r_[indices, last])
        changed = any(np.any(np.asarray(v)[support_indices] != np.asarray(v)[first]) for v in self.counters.values())
        return indices, durations, support_indices, changed


def normalize_quaternion_wxyz(q):
    q = np.asarray(q, dtype=np.float64)
    if q.shape[-1:] != (4,) or not np.isfinite(q).all():
        raise ValueError('Quaternion must be finite wxyz')
    norm = np.linalg.norm(q, axis=-1, keepdims=True)
    if not np.isfinite(norm).all() or (norm < 1e-8).any():
        raise ValueError('Invalid quaternion norm; never clamp into fake rotation')
    return q / norm


def rotation_wxyz(q):
    normalized = normalize_quaternion_wxyz(q)
    return Rotation.from_quat(normalized[..., [1, 2, 3, 0]])


def integrate_rotation(omega, durations, *, left=False):
    omega = np.asarray(omega, dtype=np.float64)
    durations = np.asarray(durations, dtype=np.float64)
    if omega.shape != (len(durations), 3) or not np.isfinite(omega).all() or not np.isfinite(durations).all() or (durations <= 0).any():
        raise ValueError('Finite angular rate and strictly positive actual intervals required')
    increments = Rotation.from_rotvec(omega * durations[:, None]).as_matrix()
    result = np.eye(3)
    for step in increments:
        result = step @ result if left else result @ step
    return result


def compute_residuals(grid, input_valid, traces):
    grid = np.asarray(grid)
    input_valid = np.asarray(input_valid)
    if grid.ndim != 1 or grid.dtype != np.int64 or len(grid) < 11 or not np.all(np.diff(grid) == 20000):
        raise ValueError('Original20ms grid, no silent regrid')
    if input_valid.dtype != bool or input_valid.shape != grid.shape or set(traces) != {'baro', 'velocity', 'attitude', 'gyro'}:
        raise ValueError('Original input mask and four explicit traces required')
    n = len(grid)
    vertical = np.full(n, np.nan); wrong_sign = np.full(n, np.nan)
    attitude = np.full((n, 3), np.nan); left = np.full((n, 3), np.nan)
    reasons = np.zeros((n, 2), dtype=np.uint16)
    ages = np.full((n, 4, 2), -1, dtype=np.int64)
    for j, t in enumerate(grid):
        if j < 10:
            reasons[j] = PREFIX
            continue
        if not input_valid[j - 10:j + 1].all():
            reasons[j] = INPUT_INVALID
            continue
        start = int(t) - HORIZON_US
        for column, groups in enumerate([('baro', 'velocity'), ('attitude', 'gyro')]):
            info = {}
            try:
                for group in groups:
                    trace = traces[group]
                    indices, durations, support, changed = trace.support(start, int(t))
                    a, b = trace.point(start), trace.point(int(t))
                    slot = ['baro', 'velocity', 'attitude', 'gyro'].index(group)
                    ages[j, slot] = [start - trace.observation_us[a], int(t) - trace.observation_us[b]]
                    info[group] = (a, b, indices, durations, support)
                    if changed:
                        reasons[j, column] |= RESET
                if reasons[j, column]:
                    continue
                if column == 0:
                    a, b, _, _, _ = info['baro']
                    _, _, indices, dt, _ = info['velocity']
                    dh = float(traces['baro'].values[b, 0] - traces['baro'].values[a, 0])
                    displacement = float(np.sum(traces['velocity'].values[indices, 0] * dt))
                    vertical[j] = dh + displacement
                    wrong_sign[j] = dh - displacement
                else:
                    a, b, _, _, support = info['attitude']
                    try:
                        # Check every record in the interval, not only endpoints.
                        normalize_quaternion_wxyz(traces['attitude'].values[support])
                        r0, r1 = rotation_wxyz(traces['attitude'].values[[a, b]]).as_matrix()
                    except ValueError:
                        reasons[j, column] |= INVALID_QUATERNION
                        continue
                    _, _, indices, dt, _ = info['gyro']
                    right_delta = integrate_rotation(traces['gyro'].values[indices], dt)
                    left_delta = integrate_rotation(traces['gyro'].values[indices], dt, left=True)
                    attitude[j] = Rotation.from_matrix((r0 @ right_delta).T @ r1).as_rotvec()
                    left[j] = Rotation.from_matrix((left_delta @ r0).T @ r1).as_rotvec()
            except ValueError:
                reasons[j, column] |= TOPIC_INVALID
    result = {'vertical_m': vertical, 'wrong_vertical_sign_m': wrong_sign, 'attitude_rad': attitude,
        'left_multiplication_rad': left, 'reason': reasons, 'endpoint_age_us': ages,
        'vertical_valid': reasons[:, 0] == 0, 'attitude_valid': reasons[:, 1] == 0}
    result['joint_valid'] = result['vertical_valid'] & result['attitude_valid']
    if not np.isfinite(vertical[result['vertical_valid']]).all() or not np.isfinite(attitude[result['attitude_valid']]).all():
        raise RuntimeError('Nonfinite value in valid residuals')
    return result


def window_residual_mask(joint_valid, starts, frames=150):
    joint_valid = np.asarray(joint_valid)
    starts = np.asarray(starts)
    if joint_valid.ndim != 1 or joint_valid.dtype != bool or starts.dtype != np.int64 or frames != 150:
        raise ValueError('Explicit original window IDs and boolean residual mask required')
    if (starts < 0).any() or (starts + frames > len(joint_valid)).any():
        raise ValueError('Original window outside crop')
    mask = np.zeros((len(starts), frames), dtype=bool)
    for i, start in enumerate(starts):
        mask[i, 10:] = joint_valid[start + 10:start + frames]
    return mask


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def save(path, value):
    with Path(path).open('x', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)


def distribution(values):
    v = np.asarray(values)
    if not len(v):
        return {'count': 0}
    if not np.isfinite(v).all():
        raise ValueError('Distribution must contain only explicitly valid samples')
    return {'count': len(v), 'mean': float(np.mean(v)), 'std': float(np.std(v)),
        'min': float(np.min(v)), 'median': float(np.median(v)), 'p95': float(np.quantile(v, .95)),
        'p99': float(np.quantile(v, .99)), 'max': float(np.max(v))}


def summarize(result):
    v, r = result['vertical_valid'], result['attitude_valid']
    ages = result['endpoint_age_us']
    return {'rows': len(v), 'vertical_valid': int(v.sum()), 'attitude_valid': int(r.sum()),
        'joint_valid': int(result['joint_valid'].sum()),
        'vertical_abs_m': distribution(np.abs(result['vertical_m'][v])),
        'vertical_signed_m': distribution(result['vertical_m'][v]),
        'wrong_sign_vertical_abs_m': distribution(np.abs(result['wrong_vertical_sign_m'][v])),
        'attitude_norm_rad': distribution(np.linalg.norm(result['attitude_rad'][r], axis=1)),
        'wrong_left_norm_rad': distribution(np.linalg.norm(result['left_multiplication_rad'][r], axis=1)),
        'vertical_endpoint_observation_skew_abs_us': distribution(np.abs(ages[v, 0] - ages[v, 1]).ravel()),
        'attitude_endpoint_observation_skew_abs_us': distribution(np.abs(ages[r, 2] - ages[r, 3]).ravel()),
        'invalid_reason_bits': {name: {'vertical': int(((result['reason'][:, 0] & flag) != 0).sum()),
            'attitude': int(((result['reason'][:, 1] & flag) != 0).sum())}
            for name, flag in [('prefix', PREFIX), ('original_input_span_invalid', INPUT_INVALID),
                ('required_native_topic_invalid', TOPIC_INVALID), ('reset_or_device_change', RESET), ('invalid_quaternion', INVALID_QUATERNION)]}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, default=Path('D:/UAV-Data/causal_input_v1'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--plan', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('Independent fresh output required')
    plan = read(args.plan)
    if plan['horizon_us'] != HORIZON_US or plan['grid_us'] != 20000:
        raise ValueError('Frozen horizon/grid mismatch')
    anchor_path = ROOT / 'artifacts/causal_input_parity_v1_20261031/fit_timing.json'
    uids = list(read(anchor_path))
    if len(uids) != 12 or Counter(uid.split('/')[0] for uid in uids) != {'SIL': 6, 'HIL': 6}:
        raise ValueError('Original12 source-train anchors required')
    policy = read(args.data / 'input_contract.json')
    inputs = {str(args.plan): sha(args.plan), str(anchor_path): sha(anchor_path),
        str(args.data / 'input_contract.json'): sha(args.data / 'input_contract.json'),
        str(args.data / 'norm_stats.json'): sha(args.data / 'norm_stats.json')}
    # No build tasks/test/Real metadata or tensors are read.
    selected = []
    for uid in uids:
        key = uid.replace('/', '__')
        path = args.data / 'case_records' / (key + '.json')
        record = read(path)
        if record['split'] != 'train' or record['role'] != 'train' or record['family'] != 'normal':
            raise ValueError('Only original-source-train normal cases allowed')
        if not Path(record['raw_ulog']).is_file():
            raise ValueError('Registered source ULog missing')
        case = args.data / 'cases' / (key + '.npz')
        expected = next(v for p, v in record['output_identities'].items() if Path(p).name == case.name)
        if sha(case) != expected or sha(record['raw_ulog']) != record['raw_ulog_sha256']:
            raise ValueError('Original case/raw identity changed')
        inputs[str(path)] = sha(path); inputs[str(case)] = expected
        inputs[record['raw_ulog']] = record['raw_ulog_sha256']
        selected.append((uid, record, case))
    args.output.mkdir(parents=True, exist_ok=False)
    save(args.output / 'selection_before_residuals.json', {'case_uids': uids, 'inputs': inputs,
        'script_sha256': sha(__file__), 'created_at': datetime.now(timezone.utc).isoformat()})
    from pyulog import ULog
    from build_full_causal_input import extract_group
    summaries = {}; all_valid = []; windows_total = 0
    for uid, record, case in selected:
        key = uid.replace('/', '__')
        with np.load(case, allow_pickle=False) as payload:
            grid = payload['timestamp_us'].copy(); row_valid = payload['row_input_valid'].copy()
            starts = payload['window_start_row'].copy(); original_ids = payload['original_cache_index'].copy()
            original_supervised = payload['supervised_valid'].copy()
            if not np.all(payload['reference_state'][payload['row_label_valid']] == 0):
                raise ValueError('Normal anchor has active fault reference')
            # Preserve exactly the canonical raw values/provenance identity.
            original_rows = payload['canonical_rows'].copy()
            provenance = {g: {k: payload[g + '__' + k].copy() for k in ['observation_us', 'available_us', 'valid']}
                for g in ['baro', 'velocity', 'attitude', 'gyro']}
        log = ULog(record['raw_ulog'], message_name_filter_list=['vehicle_air_data', 'vehicle_local_position', 'vehicle_attitude', 'sensor_combined'])
        traces = {}; diagnostics = {}
        topics = {'baro': 'vehicle_air_data', 'velocity': 'vehicle_local_position', 'attitude': 'vehicle_attitude', 'gyro': 'sensor_combined'}
        counter_names = {'baro': ['baro_device_id'], 'velocity': ['z_reset_counter', 'vz_reset_counter'],
            'attitude': ['quat_reset_counter'], 'gyro': []}
        value_flags = {'velocity': ['v_z_valid'], 'baro': [], 'attitude': [], 'gyro': []}
        for group, topic in topics.items():
            found = [ds.data for ds in log.data_list if ds.name == topic and ds.multi_id == 0]
            if len(found) != 1:
                raise ValueError('Missing/ambiguous required source topic: ' + topic)
            raw = found[0]
            # Reuse original checked extractor, including the declared gyro
            # publication-clock proxy. Retain latest message before crop origin.
            trace, offset = extract_group(raw, group, grid)
            count = len(trace.available_us)
            values = trace.values[:, 2:3] if group == 'velocity' else trace.values
            counters = {name: raw[name][offset:offset + count] for name in counter_names[group] if name in raw}
            valid = np.ones(count, dtype=bool)
            for name in value_flags[group]:
                if name in raw:
                    valid &= raw[name][offset:offset + count] == 1
            traces[group] = Trace(trace.observation_us, trace.available_us, values, int(policy['max_age_us'][group]), counters, valid)
            sample = traces[group]
            idx = np.searchsorted(sample.available_us, grid, side='right') - 1
            if (idx < 0).any():
                raise ValueError('Original input had no matching native preceding record')
            for name, source in [('observation_us', sample.observation_us), ('available_us', sample.available_us)]:
                assert np.array_equal(source[idx], provenance[group][name])
            columns = {'baro': ['baro_alt_relative'], 'velocity': ['vel_z'],
                'attitude': ['att_q_w', 'att_q_x', 'att_q_y', 'att_q_z'], 'gyro': ['gyro_x', 'gyro_y', 'gyro_z']}[group]
            indices = [policy['columns'].index(c) for c in columns]
            mask = provenance[group]['valid']
            if not np.array_equal(sample.values[idx][mask], original_rows[mask][:, indices]):
                raise ValueError('Physical topic values differ from existing causal build: ' + group)
            diagnostics[group] = {'native_records_in_retained_scope': count,
                'observation_to_publication_lag_us': distribution(sample.available_us - sample.observation_us),
                'reset_fields_present': list(counters),
                'reset_fields_missing': [name for name in counter_names[group] if name not in raw],
                'validity_flags_present': [name for name in value_flags[group] if name in raw],
                'native_counter_changes': {name: int(np.count_nonzero(np.diff(np.asarray(v, dtype=np.int64)))) for name, v in counters.items()}}
            if group == 'attitude':
                norms = np.linalg.norm(values, axis=1)
                diagnostics[group]['raw_quaternion_norm'] = distribution(norms[np.isfinite(norms)])
        result = compute_residuals(grid, row_valid, traces)
        mask = window_residual_mask(result['joint_valid'], starts)
        # Old invalid windows are not rescued by checking only140 frames.
        mask[~original_supervised] = False
        whole = original_supervised & mask[:, 10:].all(axis=1)
        summary = summarize(result)
        summary.update(case_uid=uid, split='train', original_windows=len(starts),
            originally_supervised_windows=int(original_supervised.sum()),
            windows_with_all140_joint_residual_rows_valid=int(whole.sum()),
            original_cap64_window_residual_contributions=int(mask.sum()),
            native_topics=diagnostics)
        summaries[uid] = summary
        windows_total += len(starts)
        np.savez_compressed(args.output / (key + '.npz'), timestamp_us=grid, **result,
            window_start_row=starts, original_cache_index=original_ids, original_supervised_valid=original_supervised,
            window_residual_mask=mask, all140_valid=whole)
        all_valid.append(result)
        save(args.output / (key + '.json'), summary)
        print('KINEMATIC_NORMAL_CASE', uid, 'vertical', summary['vertical_valid'], 'attitude', summary['attitude_valid'],
            'joint140_windows', int(whole.sum()), '/', len(starts), flush=True)
    for path, expected in inputs.items():
        if sha(path) != expected:
            raise RuntimeError('Read-only source identity changed during diagnosis: ' + path)
    combined = {key: np.concatenate([r[key] for r in all_valid]) for key in all_valid[0]}
    summary = {'status': 'COMPLETE_FIXED_NORMAL_ANCHOR_KINEMATIC_FEASIBILITY_CHECK',
        'completed_at': datetime.now(timezone.utc).isoformat(), 'horizon_us': HORIZON_US,
        'case_count': len(selected), 'original_windows': windows_total, 'overall': summarize(combined),
        'cases': summaries, 'normalization_fitted': False, 'classifier_training_or_inference': False,
        'test_or_Real_read': False, 'runtime_model_or_existing_data_modified': False, 'inputs': inputs,
        'script_sha256': sha(__file__),
        'limits': ['Causal publication-replay stream consistency, not synchronized observation-space truth.',
            'Native gyro is already an averaged sampled reading, not independent high-rate inertial ground truth.',
            'No expected normal residual magnitude acceptance threshold or fault sensitivity evidence.',
            'Small right-vs-left difference in near-static data cannot certify coordinates.',
            'Invalid intervals retained; no label changes, fitted thresholds, new data statistics or model activation.']}
    save(args.output / 'summary.json', summary)
    print(json.dumps({'status': summary['status'], 'case_count': len(selected), 'overall': summary['overall']}, indent=2), flush=True)


if __name__ == '__main__':
    main()
