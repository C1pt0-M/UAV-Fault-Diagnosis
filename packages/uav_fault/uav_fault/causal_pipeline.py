"""Explicit-clock causal input builder and stateful replay/runtime adapter.

No model, automatic clock conversion, freshness defaults or deployment wiring.
ULog adapters must declare publication-as-availability proxy. Live clients must
supply aligned observation/receive times, not periodic snapshot emission time.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np

from .causal_input import canonical_window, sample_available, window_input_valid

# Split combined IMU into observation groups because acceleration has its own clock.
GROUPS = {
    'controls': ('actuator_controls_0', 'timestamp_sample',
        tuple((f'actuator_ctrl_{i + 1}', f'control[{i}]') for i in range(4))),
    'outputs': ('actuator_outputs', 'timestamp',
        tuple((f'actuator_out_{i + 1}', f'output[{i}]') for i in range(4))),
    'gyro': ('sensor_combined', 'timestamp',
        tuple((f'gyro_{axis}', f'gyro_rad[{i}]') for i, axis in enumerate('xyz'))),
    'accel': ('sensor_combined', 'accelerometer_timestamp_relative',
        tuple((f'accel_{axis}', f'accelerometer_m_s2[{i}]') for i, axis in enumerate('xyz'))),
    'baro': ('vehicle_air_data', 'timestamp_sample', (('baro_alt_relative', 'baro_alt_meter'),)),
    'attitude': ('vehicle_attitude', 'timestamp_sample',
        tuple((f'att_q_{axis}', f'q[{i}]') for i, axis in enumerate('wxyz'))),
    'velocity': ('vehicle_local_position', 'timestamp_sample',
        tuple((f'vel_{axis}', f'v{axis}') for axis in 'xyz')),
    'mag': ('vehicle_magnetometer', 'timestamp_sample',
        tuple((f'mag_{axis}', f'magnetometer_ga[{i}]') for i, axis in enumerate('xyz'))),
}
MISSING, STALE, NONFINITE = 1, 2, 4


@dataclass(frozen=True)
class TopicTrace:
    observation_us: np.ndarray
    available_us: np.ndarray
    values: np.ndarray


def validate_layout(columns, limits):
    names = list(columns)
    expected = [name for _, _, fields in GROUPS.values() for name, _ in fields] + ['acc_norm']
    if len(names) != 26 or len(set(names)) != 26 or set(names) != set(expected):
        raise ValueError('Exact canonical 26-channel layout required')
    if set(limits) != set(GROUPS):
        raise ValueError('Every observation group requires an explicit age budget')
    for value in limits.values():
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < 0:
            raise ValueError('Nonnegative integer age budgets required')
    return {group: [names.index(name) for name, _ in fields] for group, (_, _, fields) in GROUPS.items()}


def finish_frame_values(matrix, columns):
    accel = matrix[:, [columns.index('accel_' + axis) for axis in 'xyz']]
    with np.errstate(over='ignore', invalid='ignore'):
        matrix[:, columns.index('acc_norm')] = np.sqrt(np.sum(accel * accel, axis=1))
    return np.isfinite(matrix).all(axis=1)


def batch_frames(traces, grid_us, columns, limits):
    layout = validate_layout(columns, limits)
    if set(traces) != set(GROUPS):
        raise ValueError('Missing/extra required observation groups')
    matrix = np.full((len(grid_us), len(columns)), np.nan, dtype=np.float64)
    provenance = {}
    for group, trace in traces.items():
        result = sample_available(trace.observation_us, trace.available_us, trace.values, grid_us, max_age_us=limits[group])
        if result.values.shape[1] != len(layout[group]):
            raise ValueError('Topic channel width differs: ' + group)
        matrix[:, layout[group]] = result.values
        reason = np.zeros(len(grid_us), dtype=np.uint8)
        found = result.source_index >= 0
        reason[~found] |= MISSING
        reason[found & (result.age_us > limits[group])] |= STALE
        rows = np.flatnonzero(found)
        reason[rows[~np.isfinite(trace.values[result.source_index[found]]).all(axis=1)]] |= NONFINITE
        provenance[group] = {'source_index': result.source_index, 'observation_us': result.observation_us,
            'available_us': result.available_us, 'age_us': result.age_us, 'reason': reason, 'valid': result.valid}
    valid = finish_frame_values(matrix, list(columns))
    for group in provenance.values(): valid &= group['valid']
    return matrix, valid, provenance


def batch_windows(matrix, row_valid, starts, columns, *, mean, std, window_frames=150):
    valid = window_input_valid(row_valid, starts, window_frames=window_frames)
    result = {}
    for start in starts[valid]:
        end = int(start) + window_frames
        result[int(start)] = canonical_window(matrix[start:end], columns, row_valid[start:end], mean=mean, std=std)
    return valid, result


class CausalInputStream:
    """Feed actual message arrivals then advance an explicit fixed-grid watermark.

    push requires source/receive clock provenance from its caller. It rejects
    messages backdated at/before a finalized grid instant: lateness must be
    represented by actual later availability, never retroactively rewrite frames.
    All frames including unavailable ones remain in the sliding window.
    """
    def __init__(self, columns, limits, *, origin_us, mean, std, step_us=20000, window_frames=150, stride_frames=5):
        self.columns = list(columns); self.layout = validate_layout(columns, limits); self.limits = dict(limits)
        for value, name in [(origin_us, 'origin'), (step_us, 'step'), (window_frames, 'window'), (stride_frames, 'stride')]:
            if isinstance(value, bool) or not isinstance(value, int) or value < (0 if name == 'origin' else 1):
                raise ValueError('Invalid integer ' + name)
        self.origin = origin_us; self.step = step_us; self.window_frames = window_frames; self.stride = stride_frames
        self.mean, self.std = np.asarray(mean).copy(), np.asarray(std).copy()
        if self.mean.shape != (26,) or self.std.shape != (26,) or not np.isfinite(self.mean).all() or not np.isfinite(self.std).all() or (self.std <= 0).any():
            raise ValueError('Explicit finite positive source normalization required')
        self.latest = {}; self.counts = {k: 0 for k in GROUPS}; self.last_arrival = -1; self.last_grid = -1; self.rows = 0
        self.buffer = deque(maxlen=window_frames); self.mask_buffer = deque(maxlen=window_frames)

    def push(self, group, observation_us, available_us, values):
        if group not in self.layout:
            raise ValueError('Unknown group')
        for value in [observation_us, available_us]:
            if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 0:
                raise ValueError('Explicit integer source/availability microseconds required')
        if observation_us > available_us or available_us < self.last_arrival or available_us <= self.last_grid:
            raise ValueError('Unaligned, reversed or retroactive message availability')
        old = self.latest.get(group)
        if old is not None and observation_us < old[0]:
            raise ValueError('Observation clock reversal; no sorting/fallback')
        v = np.asarray(values, dtype=np.float64)
        if v.shape != (len(self.layout[group]),):
            raise ValueError('Missing required message fields')
        self.latest[group] = (int(observation_us), int(available_us), v.copy(), self.counts[group])
        self.counts[group] += 1; self.last_arrival = int(available_us)

    def emit(self, grid_us):
        if isinstance(grid_us, (bool, np.bool_)) or not isinstance(grid_us, (int, np.integer)) or grid_us < 0:
            raise ValueError('Explicit nonnegative integer grid microseconds required')
        if grid_us != self.origin + self.rows * self.step or grid_us < self.last_arrival:
            raise ValueError('Wrong grid or future arrivals already ingested')
        row = np.full((1, 26), np.nan); provenance = {}; valid = True
        for group, columns in self.layout.items():
            latest = self.latest.get(group)
            if latest is None:
                meta = {'source_index': -1, 'observation_us': -1, 'available_us': -1, 'age_us': -1, 'reason': MISSING, 'valid': False}
            else:
                obs, available, values, index = latest
                age = int(grid_us) - obs; reason = 0
                if age > self.limits[group]: reason |= STALE
                if not np.isfinite(values).all(): reason |= NONFINITE
                meta = {'source_index': index, 'observation_us': obs, 'available_us': available, 'age_us': age, 'reason': reason, 'valid': reason == 0}
                if not reason: row[0, columns] = values
            valid &= meta['valid']; provenance[group] = meta
        valid = bool(finish_frame_values(row, self.columns)[0] and valid)
        self.buffer.append(row[0].copy()); self.mask_buffer.append(valid); self.rows += 1; self.last_grid = int(grid_us)
        start = self.rows - self.window_frames
        due = start >= 0 and start % self.stride == 0
        usable = bool(due and all(self.mask_buffer))
        window = canonical_window(np.stack(self.buffer), self.columns, np.asarray(self.mask_buffer), mean=self.mean, std=self.std) if usable else None
        return {'grid_us': int(grid_us), 'values': row[0], 'input_valid': valid, 'provenance': provenance,
                'window_due': due, 'window_start_row': start if due else None, 'window_input_valid': usable, 'window': window}
