"""Continuous publication-stream adapter for the frozen200ms residual formulas.

The historical independent-window prototype stays unmodified. All state and
validity are on the original continuous grid, never restarted at sliding windows.
"""
from collections import deque
import time
import numpy as np
from scipy.spatial.transform import Rotation
from validate_kinematic_residuals import (Trace, HORIZON_US, PREFIX, INPUT_INVALID,
    TOPIC_INVALID, RESET, INVALID_QUATERNION, normalize_quaternion_wxyz,
    rotation_wxyz, integrate_rotation, compute_residuals)

COLUMNS = ['rh_signed_m', 'rR_x_rad', 'rR_y_rad', 'rR_z_rad']
GROUPS = ('baro', 'velocity', 'attitude', 'gyro')


class LocalSupportTrace(Trace):
    """Identical support semantics, avoid a full native-array scan per grid."""
    def support(self, start, end):
        first, last = self.point(start), self.point(end)
        if end <= start:
            raise ValueError('Positive fixed interval required')
        lo = np.searchsorted(self.available_us, start, side='right')
        hi = np.searchsorted(self.available_us, end, side='left')
        middle = self.available_us[lo:hi]
        knots = np.r_[start, np.unique(middle), end].astype(np.int64)
        indices = np.searchsorted(self.available_us, knots[:-1], side='right') - 1
        durations = np.diff(knots) / 1e6
        if not self.valid[indices].all() or not np.isfinite(self.values[indices]).all():
            raise ValueError('interior_invalid_latest')
        if (knots[1:] - self.observation_us[indices] > self.max_age_us).any():
            raise ValueError('interior_stale_gap')
        support = np.unique(np.r_[indices, last])
        changed = any(np.any(np.asarray(v)[support] != np.asarray(v)[first]) for v in self.counters.values())
        return indices, durations, support, changed


def compute_continuous(grid, input_valid, traces):
    """One computation per original grid; only true buffer origin has PREFIX."""
    optimized = {g: LocalSupportTrace(t.observation_us, t.available_us, t.values,
        t.max_age_us, t.counters, t.valid) for g, t in traces.items()}
    # Historical function computes per-stream rows correctly; never call its
    # independent window_residual_mask here. Crop before slicing windows.
    return compute_residuals(grid, input_valid, optimized)


def continuous_window_masks(joint_valid, starts, frames=150):
    mask = np.asarray(joint_valid)
    starts = np.asarray(starts)
    if mask.ndim != 1 or mask.dtype != bool or starts.dtype != np.int64 or frames != 150:
        raise ValueError('Original150-frame window IDs and boolean continuous mask required')
    if (starts < 0).any() or (starts + frames > len(mask)).any():
        raise ValueError('Window outside original retained crop')
    return mask[starts[:, None] + np.arange(frames)]


class ContinuousResidualStream:
    """Actual arrival-driven bounded200ms buffer for replay/native adapters.

    Caller supplies original base input-valid grid including unavailable points.
    Clock reversals and retroactive arrival are rejected, not sorted/repaired.
    """
    def __init__(self, limits, *, origin_us):
        if set(limits) != set(GROUPS) or not isinstance(origin_us, int) or origin_us < 0:
            raise ValueError('Explicit four budgets and fixed-grid origin required')
        self.limits = dict(limits)
        self.origin = origin_us
        self.rows = 0
        self.last_grid = -1
        self.last_arrival = -1
        self.buffers = {g: deque() for g in GROUPS}
        self.latest_observation = {}
        self.input_history = deque(maxlen=11)
        self.counter_names = {}

    def push(self, group, observation_us, available_us, values, *, counters=None, valid=True):
        if group not in self.buffers:
            raise ValueError('Unknown residual topic')
        for t in [observation_us, available_us]:
            if isinstance(t, bool) or not isinstance(t, (int, np.integer)) or t < 0:
                raise ValueError('Explicit integer clocks required')
        if observation_us <= 0 or observation_us > available_us or available_us < self.last_arrival or available_us <= self.last_grid:
            raise ValueError('Invalid/reversed/retroactive native arrival')
        if observation_us < self.latest_observation.get(group, 0):
            raise ValueError('Observation clock reversal')
        width = {'baro': 1, 'velocity': 1, 'attitude': 4, 'gyro': 3}[group]
        values = np.asarray(values, dtype=np.float64)
        counters = {} if counters is None else dict(counters)
        if values.shape != (width,) or type(valid) not in [bool, np.bool_]:
            raise ValueError('Explicit native value width/valid flag required')
        if group in self.counter_names and self.counter_names[group] != set(counters):
            raise ValueError('Reset marker schema changed within native stream')
        self.counter_names[group] = set(counters)
        self.latest_observation[group] = int(observation_us)
        self.buffers[group].append((int(observation_us), int(available_us), values.copy(), counters, bool(valid)))
        self.last_arrival = int(available_us)

    def emit(self, grid_us, *, input_valid):
        started = time.perf_counter_ns()
        if grid_us != self.origin + self.rows * 20000 or grid_us < self.last_arrival or type(input_valid) not in [bool, np.bool_]:
            raise ValueError('Original20ms grid and current input validity required')
        self.input_history.append(bool(input_valid))
        values = np.full(4, np.nan)
        reasons = np.zeros(2, dtype=np.uint16)
        ages = np.full((4, 2), -1, dtype=np.int64)
        start = int(grid_us) - HORIZON_US
        if self.rows < 10:
            reasons[:] = PREFIX
        elif not all(self.input_history):
            reasons[:] = INPUT_INVALID
        else:
            traces = {}
            for g in GROUPS:
                buf = self.buffers[g]
                # Retain the last predecessor at/before start. Remove older
                # duplicates only when replaced; never fall back past invalid.
                while len(buf) > 1 and buf[1][1] <= start:
                    buf.popleft()
                if not buf:
                    continue
                traces[g] = LocalSupportTrace(np.array([r[0] for r in buf], dtype=np.int64),
                    np.array([r[1] for r in buf], dtype=np.int64), np.stack([r[2] for r in buf]),
                    self.limits[g], {k: np.array([r[3][k] for r in buf]) for k in self.counter_names[g]},
                    np.array([r[4] for r in buf], dtype=bool))
            for col, groups in enumerate([('baro', 'velocity'), ('attitude', 'gyro')]):
                try:
                    info = {}
                    for g in groups:
                        if g not in traces:
                            raise ValueError('missing_topic')
                        t = traces[g]
                        idx, dt, support, changed = t.support(start, int(grid_us))
                        a, b = t.point(start), t.point(int(grid_us))
                        ages[GROUPS.index(g)] = [start - t.observation_us[a], grid_us - t.observation_us[b]]
                        if changed:
                            reasons[col] |= RESET
                        info[g] = (a, b, idx, dt, support)
                    if reasons[col]:
                        continue
                    if col == 0:
                        a, b, *_ = info['baro']
                        _, _, idx, dt, _ = info['velocity']
                        values[0] = traces['baro'].values[b, 0] - traces['baro'].values[a, 0] + np.sum(traces['velocity'].values[idx, 0] * dt)
                    else:
                        a, b, _, _, support = info['attitude']
                        try:
                            normalize_quaternion_wxyz(traces['attitude'].values[support])
                            r0, r1 = rotation_wxyz(traces['attitude'].values[[a, b]]).as_matrix()
                        except ValueError:
                            reasons[col] |= INVALID_QUATERNION
                            continue
                        _, _, idx, dt, _ = info['gyro']
                        delta = integrate_rotation(traces['gyro'].values[idx], dt)
                        values[1:] = Rotation.from_matrix((r0 @ delta).T @ r1).as_rotvec()
                except ValueError:
                    reasons[col] |= TOPIC_INVALID
        valid = np.r_[reasons[0] == 0, np.repeat(reasons[1] == 0, 3)]
        if not np.isfinite(values[valid]).all():
            raise FloatingPointError('Valid continuous residual overflow')
        self.rows += 1
        self.last_grid = int(grid_us)
        # Bound memory also during warmup or invalid base spans.
        for buf in self.buffers.values():
            while len(buf) > 1 and buf[1][1] <= start:
                buf.popleft()
        return {'timestamp_us': int(grid_us), 'residual': values, 'valid': valid,
            'reason': reasons, 'endpoint_age_us': ages,
            'compute_ns': time.perf_counter_ns() - started}
