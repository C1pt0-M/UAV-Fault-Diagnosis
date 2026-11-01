"""Past-only input primitives for an explicitly configured future data version.

No implicit freshness threshold, interpolation, zero fill, training admission or
deployment change. Callers must supply observation AND availability timestamps
on a demonstrated common integer-microsecond clock.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def integer_times(values, name):
    arr = np.asarray(values)
    if arr.ndim != 1 or arr.dtype != np.dtype('int64'):
        raise ValueError(name + ' must be a one-dimensional int64 microsecond array')
    if (arr < 0).any() or (arr[1:] < arr[:-1]).any():
        raise ValueError(name + ' has negative/reversed timestamps; no implicit sorting')
    return arr


@dataclass(frozen=True)
class SampledTopic:
    values: np.ndarray
    valid: np.ndarray
    source_index: np.ndarray
    observation_us: np.ndarray
    available_us: np.ndarray
    age_us: np.ndarray


def sample_available(observation_us, available_us, values, grid_us, *, max_age_us):
    """Hold last arrived record; reject future, stale or invalid latest records.

    At duplicate arrival times the last recorded arrival wins. Nonfinite latest
    data invalidates the row rather than falling back to older finite values.
    Source identity remains recorded even when too old/nonfinite; returned values
    are NaN for invalid rows. This does not certify the supplied clock provenance.
    """
    if isinstance(max_age_us, (bool, np.bool_)) or not isinstance(max_age_us, (int, np.integer)) or max_age_us < 0:
        raise ValueError('Explicit nonnegative integer max_age_us required; no default')
    obs = integer_times(observation_us, 'observation_us')
    arrivals = integer_times(available_us, 'available_us')
    grid = integer_times(grid_us, 'grid_us')
    if len(grid) and (np.diff(grid) <= 0).any():
        raise ValueError('grid_us must be strictly increasing')
    data = np.asarray(values, dtype=np.float64)
    if data.ndim != 2 or data.shape[0] != len(obs) or len(obs) != len(arrivals) or data.shape[1] == 0:
        raise ValueError('Topic values/timestamp shapes differ or no channels declared')
    if (obs > arrivals).any():
        raise ValueError('Observation occurs after availability; clock mapping not established')
    n, channels = len(grid), data.shape[1]
    out = np.full((n, channels), np.nan, dtype=np.float64)
    selected = np.full(n, -1, dtype=np.int64)
    selected_obs = selected.copy(); selected_arrival = selected.copy(); age = selected.copy()
    valid = np.zeros(n, dtype=bool)
    if not len(obs):
        return SampledTopic(out, valid, selected, selected_obs, selected_arrival, age)
    selected[:] = np.searchsorted(arrivals, grid, side='right') - 1
    found = selected >= 0
    rows = np.flatnonzero(found); indices = selected[found]
    selected_obs[found] = obs[indices]; selected_arrival[found] = arrivals[indices]
    age[found] = grid[found] - obs[indices]
    good = (age[found] <= max_age_us) & np.isfinite(data[indices]).all(axis=1)
    valid[rows[good]] = True
    out[rows[good]] = data[indices[good]]
    return SampledTopic(out, valid, selected, selected_obs, selected_arrival, age)


def window_input_valid(row_valid, starts, *, window_frames):
    """Keep grid/indices; report whether every input row in each window is valid."""
    rows = np.asarray(row_valid); positions = np.asarray(starts)
    if rows.ndim != 1 or rows.dtype != np.bool_ or positions.ndim != 1 or positions.dtype != np.int64:
        raise ValueError('Boolean row mask and int64 window starts required')
    if isinstance(window_frames, bool) or not isinstance(window_frames, int) or window_frames <= 0:
        raise ValueError('Positive integer window_frames required')
    if (positions < 0).any() or (positions > len(rows) - window_frames).any():
        raise ValueError('Window outside unchanged input timeline')
    invalid_prefix = np.r_[0, np.cumsum(~rows, dtype=np.int64)]
    return (invalid_prefix[positions + window_frames] - invalid_prefix[positions]) == 0


def canonical_window(matrix, columns, row_valid, *, mean, std):
    """Strict shared window transform; reject invalid rather than silently fill.

    Input columns have already been mapped to the declared canonical channels.
    Absolute barometric altitude occupies baro_alt_relative until this transform.
    Compute derivatives/relative altitude in float64, cast raw window to float32,
    then apply float32 training-only statistics. No model call is performed.
    """
    data = np.asarray(matrix, dtype=np.float64)
    names = list(columns); mask = np.asarray(row_valid)
    if data.ndim != 2 or not len(data) or data.shape[1] != len(names) or len(set(names)) != len(names):
        raise ValueError('Invalid canonical matrix or duplicate/missing channel names')
    if mask.dtype != np.bool_ or mask.shape != (len(data),) or not mask.all():
        raise ValueError('Unavailable input window; do not infer normal/fault')
    required = {'accel_x', 'accel_y', 'accel_z', 'acc_norm', 'baro_alt_relative'}
    if not required.issubset(names):
        raise ValueError('Canonical acceleration/barometric channels missing')
    out = data.copy()
    acceleration = out[:, [names.index(k) for k in ['accel_x', 'accel_y', 'accel_z']]]
    if not np.isfinite(acceleration).all():
        raise ValueError('Nonfinite acceleration')
    out[:, names.index('acc_norm')] = np.sqrt(np.sum(acceleration * acceleration, axis=1))
    baro = names.index('baro_alt_relative'); out[:, baro] -= out[0, baro]
    mu, sigma = np.asarray(mean, dtype=np.float64), np.asarray(std, dtype=np.float64)
    if mu.shape != (len(names),) or sigma.shape != mu.shape or not np.isfinite(mu).all() or not np.isfinite(sigma).all() or (sigma <= 0).any():
        raise ValueError('Invalid explicit source-training normalization')
    if not np.isfinite(out).all():
        raise ValueError('Nonfinite required features; no nan_to_num')
    with np.errstate(over='raise', invalid='raise', divide='raise'):
        try:
            raw = out.astype(np.float32)
            normalized = (raw - mu.astype(np.float32)[None, :]) / sigma.astype(np.float32)[None, :]
        except FloatingPointError as exc:
            raise ValueError('Float32 transform overflow/invalid statistics') from exc
    if not np.isfinite(normalized).all():
        raise ValueError('Nonfinite normalized window')
    return normalized.T.copy()
