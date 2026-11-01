from pathlib import Path
import sys
import numpy as np
import pytest
from scipy.spatial.transform import Rotation
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts/research'))
from continuous_kinematic_residuals import (ContinuousResidualStream, compute_continuous, continuous_window_masks)
from validate_kinematic_residuals import (Trace, compute_residuals, PREFIX, RESET)


def fixture(n=220):
    grid = 1000000 + np.arange(n, dtype=np.int64) * 20000
    w = np.tile([.1, -.2, .3], (n, 1)); w[80:] = [-.2, .1, .2]
    matrices = [Rotation.from_euler('xyz', [.3, -.2, .1]).as_matrix()]
    for omega in w[:-1]:
        matrices.append(matrices[-1] @ Rotation.from_rotvec(omega * .02).as_matrix())
    q = Rotation.from_matrix(matrices).as_quat()[:, [3, 0, 1, 2]]
    traces = {'baro': Trace(grid, grid, (10 + 2 * np.arange(n) * .02)[:, None], 100000),
        'velocity': Trace(grid, grid, np.full((n, 1), -2.), 140000),
        'attitude': Trace(grid, grid, q, 80000),
        'gyro': Trace(grid, grid, w, 40000)}
    return grid, traces


def stream_result(grid, traces, base_valid):
    engine = ContinuousResidualStream({g: t.max_age_us for g, t in traces.items()}, origin_us=int(grid[0]))
    positions = {g: 0 for g in traces}
    rows = []
    # Stable availability merge is message-order replay across already monotonic
    # per-topic logs; does not repair/sort any reversed native topic.
    arrivals = sorted((int(t.available_us[i]), order, i, g) for order, (g, t) in enumerate(traces.items()) for i in range(len(t.available_us)))
    pos = 0
    for t, valid in zip(grid, base_valid):
        while pos < len(arrivals) and arrivals[pos][0] <= t:
            available, _, i, g = arrivals[pos]; tr = traces[g]
            engine.push(g, int(tr.observation_us[i]), available, tr.values[i],
                counters={k: v[i] for k, v in tr.counters.items()}, valid=bool(tr.valid[i]))
            pos += 1
        rows.append(engine.emit(int(t), input_valid=bool(valid)))
    return rows, engine


def test_continuous_matches_frozen_formulas_and_does_not_reset_per_window():
    grid, traces = fixture(); base = np.ones(len(grid), dtype=bool)
    old = compute_residuals(grid, base, traces); new = compute_continuous(grid, base, traces)
    for key in old:
        np.testing.assert_array_equal(old[key], new[key])
    masks = continuous_window_masks(new['joint_valid'], np.array([0, 20, 40], dtype=np.int64))
    assert not masks[0, :10].any() and masks[0, 10:].all()
    assert masks[1:].all()
    # The original crop can start after established log history.
    assert continuous_window_masks(new['joint_valid'][20:], np.array([0], dtype=np.int64)).all()


def test_actual_bounded_arrival_stream_matches_batch_noncommuting_rotation():
    grid, traces = fixture(); base = np.ones(len(grid), dtype=bool)
    old = compute_continuous(grid, base, traces); rows, engine = stream_result(grid, traces, base)
    values = np.array([r['residual'] for r in rows])
    np.testing.assert_allclose(values[:, 0], old['vertical_m'], rtol=0, atol=1e-12, equal_nan=True)
    np.testing.assert_allclose(values[:, 1:], old['attitude_rad'], rtol=0, atol=1e-11, equal_nan=True)
    np.testing.assert_array_equal(np.array([r['reason'] for r in rows]), old['reason'])
    assert all(len(buf) <= 12 for buf in engine.buffers.values())


def test_reset_input_gap_and_invalid_quaternion_do_not_become_normal_zeros():
    grid, traces = fixture(); base = np.ones(len(grid), dtype=bool); base[100] = False
    traces['attitude'].counters = {'quat_reset_counter': np.r_[np.zeros(60), np.ones(len(grid) - 60)]}
    traces['attitude'].values[130] = 0
    old = compute_continuous(grid, base, traces); rows, _ = stream_result(grid, traces, base)
    np.testing.assert_array_equal(np.array([r['reason'] for r in rows]), old['reason'])
    assert old['reason'][60, 1] & RESET
    assert not old['joint_valid'][100:111].any()
    assert np.isnan(rows[130]['residual'][1:]).all()


def test_no_future_prefix_change_and_latest_invalid_duplicate_refused():
    grid, traces = fixture(80); base = np.ones(len(grid), dtype=bool)
    before = compute_continuous(grid, base, traces)
    for tr in traces.values():tr.values[60:] = 100
    after = compute_continuous(grid, base, traces)
    for key in before:np.testing.assert_array_equal(before[key][:60], after[key][:60])
    engine = ContinuousResidualStream({g: t.max_age_us for g, t in traces.items()}, origin_us=1000000)
    engine.push('baro', 1000000, 1000000, [1.])
    engine.emit(1000000, input_valid=True)
    with pytest.raises(ValueError, match='retroactive'):engine.push('baro', 1000000, 1000000, [2.])


def test_no_restart_ideal_startup_uses200ms_plus150_residual_samples():
    grid, traces = fixture(170); rows, _ = stream_result(grid, traces, np.ones(len(grid), dtype=bool))
    valid = np.array([r['valid'].all() for r in rows])
    ends = np.flatnonzero(np.convolve(valid.astype(int), np.ones(150, dtype=int), 'valid') == 150) + 149
    assert ends[0] == 159 and grid[ends[0]] - grid[0] == 3180000
    assert grid[149] - grid[0] == 2980000
    assert valid[10:].all() and (np.array([r['reason'] for r in rows])[:10] == PREFIX).all()
