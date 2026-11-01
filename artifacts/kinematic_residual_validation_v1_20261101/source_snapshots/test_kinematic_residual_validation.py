from pathlib import Path
import sys

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts/research'))
from validate_kinematic_residuals import (
    Trace, compute_residuals, integrate_rotation, normalize_quaternion_wxyz,
    rotation_wxyz, window_residual_mask, PREFIX, RESET, TOPIC_INVALID,
    INVALID_QUATERNION, INPUT_INVALID,
)


def sample_traces(n=21):
    grid = 1000000 + np.arange(n, dtype=np.int64) * 20000
    traces = {
        'baro': Trace(grid, grid, (100 + 2 * (grid - grid[0]) / 1e6)[:, None], 100000),
        'velocity': Trace(grid, grid, np.full((n, 1), -2.), 140000),
        'gyro': Trace(grid, grid, np.zeros((n, 3)), 40000),
        'attitude': Trace(grid, grid, np.tile([1., 0., 0., 0.], (n, 1)), 80000),
    }
    return grid, traces


def test_constant_climb_integral_sign_and_no_prefix_zero_fill():
    grid, traces = sample_traces()
    result = compute_residuals(grid, np.ones(len(grid), dtype=bool), traces)
    assert np.isnan(result['vertical_m'][:10]).all()
    assert (result['reason'][:10] == PREFIX).all()
    assert np.max(np.abs(result['vertical_m'][10:])) < 1e-12
    assert np.allclose(result['wrong_vertical_sign_m'][10:], .8)
    assert np.max(np.abs(result['attitude_rad'][10:])) == 0


def test_nonidentity_multiaxis_rotation_right_multiplication_known_answer():
    grid, traces = sample_traces()
    omega = np.tile([.7, -.3, .5], (len(grid), 1))
    omega[5:] = [-.2, .8, .4]
    matrices = [Rotation.from_euler('xyz', [.6, -.4, .8]).as_matrix()]
    for w in omega[:-1]:
        matrices.append(matrices[-1] @ Rotation.from_rotvec(w * .02).as_matrix())
    xyzw = Rotation.from_matrix(np.asarray(matrices)).as_quat()
    traces['gyro'].values = omega
    traces['attitude'].values = xyzw[:, [3, 0, 1, 2]]
    result = compute_residuals(grid, np.ones(len(grid), dtype=bool), traces)
    assert np.max(np.linalg.norm(result['attitude_rad'][10:], axis=1)) < 1e-12
    assert np.max(np.linalg.norm(result['left_multiplication_rad'][10:], axis=1)) > .01


def test_quaternion_double_cover_and_normalization():
    q = np.array([2., .3, -.5, .7])
    assert np.isclose(np.linalg.norm(normalize_quaternion_wxyz(q)), 1)
    assert np.allclose(rotation_wxyz(q).as_matrix(), rotation_wxyz(-q).as_matrix())
    grid, traces = sample_traces()
    traces['attitude'].values[::2] *= -1
    result = compute_residuals(grid, np.ones(len(grid), dtype=bool), traces)
    assert np.max(np.abs(result['attitude_rad'][10:])) == 0


@pytest.mark.parametrize('q', [[0., 0., 0., 0.], [1e-10, 0., 0., 0.], [np.nan, 0., 0., 0.]])
def test_invalid_quaternion_never_clamped_into_valid_rotation(q):
    with pytest.raises(ValueError, match='quaternion|Quaternion'):
        normalize_quaternion_wxyz(q)


def test_quaternion_norm_overflow_is_invalid():
    with np.errstate(over='ignore'):
        with pytest.raises(ValueError, match='quaternion'):
            normalize_quaternion_wxyz([1e308, 1e308, 1e308, 1e308])


def test_invalid_latest_quaternion_masks_interval_not_older_fallback():
    grid, traces = sample_traces()
    traces['attitude'].values[10] = 0
    result = compute_residuals(grid, np.ones(len(grid), dtype=bool), traces)
    assert not result['attitude_valid'][10] and result['reason'][10, 1] & INVALID_QUATERNION
    assert np.isnan(result['attitude_rad'][10]).all()


def test_native_irregular_availability_integral_does_not_backdate_late_sample():
    available = np.array([1000000, 1055000, 1150000, 1200000], dtype=np.int64)
    observed = np.array([1000000, 1020000, 1140000, 1190000], dtype=np.int64)
    trace = Trace(observed, available, np.array([[1.], [3.], [2.], [5.]]), 200000)
    idx, dt, _, changed = trace.support(1000000, 1200000)
    integral = float(np.sum(trace.values[idx, 0] * dt))
    assert np.isclose(integral, 1 * .055 + 3 * .095 + 2 * .05)
    assert not np.isclose(integral, 1 * .02 + 3 * .12 + 2 * .06)
    assert not changed


def test_interior_gap_or_invalid_latest_not_rescued_by_fresh_endpoints():
    available = np.array([1000000, 1190000, 1200000], dtype=np.int64)
    trace = Trace(available, available, np.ones((3, 1)), 40000)
    with pytest.raises(ValueError, match='stale_gap'):
        trace.support(1000000, 1200000)
    trace = Trace(available, available, np.array([[1.], [np.nan], [2.]]), 300000)
    with pytest.raises(ValueError, match='interior_invalid'):
        trace.support(1000000, 1200000)


def test_duplicate_arrival_last_invalid_is_not_replaced():
    times = np.array([1000000, 1100000, 1100000], dtype=np.int64)
    trace = Trace(times, times, np.array([[1.], [2.], [np.nan]]), 200000)
    with pytest.raises(ValueError, match='latest'):
        trace.point(1100000)


def test_reset_crossing_and_original_input_mask_stay_explicit():
    grid, traces = sample_traces()
    traces['attitude'].counters = {'quat_reset_counter': np.r_[np.zeros(10), np.ones(len(grid) - 10)]}
    result = compute_residuals(grid, np.ones(len(grid), dtype=bool), traces)
    assert result['reason'][10, 1] & RESET
    assert result['vertical_valid'][10] and not result['joint_valid'][10]
    valid = np.ones(len(grid), dtype=bool); valid[5] = False
    result = compute_residuals(grid, valid, traces)
    assert np.all(result['reason'][10:16] & INPUT_INVALID)


def test_future_prefix_invariance():
    grid, traces = sample_traces(31)
    before = compute_residuals(grid, np.ones(len(grid), dtype=bool), traces)
    for trace in traces.values():
        trace.values[21:] = 100
    after = compute_residuals(grid, np.ones(len(grid), dtype=bool), traces)
    for key in before:
        assert np.array_equal(before[key][:21], after[key][:21], equal_nan=True), key


def test_each_original_window_excludes_own_first_ten_frames():
    mask = window_residual_mask(np.ones(200, dtype=bool), np.array([0, 30], dtype=np.int64))
    assert not mask[:, :10].any() and mask[:, 10:].all()
    assert mask.sum() == 280
    with pytest.raises(ValueError, match='outside'):
        window_residual_mask(np.ones(200, dtype=bool), np.array([60], dtype=np.int64))


def test_reject_clock_reversal_and_future_observation():
    with pytest.raises(ValueError, match='Clock'):
        Trace(np.array([100, 90]), np.array([100, 110]), np.ones((2, 1)), 40)
    with pytest.raises(ValueError, match='Clock'):
        Trace(np.array([101]), np.array([100]), np.ones((1, 1)), 40)


def test_rotation_integral_uses_actual_intervals():
    omega = np.array([[1., 0., 0.], [1., 0., 0.]])
    actual = integrate_rotation(omega, np.array([.03, .17]))
    assert np.allclose(actual, Rotation.from_rotvec([.2, 0., 0.]).as_matrix())
    with pytest.raises(ValueError, match='positive'):
        integrate_rotation(omega, np.array([.03, 0.]))
