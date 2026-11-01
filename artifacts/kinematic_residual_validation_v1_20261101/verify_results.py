"""Independent direct scalar/quaternion checks on the same12 source logs.

No model/data/stat mutation. Independently recompute fixed valid endpoints,
including largest normal residuals, using quaternion algebra rather than scipy
matrix products used by the validator. No new cases or fault/test/Real data.
"""
from pathlib import Path
import hashlib
import json
import numpy as np
from pyulog import ULog

HERE = Path(__file__).resolve().parent
RESULTS = HERE / 'results_final'


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def multiply(a, b):
    aw, ax, ay, az = a; bw, bx, by, bz = b
    return np.array([aw*bw-ax*bx-ay*by-az*bz,
        aw*bx+ax*bw+ay*bz-az*by,
        aw*by-ax*bz+ay*bw+az*bx,
        aw*bz+ax*by-ay*bx+az*bw])


def exp_quaternion(vector):
    angle = np.linalg.norm(vector)
    if angle < 1e-12:
        q = np.r_[1., .5 * vector]
    else:
        q = np.r_[np.cos(angle / 2), vector * (np.sin(angle / 2) / angle)]
    return q / np.linalg.norm(q)


def log_quaternion(q):
    q = q / np.linalg.norm(q)
    if q[0] < 0:
        q = -q
    norm = np.linalg.norm(q[1:])
    return 2 * q[1:] if norm < 1e-12 else q[1:] * (2 * np.arctan2(norm, q[0]) / norm)


def segments(raw, a, b):
    pub = raw['timestamp'].astype(np.int64)
    middle = sorted(set(pub[(pub > a) & (pub < b)].tolist()))
    times = [a] + middle + [b]
    return [(int(np.searchsorted(pub, start, side='right')) - 1, (end - start) / 1e6)
        for start, end in zip(times[:-1], times[1:])]


def main():
    output = HERE / 'independent_verification.json'
    if output.exists():
        raise FileExistsError('Preserve original verification')
    summary = read(RESULTS / 'summary.json')
    assert summary['case_count'] == 12
    assert not summary['normalization_fitted'] and not summary['classifier_training_or_inference']
    assert not summary['test_or_Real_read'] and not summary['runtime_model_or_existing_data_modified']
    selections = read(RESULTS / 'selection_before_residuals.json')
    details = {}; total_rows = 0; vertical_rows = 0; attitude_rows = 0; joint_rows = 0
    windows = 0; old_eligible = 0; complete = 0
    for uid, row in summary['cases'].items():
        key = uid.replace('/', '__')
        source = Path('D:/UAV-Data/causal_input_v1')
        receipt = read(source / 'case_records' / (key + '.json'))
        assert receipt['split'] == 'train' and receipt['family'] == 'normal'
        assert sha(receipt['raw_ulog']) == receipt['raw_ulog_sha256']
        with np.load(RESULTS / (key + '.npz'), allow_pickle=False) as archive:
            data = {k: archive[k].copy() for k in archive.files}
        with np.load(HERE / 'results' / (key + '.npz'), allow_pickle=False) as first:
            for name, value in data.items():
                assert np.array_equal(value, first[name], equal_nan=True), name
        grid = data['timestamp_us']; valid_h = data['vertical_valid']; valid_q = data['attitude_valid']
        assert np.array_equal(data['joint_valid'], valid_h & valid_q)
        assert np.array_equal(valid_h, data['reason'][:, 0] == 0)
        assert np.array_equal(valid_q, data['reason'][:, 1] == 0)
        assert np.isnan(data['vertical_m'][~valid_h]).all() and np.isnan(data['attitude_rad'][~valid_q]).all()
        assert not data['window_residual_mask'][:, :10].any()
        for i, start in enumerate(data['window_start_row']):
            expected = data['joint_valid'][start + 10:start + 150] if data['original_supervised_valid'][i] else np.zeros(140, dtype=bool)
            assert np.array_equal(data['window_residual_mask'][i, 10:], expected)
        assert int(data['all140_valid'].sum()) == row['windows_with_all140_joint_residual_rows_valid']
        for name, values, valid in [('vertical_abs_m', np.abs(data['vertical_m']), valid_h),
                ('attitude_norm_rad', np.linalg.norm(data['attitude_rad'], axis=1), valid_q)]:
            selected = values[valid]
            assert len(selected) == row[name]['count']
            assert float(np.median(selected)) == row[name]['median']
            assert float(np.quantile(selected, .95)) == row[name]['p95']
        log = ULog(receipt['raw_ulog'], message_name_filter_list=['vehicle_air_data', 'vehicle_local_position', 'vehicle_attitude', 'sensor_combined'])
        topics = {d.name: d.data for d in log.data_list if d.multi_id == 0}
        both = np.flatnonzero(data['joint_valid'])
        h_peak = int(np.nanargmax(np.abs(data['vertical_m'])))
        q_peak = int(np.nanargmax(np.linalg.norm(data['attitude_rad'], axis=1)))
        points = sorted(set([int(both[0]), int(both[len(both)//2]), int(both[-1]), h_peak, q_peak]))
        checks = []
        for index in points:
            a = int(grid[index]) - 200000; b = int(grid[index])
            check = {'row': index, 'crop_elapsed_s': float((grid[index] - grid[0]) / 1e6)}
            if valid_h[index]:
                raw_h = topics['vehicle_air_data']; times = raw_h['timestamp'].astype(np.int64)
                ia = int(np.searchsorted(times, a, side='right')) - 1; ib = int(np.searchsorted(times, b, side='right')) - 1
                height_change = float(raw_h['baro_alt_meter'][ib]) - float(raw_h['baro_alt_meter'][ia])
                raw_v = topics['vehicle_local_position']
                integral = sum(float(raw_v['vz'][j]) * dt for j, dt in segments(raw_v, a, b))
                expected = height_change + integral
                error = abs(expected - float(data['vertical_m'][index]))
                assert error < 1e-12
                check.update(vertical_m=expected, independent_vertical_abs_difference=error)
            if valid_q[index]:
                raw_q = topics['vehicle_attitude']; times = raw_q['timestamp'].astype(np.int64)
                ia = int(np.searchsorted(times, a, side='right')) - 1; ib = int(np.searchsorted(times, b, side='right')) - 1
                q0 = np.array([raw_q[f'q[{i}]'][ia] for i in range(4)], dtype=np.float64); q0 /= np.linalg.norm(q0)
                qt = np.array([raw_q[f'q[{i}]'][ib] for i in range(4)], dtype=np.float64); qt /= np.linalg.norm(qt)
                right = q0.copy(); left = q0.copy(); gyro = topics['sensor_combined']
                for j, dt in segments(gyro, a, b):
                    omega = np.array([gyro[f'gyro_rad[{i}]'][j] for i in range(3)], dtype=np.float64)
                    dq = exp_quaternion(omega * dt)
                    right = multiply(right, dq); left = multiply(dq, left)
                right_error = log_quaternion(multiply(right * np.array([1, -1, -1, -1]), qt))
                left_error = log_quaternion(multiply(left * np.array([1, -1, -1, -1]), qt))
                error = float(np.max(np.abs(right_error - data['attitude_rad'][index])))
                assert error < 1e-11
                assert np.max(np.abs(left_error - data['left_multiplication_rad'][index])) < 1e-11
                check.update(attitude_norm_rad=float(np.linalg.norm(right_error)), independent_rotation_component_max_difference=error)
            checks.append(check)
        details[uid] = {'checks': checks, 'first_and_final_runs_exact_equal_all_arrays': True}
        total_rows += len(grid); vertical_rows += int(valid_h.sum()); attitude_rows += int(valid_q.sum())
        joint_rows += int(data['joint_valid'].sum()); windows += len(data['window_start_row'])
        old_eligible += int(data['original_supervised_valid'].sum()); complete += int(data['all140_valid'].sum())
    assert total_rows == 29734 and vertical_rows == 29514 and attitude_rows == 29392 and joint_rows == 29391
    assert windows == 768 and old_eligible == 760 and complete == 707
    for path, expected in selections['inputs'].items():
        assert sha(path) == expected, path
    script = Path('scripts/research/validate_kinematic_residuals.py')
    assert sha(script) == summary['script_sha256']
    result = {'status': 'PASS_INDEPENDENT_KINEMATIC_RESIDUAL_NORMAL_CHECKS', 'cases': details,
        'rows': total_rows, 'vertical_valid_rows': vertical_rows, 'attitude_valid_rows': attitude_rows, 'joint_valid_rows': joint_rows,
        'original_windows': windows, 'original_supervised_windows': old_eligible, 'new_all140_joint_valid_windows': complete,
        'all_original_inputs_unchanged': True, 'normal_outlier_values_preserved': True,
        'earlier_vs_final_execution_all_arrays_exact_equal': True,
        'no_training_inference_or_final_test_Real_access': True, 'verifier_sha256': sha(__file__)}
    with output.open('x', encoding='utf-8') as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
    print(json.dumps({k: v for k, v in result.items() if k != 'cases'}, indent=2))


if __name__ == '__main__':
    main()
