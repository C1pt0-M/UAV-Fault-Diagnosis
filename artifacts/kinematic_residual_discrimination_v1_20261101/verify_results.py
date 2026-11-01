"""Independent saved-output counts, medians and pairwise AUC verification.

Recompute selected window features from the untouched case shards plus saved
residuals; manually reproduce legacy9 with NumPy (not the model method).
No classifier, threshold choice, new native decoding or test/Real reads.
"""
from pathlib import Path
import json
import hashlib
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
RESULTS = HERE / 'results'
DATA = Path('D:/UAV-Data/causal_input_v1')
NEW = ['rh_signed_m', 'rR_x_rad', 'rR_y_rad', 'rR_z_rad', 'rh_abs_m', 'rR_norm_rad']
OLD = ['old_output_std', 'old_output_range', 'old_ctrl_output_gap_mean', 'old_ctrl_output_gap_std',
    'old_effort', 'old_acc_deviation', 'old_gyro_norm', 'old_vertical_consistency', 'old_attitude_energy']
FEATURES = NEW + OLD


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def main():
    output = HERE / 'independent_verification.json'
    if output.exists():
        raise FileExistsError('Preserve completed independent verification')
    summary = read(RESULTS / 'summary.json')
    plan = read(HERE / 'plan.json')
    f = pd.read_csv(RESULTS / 'window_features.csv', keep_default_na=False, float_precision='round_trip')
    if f.common_eligible.dtype != bool:
        raise ValueError('Explicit boolean eligibility expected')
    assert len(f) == 2258 and int(f.common_eligible.sum()) == 2072
    assert f.case_uid.nunique() == 36 and not f.original_cache_index.duplicated().any()
    assert set(f.case_uid) == {r['case_uid'] for r in plan['cases']}
    assert set(f['split']) == {'train', 'val'}
    reference = read(RESULTS / 'diagnostic_reference_train_normal.json')
    normal_reference = f[(f['split'] == 'train') & f.kind.eq('normal') & f.common_eligible]
    assert len(normal_reference) == 340
    for name in FEATURES:
        x = normal_reference[name].to_numpy(dtype=float)
        q1, median, q3 = np.quantile(x, [.25, .5, .75])
        assert median == reference[name]['median'] and q3 - q1 == reference[name]['iqr']
    normal_stages = read(RESULTS / 'normal_case_stage_stability.json')
    for row in normal_stages:
        case = f[f.case_uid.eq(row['case_uid']) & f.common_eligible]
        for label, quarter in [('early', 0), ('late', 3)]:
            x = case.loc[case.crop_quarter.eq(quarter), row['feature']].to_numpy(dtype=float)
            assert len(x) == row[label]['windows']
            if len(x):
                assert float(np.median(x)) == row[label]['median']
                assert float(np.quantile(x, .95)) == row[label]['p95']
        if row['late_minus_early_median'] is not None:
            delta = row['late']['median'] - row['early']['median']
            assert delta == row['late_minus_early_median']
            assert delta / reference[row['feature']]['iqr'] == row['shift_in_train_normal_iqr']
    responses = read(RESULTS / 'matched_fault_responses.json')
    errors = []
    for row in responses:
        case = f[f.case_uid.eq(row['case_uid']) & f.common_eligible]
        for label in ['active', 'pre_fault']:
            x = case.loc[case.window_group.eq(label), row['feature']].to_numpy(dtype=float)
            assert len(x) == row[label]['windows']
            if len(x):
                assert float(np.median(x)) == row[label]['median']
        weighted = 0.; count = 0
        for detail in row['quarter_details']:
            q = detail['crop_quarter']
            x = case.loc[case.window_group.eq('active') & case.crop_quarter.eq(q), row['feature']].to_numpy(dtype=float)
            controls = f[f.kind.eq('normal') & f.common_eligible & f['split'].eq(row['split']) & f.domain.eq(row['domain']) & f.crop_quarter.eq(q)]
            y = controls[row['feature']].to_numpy(dtype=float)
            assert len(x) == detail['fault_windows'] and len(y) == detail['normal_windows']
            if len(x) and len(y):
                # Independent pairwise comparison, no rankdata/roc library.
                difference = x[:, None] - y[None, :]
                auc = float(np.mean((difference > 0) + .5 * (difference == 0)))
                errors.append(abs(auc - detail['auc_larger_fixed_feature_means_fault']))
                assert abs(auc - detail['auc_larger_fixed_feature_means_fault']) < 1e-14
                weighted += len(x) * auc; count += len(x)
        assert count == row['matched_active_windows']
        if count: assert abs(weighted / count - row['matched_crop_quarter_auc']) < 1e-14
    columns = read(DATA / 'input_contract.json')['columns']; index = {k: i for i, k in enumerate(columns)}
    norm = read(DATA / 'norm_stats.json')
    mu = np.asarray(norm['mean'], dtype=np.float32)[:, None]
    std = np.asarray(norm['std'], dtype=np.float32)[:, None]
    max_new = 0.; max_old = 0.; checked = 0
    for uid, case in f.groupby('case_uid', sort=True):
        key = uid.replace('/', '__')
        with np.load(DATA / 'cases' / (key + '.npz'), allow_pickle=False) as payload:
            raw = payload['canonical_rows']; ids = payload['original_cache_index']; refs = payload['reference_state']; valid_labels = payload['row_label_valid']
            source_starts = payload['window_start_row']; supervised = payload['supervised_valid']
            with np.load(RESULTS / (key + '_residuals.npz'), allow_pickle=False) as residual:
                assert np.array_equal(ids, residual['original_cache_index'])
                assert not residual['window_residual_mask'][:, :10].any()
                by_id = {int(v): i for i, v in enumerate(ids)}
                for row in case.itertuples():
                    i = by_id[row.original_cache_index]; start = int(row.window_start_row); end = int(row.window_end_row)
                    assert start == source_starts[i] and end - start == 150
                    assert row.crop_quarter == min(3, 4 * (end - 1) // (len(raw) - 1))
                    assert bool(row.common_eligible) == bool(supervised[i] and residual['window_residual_mask'][i, 10:].all())
                    assert row.qualified_y == refs[end - 1]
                    if row.window_group == 'active':assert valid_labels[start:end].all() and np.all(refs[start:end] == 1)
                    if row.window_group == 'pre_fault':assert valid_labels[start:end].all() and np.all(refs[start:end] == 0) and row.phase == 'pre_fault'
                eligible = case[case.common_eligible]
                picked = sorted(set([int(eligible.index[0]), int(eligible.index[len(eligible)//2]), int(eligible.index[-1])]))
                for _, row in f.loc[picked].iterrows():
                    start, end = int(row.window_start_row), int(row.window_end_row)
                    rh = residual['vertical_m'][start + 10:end]; rr = residual['attitude_rad'][start + 10:end]
                    values = np.vstack([rh[None, :], rr.T, np.abs(rh)[None, :], np.linalg.norm(rr, axis=1)[None, :]])
                    actual = np.median(values, axis=1)
                    expected = row[NEW].to_numpy(dtype=float)
                    max_new = max(max_new, float(np.max(np.abs(actual - expected))))
                    assert np.max(np.abs(actual - expected)) < 1e-12
                    phys = raw[start:end].copy()
                    accel = phys[:, [index['accel_' + a] for a in 'xyz']]
                    phys[:, index['acc_norm']] = np.sqrt(np.sum(accel * accel, axis=1))
                    phys[:, index['baro_alt_relative']] -= phys[0, index['baro_alt_relative']]
                    z = (phys.astype(np.float32).T - mu) / std
                    ctrl = z[[index[f'actuator_ctrl_{j}'] for j in range(1, 5)]]
                    out = z[[index[f'actuator_out_{j}'] for j in range(1, 5)]]
                    gap = np.abs(ctrl - out)
                    effort = np.concatenate([ctrl, out]).mean(axis=0) / np.maximum(z[index['acc_norm']], .1)
                    old = np.vstack([out.std(axis=0), np.ptp(out, axis=0), gap.mean(axis=0), gap.std(axis=0), effort,
                        np.abs(z[index['acc_norm']]), np.linalg.norm(z[[index['gyro_' + a] for a in 'xyz']], axis=0),
                        np.abs(np.diff(z[index['baro_alt_relative']], prepend=z[index['baro_alt_relative'], 0]) - z[index['vel_z']]),
                        np.linalg.norm(z[[index['att_q_' + a] for a in 'xyz']], axis=0)])[:, 10:]
                    expected_old = row[OLD].to_numpy(dtype=float)
                    difference = float(np.max(np.abs(np.median(old.astype(float), axis=1) - expected_old)))
                    max_old = max(max_old, difference)
                    assert np.allclose(np.median(old.astype(float), axis=1), expected_old, rtol=2e-6, atol=2e-6)
                    checked += 1
    for path, expected in summary['inputs'].items():assert sha(path) == expected, path
    assert summary['script_sha256'] == sha('scripts/research/check_kinematic_feature_response.py')
    result = {'status': 'PASS_INDEPENDENT_FIXED36_SOURCE_RESPONSE_CHECKS', 'cases': 36,
        'original_admitted_windows': 2258, 'common_eligible_windows': 2072,
        'selected_train_normal_diagnostic_reference_windows': 340,
        'all_stage_statistics_and_matched_auc_recomputed': True,
        'auc_pairwise_max_abs_difference': max(errors), 'new_feature_max_abs_difference': max_new,
        'numpy_vs_torch_legacy_feature_max_abs_difference': max_old, 'representative_windows_independently_rebuilt': checked,
        'all_original_input_identities_unchanged': True, 'classifier_training_or_inference': False,
        'test_or_Real_read': False, 'verifier_sha256': sha(__file__)}
    with output.open('x', encoding='utf-8') as stream:json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
