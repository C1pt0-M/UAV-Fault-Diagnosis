"""Synthetic data only. Never invoke final startup/scoring on real inputs."""
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts/research'))
import internal_test_final_v1 as final


def parity_fixture(tmp_path, m0=.002, gbt=1e-6):
    uid = 'SIL/hover/GPS/synthetic'
    a = np.full(150, np.nan); b = a.copy()
    a[149] = m0; b[149] = gbt
    np.savez(tmp_path / (uid.replace('/', '__') + '.npz'), m0=a, gbt=b)
    ledger = pd.DataFrame({'original_cache_index': [11], 'case_uid': [uid], 'qualified_y': [0],
                           'window_start_row': [0], 'window_end_row': [150]})
    saved = {'original_indices': np.array([11], np.int64), 'labels': np.array([0], np.int64),
             'm0_probabilities': np.array([[1., 0.]]), 'gbt_prob': np.array([0.])}
    return ledger, saved


def test_parity_both_exact_limits_and_exclusive_end(tmp_path):
    ledger, saved = parity_fixture(tmp_path)
    gate = final.parity_gate(ledger, saved, tmp_path)
    assert gate['pass'] and gate['aligned_windows'] == 1
    assert gate['GBT_max_abs_difference'] == 1e-6
    assert gate['M0_max_abs_difference'] == .002
    assert final.align_windows(ledger, saved['original_indices']).physical_end_row.tolist() == [149]


@pytest.mark.parametrize('arm', ['M0', 'GBT'])
def test_parity_just_beyond_either_limit_stops(tmp_path, arm):
    ledger, saved = parity_fixture(tmp_path,
        m0=np.nextafter(.002, np.inf) if arm == 'M0' else .002,
        gbt=np.nextafter(1e-6, np.inf) if arm == 'GBT' else 1e-6)
    assert not final.parity_gate(ledger, saved, tmp_path)['pass']


def test_parity_does_not_drop_missing_or_mismatched_labels(tmp_path):
    ledger, saved = parity_fixture(tmp_path, m0=np.nan)
    gate = final.parity_gate(ledger, saved, tmp_path)
    assert not gate['pass'] and gate['missing_window_original_indices'] == [11]
    ledger, saved = parity_fixture(tmp_path)
    saved['labels'][0] = 1
    gate = final.parity_gate(ledger, saved, tmp_path)
    assert not gate['pass'] and not gate['labels_equal']


def test_alignment_rejects_guessed_offsets_duplicate_and_missing_indices(tmp_path):
    ledger, saved = parity_fixture(tmp_path)
    ledger.loc[0, 'window_end_row'] = 149
    with pytest.raises(ValueError, match='end-exclusive'):
        final.align_windows(ledger, saved['original_indices'])
    with pytest.raises(ValueError, match='Unique'):
        final.align_windows(ledger, np.array([11, 11], dtype=np.int64))
    with pytest.raises(ValueError, match='same indices'):
        final.align_windows(ledger, np.array([12], dtype=np.int64))


def test_lock_gate_exclusive_creation_and_existing_target_refusal(tmp_path):
    out, scores = tmp_path / 'out', tmp_path / 'scores'
    final.fresh_targets(out, scores)
    info = {'config_sha256': 'cfg', 'models': {}, 'saved_predictions_sha256': 'pred'}
    with pytest.raises(RuntimeError, match='failed parity'):
        final.write_lock(tmp_path / 'blocked.lock', info, {'pass': False})
    assert not (tmp_path / 'blocked.lock').exists()
    out.mkdir()
    lock = final.write_lock(out / 'TEST_EVALUATED.lock', info, {'pass': True})
    assert lock['operating_points'] == {'GBT_window': .3744, 'M0_window': 'argmax', 'GBT_event': .95, 'M0_event': .6}
    with pytest.raises(FileExistsError):
        final.write_lock(out / 'TEST_EVALUATED.lock', info, {'pass': True})
    with pytest.raises(FileExistsError):
        final.fresh_targets(out, scores)
    scores.mkdir()
    with pytest.raises(FileExistsError):
        final.fresh_targets(tmp_path / 'another', scores)


def test_scoring_recipe_raw_gbt_normalized_m0_invalid_and_short_windows():
    import torch
    from replay_ens_events_val import case_windows, stats
    rows = np.arange(151 * 26, dtype=float).reshape(151, 26) / 10000
    valid = np.ones(151, dtype=bool); valid[0] = False
    ends, raw = case_windows(rows, valid)
    seen = {}
    class FakeM0:
        def __call__(self, x):
            seen['m0'] = x.numpy().copy()
            return torch.zeros((len(x), 2))
    class FakeGBT:
        def predict_proba(self, x):
            seen['gbt'] = x.copy()
            return np.tile([.4, .6], (len(x), 1))
    mu = np.ones((1, 26, 1), np.float32); sd = mu * 2
    pm, pg = final.score_arrays(rows, valid, FakeM0(), FakeGBT(), mu, sd)
    assert ends.tolist() == [150] and np.flatnonzero(np.isfinite(pm)).tolist() == [150]
    np.testing.assert_array_equal(seen['m0'], (raw - mu) / sd)
    np.testing.assert_array_equal(seen['gbt'], stats(raw))
    assert pm[150] == .5 and pg[150] == np.float32(.6)
    pm, pg = final.score_arrays(rows[:149], valid[:149], FakeM0(), FakeGBT(), mu, sd)
    assert np.isnan(pm).all() and np.isnan(pg).all()


def test_nonfinite_model_output_never_becomes_a_silently_dropped_window():
    import torch
    class FakeM0:
        def __call__(self, x):
            return torch.full((len(x), 2), float('nan'))
    class FakeGBT:
        def predict_proba(self, x):
            return np.full((len(x), 2), np.nan)
    with pytest.raises(ValueError, match='Nonfinite model output'):
        final.score_arrays(np.zeros((150, 26)), np.ones(150, bool), FakeM0(), FakeGBT(),
                           np.zeros((1, 26, 1), np.float32), np.ones((1, 26, 1), np.float32))


def window_fixture(n=100):
    uid = 'SIL/hover/GPS/synthetic'
    meta = pd.DataFrame({'case_uid': [uid] * n, 'window_end_row': [150] * n})
    cases = pd.DataFrame([{'case_uid': uid, 'domain': 'SIL', 'flight': 'hover', 'cls': 'GPS',
                           'mode': 4, 'excluded': True, 'association_group': 'real-synthetic-group'}])
    p0 = np.zeros((n, 2)); p0[:90] = [0, 1]; p0[90:] = [1, 0]
    p1 = np.zeros(n); p1[:89] = .3744
    saved = {'labels': np.ones(n, np.int64), 'm0_probabilities': p0, 'gbt_prob': p1}
    return meta, saved, cases


def test_F3_one_pp_exact_boundary_and_argmax_tie():
    meta, saved, cases = window_fixture()
    result, _, _ = final.window_metrics(meta, saved, cases)
    assert result['F3']['GPS']['pass'] and result['F3']['GPS']['delta_pp'] == -1
    saved['gbt_prob'][88] = np.nextafter(.3744, -np.inf)
    assert not final.window_metrics(meta, saved, cases)[0]['F3']['GPS']['pass']
    saved['labels'][:] = 0; saved['m0_probabilities'][:] = .5
    result, pred, _ = final.window_metrics(meta, saved, cases)
    assert not pred['M0'].any() and result['overall']['M0']['normal_FP'] == 0


def test_F3_window_exclusion_strict_after30_not_at30():
    meta, saved, cases = window_fixture(100)
    meta.loc[0, 'window_end_row'] = 1500
    meta.loc[1, 'window_end_row'] = 1501
    result, _, _ = final.window_metrics(meta, saved, cases)
    assert result['excluded_late_hover_mode4']['windows'] == 1
    assert result['F3']['GPS']['windows'] == 99
    assert result['excluded_late_hover_mode4']['recall']['GBT'] == 1


def primary_fixture(candidate=89):
    b = {'all_cases': {'fault_cases': 100, 'indep_false_events': 2, 'indep_events_per_h': 2.},
         'excluding': {'fault_cases': 90, 'fault_detected': 90, 'detected_by_class': {'motor': 90}}}
    p = {'all_cases': {'fault_cases': 100, 'indep_false_events': 1, 'indep_events_per_h': 1.},
         'excluding': {'fault_cases': 90, 'fault_detected': candidate, 'detected_by_class': {'motor': candidate}}}
    return {'M0': b, 'GBT': p}


def test_F1_strict_F2_full_denominator_and_flag_not_a_veto():
    x = primary_fixture()
    assert final.event_primary(x)['verdict'] == 'CONFIRMED'
    x['GBT']['excluding']['fault_detected'] = 88
    assert final.event_primary(x)['verdict'] == 'NOT_CONFIRMED'
    x = primary_fixture(90)
    x['M0']['excluding']['detected_by_class'] = {'GPS': 10, 'motor': 80}
    x['GBT']['excluding']['detected_by_class'] = {'GPS': 7, 'motor': 83}
    result = final.event_primary(x)
    assert result['verdict'] == 'CONFIRMED' and result['F2']['flagged_classes_deficit_gt2'] == ['GPS']
    x['GBT']['all_cases']['indep_events_per_h'] = 2.
    assert not final.event_primary(x)['F1']['pass']


def test_F4_two_second_strict_flag_and_all_common_includes_excluded():
    summaries = {a: {'per_class_all': {c: {'confirm_median_s': None} for c in final.CLASSES}} for a in ('M0', 'GBT')}
    summaries['M0']['per_class_all']['GPS'] = {'confirm_median_s': 1., 'confirm_p90_s': 2.}
    summaries['GBT']['per_class_all']['GPS'] = {'confirm_median_s': 3., 'confirm_p90_s': 4.}
    d = pd.DataFrame([{'case_uid': 'synthetic', 'arm': a, 'onset_th': th, 'detected': True,
                       'confirm_latency_s': delay, 'excluded': True} for a, th, delay in [('M0', .6, 1), ('GBT', .95, 3)]])
    result = final.latency_metrics(summaries, d)
    assert result['F4']['GPS']['flag_slower_gt2s'] is False
    assert result['paired_all_common_detected']['cases'] == 1
    assert result['paired_all_common_detected']['median'] == 2
    summaries['GBT']['per_class_all']['GPS']['confirm_median_s'] = 3.001
    assert final.latency_metrics(summaries, d)['F4']['GPS']['flag_slower_gt2s'] is True


def test_bootstrap_shared_group_plan_and_undefined_not_dropped():
    uids = ['SIL/hover/no_fault/a', 'HIL/hover/no_fault/a', 'SIL/hover/no_fault/b']
    cases = pd.DataFrame({'case_uid': uids, 'association_group': ['a', 'a', 'b']})
    frame = pd.DataFrame({'case_uid': uids}); y = np.zeros(3, np.int64)
    pred = {'M0': np.ones(3, bool), 'GBT': np.array([True, False, False])}
    data = pd.DataFrame([{'case_uid': uid, 'association_group': group, 'cls': 'no_fault', 'arm': arm,
        'onset_th': .6 if arm == 'M0' else .95, 'false_events': int(pred[arm][i]), 'normal_exposure_s': 1.}
        for i, (uid, group) in enumerate(zip(uids, cases.association_group)) for arm in ('M0', 'GBT')])
    e = final.events.bootstrap_rates(data, {'GBT': .95, 'M0': .6}, reps=50)
    w = final.window_fp_bootstrap(frame, y, pred, cases, replicates=50)
    assert w['groups'] == e['groups'] == 2
    np.testing.assert_allclose(w['ci95'], e['comparisons']['GBT_over_M0_0.60']['ci95'])
    pred['M0'][:] = False
    w = final.window_fp_bootstrap(frame, y, pred, cases, replicates=23)
    assert w['ci95'] is None and w['undefined_replicates'] == 23 and w['ratio'] is None


def test_source_boundary_protects_historical_lock_and_other_holdouts(tmp_path):
    code = r'''
import sys,json
from pathlib import Path
sys.path.insert(0, sys.argv[1])
import pandas as pd
import internal_test_final_v1 as f
p=Path(sys.argv[2]);f.DATA_ROOT=p/'data';f.DATA=f.DATA_ROOT/'causal';f.SCORES=f.DATA_ROOT/'scores'
f.OLD_LOCK=p/'historical.lock';f.SAVED=p/'saved.npz';f.CFG=p/'cfg.json'
f.DATA.mkdir(parents=True);f.SCORES.mkdir();(f.DATA/'cases').mkdir()
f.CFG.write_text(json.dumps({'frozen_models':{}}));f.OLD_LOCK.write_text('old');f.SAVED.write_bytes(b'old')
(f.DATA/'tasks.json').write_text('[]');(f.DATA/'cases/SIL__hover__GPS__synthetic.npz').write_bytes(b'synthetic')
f.input_boundary(pd.DataFrame({'case_uid':['SIL/hover/GPS/synthetic']}))
blocked=0
for work in (lambda:f.OLD_LOCK.write_text('changed'),lambda:f.OLD_LOCK.unlink(),
 lambda:(f.DATA/'tasks.json').write_text('changed'),lambda:(f.DATA/'cases/Real__forbidden.npz').read_bytes()):
 try:work()
 except PermissionError:blocked+=1
assert blocked==4
assert (f.DATA/'cases/SIL__hover__GPS__synthetic.npz').read_bytes()==b'synthetic'
(f.SCORES/'new.npz').write_bytes(b'new')
print('SYNTHETIC_BOUNDARY_PASS')
'''
    result = subprocess.run([sys.executable, '-X', 'utf8', '-I', '-B', '-c', code,
        str(final.ROOT / 'scripts/research'), str(tmp_path)], capture_output=True, text=True, encoding='utf-8')
    assert result.returncode == 0, result.stdout + result.stderr
    assert 'SYNTHETIC_BOUNDARY_PASS' in result.stdout
