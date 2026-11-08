"""Synthetic checks only; no inference, training, or real held-out data."""
from pathlib import Path
import json
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts/research'))
import numpy as np
import pandas as pd
import pytest
import gbt_event_operating_point_v1 as g


@pytest.mark.parametrize('gap_frames,expected_false', [(20, 0), (40, 1)])
def test_event_fields_equal_frozen_case_events(tmp_path, monkeypatch, gap_frames, expected_false):
    uid = 'SIL/hover/GPS/example'
    p = np.r_[np.full(30, .1), np.full(40, .9), np.full(gap_frames, .1), np.full(60, .9)].astype(np.float32)
    times = np.arange(len(p), dtype=np.int64) * 20000
    p[:3] = np.nan
    normal_rows = 60 + gap_frames
    np.savez(tmp_path / (uid.replace('/', '__') + '.npz'), timestamp_us=times, m0=p, gbt=p,
             reference_state=np.r_[np.zeros(normal_rows), np.ones(len(p) - normal_rows)].astype(np.int8), row_label_valid=np.ones(len(p), dtype=bool))
    monkeypatch.setattr(g.reference, 'SCORES', tmp_path)
    monkeypatch.setattr(g.reference, 'GRID', [.75])
    old = g.reference.case_events(uid)
    new = g.case_records((uid, [('M0', (.6,)), ('ENS', (.75,))], tmp_path))
    for a, b in zip(old, new):
        for key, value in a.items():
            if isinstance(value, float) and np.isnan(value):
                assert np.isnan(b[key])
            else:
                assert b[key] == value
    assert new[0]['false_events'] == expected_false
    assert new[0]['spans_onset'] is (expected_false == 0)
    assert new[0]['detected'] is (expected_false == 1)
    if expected_false:
        assert new[0]['confirm_latency_s'] >= 0
    else:
        assert np.isnan(new[0]['confirm_latency_s'])  # Merged pre-onset start is not a new detection.
    assert g.BASE == g.reference.BASE


def event_table(*, extra_at_09=False, bad_class=False, delay=1.5):
    rows = []
    for arm in ('M0', 'GBT'):
        for th in g.GRID:
            for cls in ('no_fault', *g.CLASSES, 'GPS_extra', 'GPS_excluded'):
                excluded = cls == 'GPS_excluded'
                normal = cls == 'no_fault'
                extra = cls == 'GPS_extra'
                detected = not normal and (not extra or (arm == 'GBT' and (th == .95 or extra_at_09 and th == .9)))
                if arm == 'GBT' and bad_class and cls == 'barometer':
                    detected = False
                if arm == 'GBT' and excluded:
                    detected = False
                rows.append({'case_uid': cls, 'arm': arm, 'onset_th': th, 'excluded': excluded,
                    'cls': 'GPS' if cls.startswith('GPS') else cls, 'flight': 'hover',
                    'false_events': (0 if th >= .9 else 10) if normal else 0,
                    'normal_exposure_s': 3600. if normal else 0., 'false_occupancy_s': 0.,
                    'post_onset_scored': not normal, 'detected': detected, 'spans_onset': False,
                    'confirm_latency_s': (1. if arm == 'M0' else delay) if detected else np.nan})
    return pd.DataFrame(rows)


def test_selection_constraints_exclusion_more_detected_then_lower_onset():
    table, chosen = g.selection_table(event_table())
    assert chosen == {'GBT': .95, 'M0': .9}
    assert all(r['constraint_a'] and r['constraint_b'] for r in table)
    assert all(r['baseline_targets']['GPS'] == 1 for r in table)  # Excluded case cannot impose a target.
    _, chosen = g.selection_table(event_table(extra_at_09=True))
    assert chosen['GBT'] == .9


@pytest.mark.parametrize('bad_class,delay', [(True, 1.5), (False, 1.500001)])
def test_no_feasible_does_not_relax_class_or_latency(bad_class, delay):
    table, chosen = g.selection_table(event_table(bad_class=bad_class, delay=delay))
    assert chosen['GBT'] is None and chosen['M0'] is not None
    rows = [r for r in table if r['arm'] == 'GBT']
    assert not any(r['feasible'] for r in rows)
    assert all(not r['constraint_a' if bad_class else 'constraint_b'] for r in rows)


def test_paired_statistic_is_median_of_differences_not_difference_of_medians():
    rows = []
    for arm, delays in [('M0', [0., 10., 11.]), ('GBT', [11., 0., 20.])]:
        for i, delay in enumerate(delays):
            rows.append({'case_uid': str(i), 'arm': arm, 'onset_th': .6, 'excluded': False,
                         'detected': True, 'confirm_latency_s': delay})
    result = g.paired_latency(pd.DataFrame(rows), 'GBT', .6)
    assert result['median'] == 9.
    assert result['candidate_median_s'] - result['baseline_median_s'] == 1.


def test_no_common_detected_latency_is_unavailable():
    d = event_table()
    d.loc[d.arm.eq('GBT'), 'detected'] = False
    assert g.paired_latency(d, 'GBT', .6)['median'] is None


@pytest.mark.parametrize('missing,expected', [(4, True), (5, False)])
def test_ev3_uses_registered_one_percent_and_flags_without_class_veto(missing, expected):
    rows = []
    for arm in ('M0', 'GBT'):
        for i in range(473):
            normal = i == 472
            detected = not normal and not (arm == 'GBT' and 8 <= i < 8 + missing)
            rows.append({'case_uid': str(i), 'arm': arm, 'onset_th': .6, 'excluded': i < 8,
                'cls': 'no_fault' if normal else 'GPS', 'false_events': (2 if arm == 'GBT' else 3) if normal else 0,
                'normal_exposure_s': 3600. if normal else 0., 'false_occupancy_s': 0.,
                'post_onset_scored': not normal, 'detected': detected, 'spans_onset': False})
    d = pd.DataFrame(rows)
    result = g.acceptance(d, d, {'GBT': .6})
    ev3 = result['criteria']['EV3']
    assert ev3['pass'] is expected and ev3['scored_fault_cases'] == 472 and ev3['scored_fault_cases_excluding'] == 464
    assert ev3['allowance_cases'] == 4.72 and ev3['flagged_classes_deficit_gt2'] == ['GPS']
    assert result['verdict'] == ('PASS' if expected else 'FAIL')
    json.dumps(result, allow_nan=False)


def test_bootstrap_is_paired_and_does_not_redraw_undefined_rates():
    rows = []
    for arm in ('M0', 'GBT'):
        for uid, normal, group in [('n', True, '001'), ('f', False, '002')]:
            rows.append({'case_uid': uid, 'arm': arm, 'onset_th': .6, 'association_group': group,
                'cls': 'no_fault' if normal else 'GPS', 'false_events': 1 if normal and arm == 'M0' else 0,
                'normal_exposure_s': 3600. if normal else 0.})
    result = g.bootstrap_rates(pd.DataFrame(rows), {'GBT': .6, 'M0': .6}, reps=40)
    assert result['replicates'] == 40 and result['groups'] == 2 and result['onset_reselected'] is False
    for x in result['comparisons'].values():
        assert x['undefined_replicates'] > 0 and x['ci95'] is None


def test_source_boundary_denies_val_before_selection_and_any_source_write(tmp_path):
    data = tmp_path / 'data'
    scores = data / 'scores'
    scores.mkdir(parents=True)
    out = tmp_path / 'output'
    out.mkdir()
    train_uid, val_uid = 'SIL/hover/GPS/train', 'SIL/hover/GPS/val'
    for uid in (train_uid, val_uid):
        (scores / (uid.replace('/', '__') + '.npz')).write_bytes(b'existing saved scores')
    ledger = data / 'train.csv'
    ledger.write_text('case_uid,association_group\n', encoding='utf-8')
    script = f'''
import sys,json
from pathlib import Path
sys.path.insert(0,{str(ROOT / 'scripts/research')!r})
import gbt_event_operating_point_v1 as g
g.DATA_ROOT=Path({str(data)!r})
scores=Path({str(scores)!r});out=Path({str(out)!r})
g.install_source_boundary(scores,Path({str(ledger)!r}),out,({train_uid!r},),({val_uid!r},))
val=scores/({val_uid!r}.replace('/','__')+'.npz')
try: val.read_bytes()
except PermissionError: pass
else: raise AssertionError('val read before selection')
(out/'selection_train_only.json').write_text(json.dumps({{'chosen_onsets':{{'GBT':.6,'M0':.6}}}}),encoding='utf-8')
assert val.read_bytes()==b'existing saved scores'
try: val.write_bytes(b'forbidden overwrite')
except PermissionError: pass
else: raise AssertionError('source overwrite')
try: (g.DATA_ROOT/'cache/val.pt').read_bytes()
except PermissionError: pass
else: raise AssertionError('raw val cache read')
'''
    result = subprocess.run([sys.executable, '-I', '-B', '-c', script], capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr


def test_availability_mismatch_stops_instead_of_changing_exposure(tmp_path):
    uid = 'SIL/hover/GPS/example'
    np.savez(tmp_path / (uid.replace('/', '__') + '.npz'), timestamp_us=np.array([0, 20000], dtype=np.int64),
        m0=np.array([np.nan, .8]), gbt=np.array([.7, .8]), reference_state=np.array([0, 1], dtype=np.int8),
        row_label_valid=np.ones(2, dtype=bool))
    with pytest.raises(ValueError, match='availability'):
        g.case_records((uid, [('GBT', (.6,))], tmp_path))
