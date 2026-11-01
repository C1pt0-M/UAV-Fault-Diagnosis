"""Fixed-crop diagnostic references must not mutate source labels."""
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts/research'))
from validate_fixed_crop_labels import interval_states, microseconds, validate_case, digest


def test_integer_microseconds_keep_exact_onset_and_exclusive_offset():
    q = np.array([1000000, 2000000, 3000000, 4000000], dtype=np.int64)
    state, valid = interval_states(q, [{'start_s': 2., 'end_s': 4.}], 1000000, 4000000)
    assert state.tolist() == [0, 1, 1, 0]
    assert valid.all()


def test_missing_coverage_is_unknown_not_normal():
    state, valid = interval_states(np.array([0, 1, 2, 3]), [], 1, 2)
    assert state.tolist() == [-1, 0, 0, -1]
    assert valid.tolist() == [False, True, True, False]


def test_activation_after_fixed_crop_cannot_be_compressed_inside_it():
    state, valid = interval_states(np.arange(0, 3000000, 20000),
                                  [{'start_s': 5., 'end_s': None}], 0, 6000000)
    assert valid.all() and not state.any()


@pytest.mark.parametrize('events', [
    [{'start_s': 2., 'end_s': 2.}],
    [{'start_s': 2., 'end_s': None}, {'start_s': 3., 'end_s': None}],
    [{'start_s': 2., 'end_s': 4.}, {'start_s': 3., 'end_s': 5.}],
])
def test_invalid_event_order_or_overlap_rejected(events):
    with pytest.raises(ValueError):
        interval_states(np.arange(6, dtype=np.int64), events, 0, 5)


@pytest.mark.parametrize('value', [float('nan'), float('inf'), .0000001])
def test_nonfinite_or_submicrosecond_event_rejected(value):
    with pytest.raises(ValueError):
        microseconds(value)


def test_two_raw_references_need_not_be_forced_to_agree():
    q = np.array([0, 1000000, 2000000], dtype=np.int64)
    a, _ = interval_states(q, [{'start_s': 1., 'end_s': None}], 0, 2000000)
    b, _ = interval_states(q, [{'start_s': 2., 'end_s': None}], 0, 2000000)
    assert (a == b).tolist() == [True, False, True]


def test_diagnostic_sidecar_preserves_crop_and_source_file(tmp_path):
    source = tmp_path / 'case.csv'
    q = np.arange(150, dtype=np.int64) * 20000
    pd.DataFrame({'Timestamp': q, 'UAVState_data_fault_state': np.ones(150)}).to_csv(source, index=False)
    before = source.read_bytes()
    row = dict(case_uid='SIL/hover/motor/example', domain='SIL', split='train',
               flight_status='hover', source_fault_type='motor', final_output_path=str(source))
    old = dict(csv_sha256=digest(source), rows=150)
    links = [dict(window_start_row=0, window_end_row=150, label=1, cache_index=0)]
    saved = dict(gtd_start_s=0., gtd_end_s=6., gtd_events=[{'start_s': 5., 'end_s': None}],
                 command_events=[{'start_s': 5.1, 'end_s': None, 'last_observed_s': 6.}],
                 raw_gtd_sha256='test-gtd', ulog_sha256='test-ulog', command_reference_covered_rows=150,
                 raw_onset_after_crop=True)
    out = tmp_path / 'diagnostics'
    (out / 'label_references').mkdir(parents=True)
    record, windows, fresh = validate_case((row, old, links), saved, out, False)
    assert source.read_bytes() == before
    assert record['changed_windows'] == 1 and record['all_crop_reference_inactive']
    assert windows[0]['gtd_state_reference'] == 0 and windows[0]['existing_label'] == 1
    assert fresh is None
    with np.load(next((out / 'label_references').glob('*.npz')), allow_pickle=False) as d:
        np.testing.assert_array_equal(d['timestamp_us'], q)
        assert d['existing_state'].all() and not d['gtd_state_reference'].any()
        assert d['gtd_covered'].all()
