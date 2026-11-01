from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts/data'))
from repair_internal_test_labeltime import eligible, phase_info, build_window_metadata


def test_disagreement_is_unknown_not_normal():
    gtd = np.array([0, 1, 1, 0], dtype=np.int8)
    cmd = np.array([0, 0, 1, 0], dtype=np.int8)
    cover = np.array([True, True, True, False])
    labels, valid = eligible(gtd, cmd, cover, np.ones(4, dtype=bool), False, np.ones(4, dtype=bool))
    assert labels.tolist() == [0, -1, 1, -1]
    assert valid.tolist() == [True, False, True, False]


def test_clock_review_keeps_references_but_blocks_eligibility():
    a = np.array([0, 1], dtype=np.int8); yes = np.ones(2, dtype=bool)
    labels, valid = eligible(a, a, yes, yes, True, yes)
    assert labels.tolist() == [0, 1] and not valid.any()


def test_nonfinite_input_is_not_silently_approved():
    a = np.array([0, 1], dtype=np.int8); yes = np.ones(2, dtype=bool)
    labels, valid = eligible(a, a, yes, yes, False, np.array([True, False]))
    assert labels.tolist() == [0, 1] and valid.tolist() == [True, False]
    with pytest.raises(ValueError, match='shape mismatch'):
        eligible(a, a, yes[:1], yes, False, yes)


def test_checkpoint_metadata_keeps_original_identity_and_grid():
    row = {'case_uid': 'SIL/case', 'domain': 'SIL', 'level2_label': 'motor', 'final_output_path': 'old.csv'}
    n = 500; q = np.arange(n, dtype=np.int64) * 20000
    old = np.zeros(n, dtype=np.int8); ref = old.copy(); ref[300:] = 1
    yes = np.ones(n, dtype=bool); no = np.zeros(n, dtype=bool)
    records = build_window_metadata(row, q, old, ref, ref, yes, yes, q, no, False, Path('new.csv'))
    assert len(records) == 64
    assert records[0]['window_start_row'] == 0 and records[-1]['window_end_row'] == 500
    assert all(r['historical_source_csv'] == 'old.csv' and r['source_csv'] == 'new.csv' for r in records)
    assert all(r['endpoint_evaluation_valid'] for r in records)
    assert records[-1]['qualified_label'] == 1


def test_phase_does_not_call_cleared_fault_stable_fault():
    state = np.array([0, 0, 1, 1, 0, 0], dtype=np.int8)
    assert phase_info(state, 0, 2) == ('pre_fault', 0.0, 2)
    assert phase_info(state, 2, 4) == ('stable_fault', 1.0, 2)
    assert phase_info(state, 4, 6) == ('post_fault_normal', 0.0, 2)
    assert phase_info(state, 1, 4) == ('transition', 2 / 3, 2)
    assert phase_info(np.array([-1, 0]), 0, 2) == ('unknown_reference', None, -1)
