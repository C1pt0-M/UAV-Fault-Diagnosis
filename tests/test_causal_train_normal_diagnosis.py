from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts/research'))
from test_causal_train_normals import normal_error_summary


def metadata():
    return pd.DataFrame({
        'case_uid': ['SIL/hover/no_fault/A'] * 4 + ['HIL/hover/motor/B'],
        'domain': ['SIL'] * 4 + ['HIL'], 'flight_status': ['hover'] * 5,
        'case_level2_label': ['normal'] * 4 + ['motor'],
        'normal_origin': ['original_normal_case'] * 4 + ['normal_part_of_fault_case'],
        'cache_index': [0, 1, 2, 3, 7], 'label': [0] * 5,
        'window_end_row': [200, 400, 600, 801, 150],
    })


def test_normal_origin_domain_and_original_crop_quarters():
    f = metadata()
    result, cases = normal_error_summary(f, np.array([0, 0, 1, 1, 0]),
        np.array([.1, .2, .7, .8, .2]), {'SIL/hover/no_fault/A': 801})
    assert result['all_normal']['windows'] == 5
    assert result['original_normal_hover']['windows'] == 4
    assert result['original_normal_hover']['false_positives'] == 2
    assert result['inactive_fault_case_normal']['false_positives'] == 0
    assert [r['windows'] for r in result['hover_by_original_crop_quarter']] == [1, 1, 1, 1]
    assert [r['false_positives'] for r in result['hover_by_original_crop_quarter']] == [0, 0, 1, 1]
    assert cases.false_positive_rate.tolist() == [.5]
    assert not result['crop_quarters_are_physical_flight_phase']


def test_never_normalize_progress_by_last_selected_valid_window():
    f = metadata().iloc[:3].copy()
    result, _ = normal_error_summary(f, np.array([0, 0, 1]),
        np.array([.1, .2, .7]), {'SIL/hover/no_fault/A': 1601})
    assert [r['windows'] for r in result['hover_by_original_crop_quarter']] == [2, 1, 0, 0]


def test_reject_faults_duplicate_ids_and_unknown_crop_lengths():
    f = metadata(); f.loc[0, 'label'] = 1
    with pytest.raises(ValueError, match='Normal-only'):
        normal_error_summary(f, np.zeros(5), np.zeros(5), {'SIL/hover/no_fault/A': 801})
    f = metadata(); f.loc[1, 'cache_index'] = 0
    with pytest.raises(ValueError, match='unique'):
        normal_error_summary(f, np.zeros(5), np.zeros(5), {'SIL/hover/no_fault/A': 801})
    with pytest.raises(ValueError, match='crop length'):
        normal_error_summary(metadata(), np.zeros(5), np.zeros(5), {})


def test_reject_nonfinite_or_mismatched_predictions():
    with pytest.raises(ValueError, match='shape'):
        normal_error_summary(metadata(), np.zeros(4), np.zeros(5), {})
    with pytest.raises(ValueError, match='Invalid'):
        normal_error_summary(metadata(), np.zeros(5), np.array([.1, .2, np.nan, .4, .5]), {})
