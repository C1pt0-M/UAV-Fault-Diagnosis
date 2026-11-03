"""Small runnable checks for the saved-prediction-only M3 error analysis."""
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts/research'))
from analyze_m3_error_samples import transitions, counts, load_predictions, linked_patterns


def test_all_eight_transitions_and_risk_denominators():
    y = np.array([1, 1, 0, 0, 1, 1, 0, 0])
    ref = np.array([0, 1, 1, 0, 0, 1, 1, 0])
    pred = np.array([1, 0, 0, 1, 0, 1, 1, 0])
    event = transitions(y, ref, pred)
    assert event.tolist() == ['FN_to_TP', 'TP_to_FN', 'FP_to_TN', 'TN_to_FP',
                              'both_FN', 'both_TP', 'both_FP', 'both_TN']
    f = pd.DataFrame({'event': event, 'case_uid': ['a'] * 8, 'association_group': ['1'] * 8})
    result = counts(f)
    assert result['TP_to_FN_reference_risk_windows'] == 2
    assert result['TP_to_FN_fraction_of_reference_risk'] == .5
    assert result['net_fault_TP_delta'] == result['net_normal_FP_delta'] == 0


def test_prediction_identity_order_and_tie(tmp_path):
    ids, y = np.array([7, 9]), np.array([1, 0])
    p = tmp_path / 'saved.npz'
    np.savez(p, original_indices=ids, labels=y, probabilities=np.array([[.5, .5], [.4, .6]]))
    scores, pred = load_predictions(p, ids, y)
    assert pred.tolist() == [0, 1] and scores.tolist() == [.5, .6]
    with pytest.raises(ValueError, match='ID/label/order'):
        load_predictions(p, ids[::-1], y)
    np.savez(p, original_indices=ids, labels=y, probabilities=np.array([[np.nan, .5], [.4, .6]]))
    with pytest.raises(ValueError, match='finite'):
        load_predictions(p, ids, y)


def test_linked_recovery_and_false_positive_undo_are_disjoint():
    f = pd.DataFrame({'label': [1, 1, 0, 0], 'pred_C0': [1, 0, 1, 0],
                      'pred_CN': [0, 0, 0, 0], 'pred_CON': [1, 1, 1, 1]})
    masks = linked_patterns(f)
    assert masks['CN_lost_C0_TP_recovered_CON'].tolist() == [True, False, False, False]
    assert masks['both_C0_CN_FN_recovered_CON'].tolist() == [False, True, False, False]
    assert masks['CN_corrected_C0_FP_undone_CON'].tolist() == [False, False, True, False]
    assert masks['both_C0_CN_TN_new_FP_CON'].tolist() == [False, False, False, True]
