from pathlib import Path
import sys
import numpy as np
import pytest
import torch
sys.path[:0] = [str(Path(__file__).resolve().parents[1]/'scripts/research'),
               str(Path(__file__).resolve().parents[1]/'packages/uav_fault')]
from analyze_m1_balanced_window_differences import align_prediction, change_groups, decompose, residual_summaries


def test_original_id_join_survives_reordered_prediction_rows_and_rejects_bad_labels():
    payload = {'original_indices': np.array([90, 10, 50]), 'compact_indices': np.array([2, 0, 1]),
               'labels': np.array([0, 1, 1]), 'probabilities': np.array([[.8, .2], [.1, .9], [.3, .7]])}
    out = align_prediction(payload, np.array([10, 50, 90]), np.array([1, 1, 0]))
    np.testing.assert_array_equal(out, [[.1, .9], [.3, .7], [.8, .2]])
    with pytest.raises(ValueError, match='different labels'):
        align_prediction(payload, np.array([10, 50, 90]), np.array([0, 1, 0]))
    duplicate = dict(payload, original_indices=np.array([10, 10, 90]))
    with pytest.raises(ValueError, match='unique'):
        align_prediction(duplicate, np.array([10, 50, 90]), np.array([1, 1, 0]))


def test_gross_changes_do_not_equal_net_miss_or_false_positive_changes():
    groups = change_groups([1, 1, 1, 1, 0, 0, 0, 0],
                           [1, 0, 0, 1, 1, 0, 1, 0],
                           [0, 1, 0, 1, 0, 1, 1, 0])
    assert groups.tolist() == ['new_miss', 'recovered_fault', 'shared_miss', 'shared_TP',
                               'resolved_FP', 'new_FP', 'shared_FP', 'shared_TN']


def test_linear_head_margin_decomposition_separates_shared_path_and_adapter():
    # First changed point is pushed from positive to negative by the adapter;
    # second stays negative on the learned shared path despite a positive adapter.
    shared = torch.tensor([[.8, .2], [1.7, .1]], dtype=torch.float64)
    addition = torch.tensor([[0., -.6], [0., .2]], dtype=torch.float64)
    head = torch.nn.Linear(2, 2, dtype=torch.float64)
    with torch.no_grad():
        head.weight.copy_(torch.tensor([[1., 0.], [0., 1.]], dtype=torch.float64))
        head.bias.copy_(torch.tensor([-.5, .5], dtype=torch.float64))
    common_logits = head(shared)
    full_logits = head(shared+addition)
    contribution = (addition@(head.weight[1]-head.weight[0])).detach().numpy()
    result = decompose(torch.tensor([[0., .4], [0., .2]], dtype=torch.float64).softmax(1).numpy(),
                       full_logits.softmax(1).detach().numpy(), contribution)
    expected = (common_logits[:, 1]-common_logits[:, 0]).detach().numpy()
    np.testing.assert_allclose(result.M1_common_path_margin, expected, rtol=0, atol=1e-14)
    np.testing.assert_allclose(result.M1_margin, result.M1_common_path_margin+result.M1_residual_margin)
    assert result.residual_pivotal_to_normal.tolist() == [True, False]
    assert result.M1_common_path_margin.iloc[0] > 0
    assert result.M1_common_path_margin.iloc[1] < 0
    # Saturated scores remain missing; no silent clipping.
    saturated = decompose([[0., 1.]], [[0., 1.]], [2.])
    assert not saturated.finite_saved_log_odds.iloc[0]
    assert np.isnan(saturated.M1_margin.iloc[0])


def test_residual_summaries_keep_physical_signed_units_and_train_scaling():
    r = np.zeros((1, 4, 150));r[:, 0] = -.04;r[:, 1] = .003;r[:, 2] = .004
    result = residual_summaries(r, [0.]*4, [.02, .001, .001, .001])
    assert result.N1_signed_median.iloc[0] == -.04
    assert result.N1_abs_median.iloc[0] == .04
    assert result.N1_train_z_rms.iloc[0] == pytest.approx(2.)
    assert result.attitude_norm_median_rad.iloc[0] == pytest.approx(.005)
    bad = r.copy();bad[0, 0, 10] = np.nan
    with pytest.raises(ValueError, match='Finite physical'):
        residual_summaries(bad, [0.]*4, [1.]*4)
