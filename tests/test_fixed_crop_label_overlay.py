"""Audit label overlays and synthetic masking semantics; never train a UAV model."""
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts/research'))
sys.path.insert(0, str(ROOT / 'packages/uav_fault'))
from prepare_fixed_crop_label_overlay import prepare_frame
from uav_fault.models.cnn.focal_loss import FocalLoss


@pytest.fixture
def data():
    keys = pd.DataFrame({'split': ['train'] * 5, 'cache_index': np.arange(5),
        'case_uid': ['case' + str(i) for i in range(5)],
        'window_start_row': [0] * 5, 'window_end_row': [150] * 5})
    old = np.array([1, 0, 0, 0, 1])
    gtd = np.array([0, 1, 1, 0, 1])
    cmd = np.array([0, 1, 0, 0, 1])
    p = keys.assign(domain='SIL', existing_label=old, proposed_label=[0, 1, -1, 0, 1],
        gtd_state_reference=gtd, command_state_reference=cmd,
        source_references_agree=gtd == cmd, label_reference_eligible=gtd == cmd,
        input_clock_review_required=[False, False, False, False, True])
    q = keys.assign(gtd_endpoint_reference=gtd, command_endpoint_reference=cmd,
        input_clock_requires_review=[False, False, False, False, True],
        sensor_gap_over_0_1s_intersects_window=[True, False, False, False, np.nan])
    links = keys.assign(label=old, case_level2_label=['motor', 'motor', 'motor', 'normal', 'sensor'], flight_status='hover')
    return p, q, links, np.r_[old, 0, 0]


def test_overlay_preserves_every_index_rebuilds_origins_and_retains_suffix(data):
    p, q, links, y = data
    before = p.copy(deep=True); labels = y.copy()
    result = prepare_frame(p, q, links, y, 'train', 2)
    assert result.cache_index.tolist() == list(range(7))
    assert result.reference_label.tolist() == [0, 1, -1, 0, 1, 0, 0]
    assert result.pre_forward_candidate.tolist() == [True, True, False, True, False, True, True]
    assert result.normal_origin_reference.tolist() == ['normal_part_of_fault_case', 'fault_endpoint',
        'uncertain_endpoint', 'original_normal_case', 'fault_endpoint', 'rflysim_normal', 'rflysim_normal']
    assert result.loc[5:, 'case_uid'].isna().all()
    assert result.loc[5:, 'gap_over_0_1s_flag'].isna().all()
    assert not result.loc[5:, 'gap_over_0_1s_status_known'].any()
    assert result.loc[0, 'gap_over_0_1s_flag'] is True or result.loc[0, 'gap_over_0_1s_flag'] == True
    pd.testing.assert_frame_equal(p, before)
    np.testing.assert_array_equal(y, labels)
    assert not {'phase', 'fault_fraction', 'fault_onset_row'}.intersection(result.columns)


def test_overlapping_masks_do_not_remove_extra_indices(data):
    p, q, links, y = data
    p.loc[2, 'input_clock_review_required'] = True
    q.loc[2, 'input_clock_requires_review'] = True
    result = prepare_frame(p, q, links, y, 'train', 2)
    assert result.loc[~result.pre_forward_candidate, 'cache_index'].tolist() == [2, 4]


def test_shuffled_input_is_key_aligned_not_positionally_assigned(data):
    p, q, links, y = data
    expected = prepare_frame(p, q, links, y, 'train', 2)
    actual = prepare_frame(p.iloc[::-1], q.sample(frac=1, random_state=2), links.iloc[::-1], y, 'train', 2)
    pd.testing.assert_frame_equal(expected, actual)


@pytest.mark.parametrize('mutation', ['duplicate', 'wrong_case', 'old_label', 'lost_clock', 'false_agreement', 'forced_binary', 'missing_bool', 'invalid_label'])
def test_bad_identity_labels_or_flags_fail_closed(data, mutation):
    p, q, links, y = data
    if mutation == 'duplicate': p = pd.concat([p, p.iloc[:1]], ignore_index=True)
    elif mutation == 'wrong_case': q.loc[0, 'case_uid'] = 'different'
    elif mutation == 'old_label': y[0] = 0
    elif mutation == 'lost_clock': p.loc[4, 'input_clock_review_required'] = False
    elif mutation == 'false_agreement': p.loc[2, 'source_references_agree'] = True
    elif mutation == 'forced_binary': p.loc[2, 'proposed_label'] = 0
    elif mutation == 'missing_bool': p['label_reference_eligible'] = p.label_reference_eligible.astype(object); p.loc[0, 'label_reference_eligible'] = None
    elif mutation == 'invalid_label': p.loc[0, 'proposed_label'] = 2
    with pytest.raises(ValueError):
        prepare_frame(p, q, links, y, 'train', 2)


def test_unverified_suffix_is_not_assumed_normal(data):
    p, q, links, y = data
    y[-1] = 1
    with pytest.raises(ValueError, match='suffix'):
        prepare_frame(p, q, links, y, 'train', 2)


def test_source_label_vector_cannot_be_mistaken_for_full_train_cache(data):
    p, q, links, y = data
    with pytest.raises(ValueError, match='appended'):
        prepare_frame(p, q, links, y[:-2], 'train', 2)


def test_current_focal_loss_rejects_minus_one_and_subset_accepts_binary():
    logits = torch.tensor([[1., -1.], [0., 0.], [-1., 1.]])
    labels = torch.tensor([0, -1, 1])
    with pytest.raises((RuntimeError, IndexError)):
        FocalLoss()(logits, labels)
    keep = labels >= 0
    assert torch.isfinite(FocalLoss()(logits[keep], labels[keep]))


def test_loss_only_mask_does_not_prevent_batchnorm_state_contamination():
    x = torch.tensor([[[1., 2.]], [[3., 4.]], [[100., 200.]]])
    keep = torch.tensor([True, True, False])
    full = torch.nn.BatchNorm1d(1)
    filtered = torch.nn.BatchNorm1d(1)
    changed_unknown = torch.nn.BatchNorm1d(1)
    changed = x.clone(); changed[-1] *= 1000
    with torch.no_grad():
        full(x)  # masking loss after this point cannot undo state changes
        filtered(x[keep])
        changed_unknown(changed[keep])
    assert not torch.equal(full.running_mean, filtered.running_mean)
    torch.testing.assert_close(filtered.running_mean, changed_unknown.running_mean)
    torch.testing.assert_close(filtered.running_var, changed_unknown.running_var)
