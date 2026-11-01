from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts/research'))
from check_kinematic_feature_response import (
    FEATURES, legacy_features, fixed_auc, fit_diagnostic_reference,
    quarter, whole_window_group, matched_fault_responses, phase_stability,
)
from uav_fault.model_contract import RFLY_SEQ_COLUMNS_V2
from uav_fault.models.cnn.lstm import OnlineKGMSTCA


def frame():
    rows = []
    for split, kind, uid, offset in [('train', 'normal', 'n1', 0), ('val', 'normal', 'n2', 100),
            ('train', 'fault', 'f1', 1000)]:
        for i in range(4):
            row = dict(split=split, kind=kind, case_uid=uid, domain='SIL', subtype='motor' if kind == 'fault' else 'no_fault',
                association_group=uid, common_eligible=True, crop_quarter=i,
                window_group='active' if kind == 'fault' else 'normal_other_phase')
            row.update({feature: float(i + offset) for feature in FEATURES})
            rows.append(row)
    return pd.DataFrame(rows)


def test_fixed_auc_ties_and_no_sign_flip():
    assert fixed_auc([0, 1], [2, 3]) == 1
    assert fixed_auc([2, 3], [0, 1]) == 0
    assert fixed_auc([1, 1], [1, 1]) == .5
    assert fixed_auc([], [1]) is None
    with pytest.raises(ValueError):
        fixed_auc([np.nan], [1])


def test_reference_never_uses_val_or_fault_or_invalid():
    f = frame(); f.loc[len(f)] = dict(f.iloc[0], common_eligible=False, **{k: 1e9 for k in FEATURES})
    reference = fit_diagnostic_reference(f)
    for feature in FEATURES:
        assert reference[feature]['median'] == 1.5
        assert reference[feature]['iqr'] == 1.5
        assert reference[feature]['windows'] == 4
    f.loc[(f['split'] == 'train') & f.kind.eq('normal'), FEATURES] = 0
    reference = fit_diagnostic_reference(f)
    assert all(not r['usable_scale'] for r in reference.values())


def test_normal_stage_comparisons_are_case_paired():
    f = frame(); reference = fit_diagnostic_reference(f)
    rows = phase_stability(f, reference)
    assert len(rows) == 2 * len(FEATURES)
    assert all(r['late_minus_early_median'] == 3 for r in rows)
    assert all(r['shift_in_train_normal_iqr'] == 2 for r in rows)


def test_fault_auc_matches_domain_split_and_crop_quarter():
    f = frame()
    # Extreme validation-normal values must not contaminate train comparisons.
    reference = fit_diagnostic_reference(f)
    rows = matched_fault_responses(f, reference)
    assert len(rows) == len(FEATURES)
    assert all(r['matched_crop_quarter_auc'] == 1 for r in rows)
    assert all(r['matched_active_windows'] == 4 for r in rows)
    for r in rows:
        assert [q['normal_windows'] for q in r['quarter_details']] == [1, 1, 1, 1]
        assert r['active_minus_pre_median'] is None


def test_whole_label_reference_excludes_mixed_or_unknown_windows():
    known = np.ones(150, dtype=bool)
    assert whole_window_group(np.zeros(150), known, 'pre_fault') == 'pre_fault'
    assert whole_window_group(np.ones(150), known, 'fault') == 'active'
    assert whole_window_group(np.r_[np.zeros(75), np.ones(75)], known, 'fault') == 'mixed_boundary'
    known[0] = False
    assert whole_window_group(np.ones(150), known, 'fault') == 'label_unknown_in_window'


def test_quarters_use_original_crop_not_last_qualified_window():
    assert quarter(200, 1601) == 0
    assert quarter(600, 1601) == 1
    assert quarter(1601, 1601) == 3
    with pytest.raises(ValueError):
        quarter(1602, 1601)


def test_legacy_formula_exactly_reuses_existing_function_without_network_forward():
    rng = np.random.default_rng(10)
    x = rng.normal(size=(26, 150)).astype(np.float32)
    columns = list(RFLY_SEQ_COLUMNS_V2)
    expected_model = OnlineKGMSTCA(in_channels=26, hidden_size=64, num_classes=2, columns=columns)
    with torch.inference_mode():
        expected = expected_model._relation_features(torch.from_numpy(x).unsqueeze(0))[0].numpy()
    assert np.array_equal(legacy_features(x, columns), expected.astype(np.float64))
    with pytest.raises(ValueError):
        legacy_features(x.astype(np.float64), columns)
