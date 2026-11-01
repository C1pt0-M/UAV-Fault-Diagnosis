"""python -m pytest -q tests/test_diagnose_normal_adaptation_v1.py"""
import importlib.util
from pathlib import Path
import sys

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts/research"))
SPEC = importlib.util.spec_from_file_location("normal_adaptation_diagnosis", ROOT / "scripts/research/diagnose_normal_adaptation_v1.py")
diagnosis = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(diagnosis)


def test_ema_weights_match_reset_mean_and_variance_recurrences():
    residual, weights = diagnosis.ema_weights(0.1, 10)
    assert residual == pytest.approx(0.3486784401)
    assert residual + sum(weights) == pytest.approx(1)
    assert weights[-1] / weights[0] == pytest.approx(1 / 0.9**9)
    means, variances = np.arange(1, 11), np.arange(1, 11) ** 2
    mean, var = 0.0, 1.0
    for m, v in zip(means, variances):
        mean = 0.9 * mean + 0.1 * m
        var = 0.9 * var + 0.1 * v
    assert mean == pytest.approx(np.dot(weights, means))
    assert var == pytest.approx(residual + np.dot(weights, variances))
    assert diagnosis.ema_weights(1.0, 3) == (0.0, [0.0, 0.0, 1.0])
    with pytest.raises(ValueError):
        diagnosis.ema_weights(0, 10)


def test_probability_summary_checks_ranges_and_hysteresis_boundaries():
    result = diagnosis.probability_summary([0, 0.4, 0.6, 0.95, 1])
    assert result["fraction_below_clear_0_4"] == 0.2
    assert result["fraction_at_or_above_onset_0_6"] == 0.6
    assert result["fraction_at_or_above_0_95"] == 0.4
    assert result["fraction_exactly_one_float32"] == 0.2
    for bad in ([], [float("nan")], [-0.01], [1.01], [[0.5]]):
        with pytest.raises(ValueError):
            diagnosis.probability_summary(bad)


def test_feature_summary_uses_source_norm_and_only_selected_windows():
    x = torch.arange(24).float().view(3, 2, 4)
    mean = torch.tensor([1.0, 2.0]).view(1, 2, 1)
    std = torch.tensor([2.0, 3.0]).view(1, 2, 1)
    result = diagnosis.feature_summary(x, np.array([0, 2]), mean, std)
    expected = ((x[[0, 2]] - mean) / std).double().numpy()
    assert result["windows"] == 2
    assert result["frames_per_channel_with_overlap"] == 8
    np.testing.assert_allclose(result["z_mean"], expected.mean(axis=(0, 2)))
    np.testing.assert_allclose(result["z_std"], expected.std(axis=(0, 2)))
    np.testing.assert_allclose(result["fraction_abs_z_over_3"], (np.abs(expected) > 3).mean(axis=(0, 2)))
    with pytest.raises(ValueError, match="empty"):
        diagnosis.feature_summary(x, [], mean, std)
    with pytest.raises(ValueError, match="normalization"):
        diagnosis.feature_summary(x, [0], mean, torch.zeros_like(std))


def test_bn_summary_tracks_initialization_floor_and_rejects_weight_changes():
    before = {"weight": torch.ones(2), "bn.running_mean": torch.zeros(2),
              "bn.running_var": torch.tensor([0.05, 0.1]), "bn.num_batches_tracked": torch.tensor(100)}
    after = {"weight": torch.ones(2), "bn.running_mean": torch.ones(2),
             "bn.running_var": torch.tensor([0.4, 0.5]), "bn.num_batches_tracked": torch.tensor(10)}
    metadata = {"bn": {"momentum": 0.1, "num_batches_tracked": 10}}
    result = diagnosis.bn_summary(before, after, metadata)["bn"]
    assert result["reset_variance_one_additive_floor"] == pytest.approx(0.3486784401)
    assert result["new_over_source_variance_median"] == pytest.approx(6.5)
    assert result["reset_floor_share_of_saved_variance_median"] > 0.7
    after["weight"][0] = 2
    with pytest.raises(ValueError, match="changed"):
        diagnosis.bn_summary(before, after, metadata)
