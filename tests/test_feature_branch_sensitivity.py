"""python -m pytest -q tests/test_feature_branch_sensitivity.py"""
import importlib.util
from pathlib import Path
import sys

import pytest
import torch

from uav_fault.models.cnn.lstm import OnlineKGMSTCA, RFLY_SEQ_COLUMNS_V2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts/research"))
SPEC = importlib.util.spec_from_file_location("feature_branch_sensitivity", ROOT / "scripts/research/check_feature_branch_sensitivity.py")
probe = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(probe)


@pytest.fixture
def model():
    torch.manual_seed(13)
    return OnlineKGMSTCA(in_channels=26, hidden_size=8, num_classes=2, columns=RFLY_SEQ_COLUMNS_V2).eval()


def test_branch_decomposition_reproduces_frozen_fused_logits_without_state_changes(model):
    x = torch.randn(3, 26, 150)
    original = {name: value.clone() for name, value in model.state_dict().items()}
    with torch.no_grad():
        expected = model(x)
        logits, details = probe.explain_batch(model, x)
    torch.testing.assert_close(logits, expected)
    reconstructed = details["raw_margin"] + details["relation_margin"] + details["head_bias"]
    torch.testing.assert_close(reconstructed, logits[:, 1] - logits[:, 0])
    assert details["relation_input_window_mean"].shape == (3, 9)
    assert torch.all((details["gate_alpha"] >= 0) & (details["gate_alpha"] <= 1))
    assert all(torch.equal(value, model.state_dict()[name]) for name, value in original.items())


def test_explanation_rejects_training_before_bn_mutation(model):
    model.train()
    original = {name: value.clone() for name, value in model.state_dict().items()}
    with pytest.raises(ValueError, match="eval-mode"):
        probe.explain_batch(model, torch.randn(3, 26, 150))
    assert all(torch.equal(value, model.state_dict()[name]) for name, value in original.items())


def test_channel_translation_is_selected_reversible_and_does_not_modify_input():
    x = torch.randn(2, 26, 150)
    original = x.clone()
    delta = torch.linspace(-1, 1, 26)
    indices = [2, 5, 8]
    shifted = probe.translate_channels(x, indices, delta, 1)
    assert torch.equal(x, original)
    untouched = [i for i in range(26) if i not in indices]
    assert torch.equal(x[:, untouched], shifted[:, untouched])
    torch.testing.assert_close(shifted[:, indices] - x[:, indices], delta[indices].view(1, -1, 1).expand(2, -1, 150))
    torch.testing.assert_close(probe.translate_channels(shifted, indices, delta, -1), x)
    for bad in ([2, 2], [-1], [26]):
        with pytest.raises(ValueError, match="indices"):
            probe.translate_channels(x, bad, delta, 1)


def test_relation_branch_excludes_magnetometer_and_clamps_normalized_acceleration(model):
    x = torch.randn(2, 26, 150)
    shifted = x.clone()
    for name in ("mag_x", "mag_y", "mag_z"):
        shifted[:, RFLY_SEQ_COLUMNS_V2.index(name)] += 5
    assert torch.equal(model._relation_features(x), model._relation_features(shifted))
    simple = torch.ones(1, 26, 3)
    simple[:, RFLY_SEQ_COLUMNS_V2.index("acc_norm")] = torch.tensor([-2, 0.1, 0.2])
    relation = model._relation_features(simple)
    torch.testing.assert_close(relation[0, 4], torch.tensor([10.0, 10.0, 5.0]))
