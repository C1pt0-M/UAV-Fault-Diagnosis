"""CPU estimator checks; python -m pytest -q tests/test_check_bn_estimator_dev_v1.py"""
import importlib.util
from pathlib import Path
import sys

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts/research"))
SPEC = importlib.util.spec_from_file_location("bn_estimator_development", ROOT / "scripts/research/check_bn_estimator_dev_v1.py")
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)


def test_cumulative_removes_initialization_and_averages_within_batch_moments():
    model = torch.nn.Sequential(torch.nn.BatchNorm1d(2), torch.nn.Dropout(0.9))
    model[0].running_mean.fill_(100)
    model[0].running_var.fill_(100)
    first = torch.arange(64 * 2 * 4).float().view(64, 2, 4) / 100
    second = first + 10
    x = torch.cat([first, second])
    parameters = {key: value.clone() for key, value in model.named_parameters()}
    bn = runner.recalibrate_cumulative(model, x)["0"]
    expected_mean = (first.mean((0, 2)) + second.mean((0, 2))) / 2
    expected_var = (first.var((0, 2), unbiased=True) + second.var((0, 2), unbiased=True)) / 2
    torch.testing.assert_close(model[0].running_mean, expected_mean)
    torch.testing.assert_close(model[0].running_var, expected_var)
    assert torch.all(x.var((0, 2), unbiased=True) > model[0].running_var)  # not pooled variance
    assert bn["momentum"] is None and bn["num_batches_tracked"] == 2
    for key, value in model.named_parameters():
        assert torch.equal(value, parameters[key]) and not value.requires_grad and value.grad is None
    assert not any(layer.training for layer in model.modules())


def test_only_estimator_differs_not_training_batch_activations():
    torch.manual_seed(9)
    source = torch.nn.Sequential(torch.nn.Conv1d(2, 4, 1), torch.nn.BatchNorm1d(4), torch.nn.Dropout(0.8))
    x = torch.randn(128, 2, 8)
    models, activations = [], []
    for cumulative in (False, True):
        model = torch.nn.Sequential(torch.nn.Conv1d(2, 4, 1), torch.nn.BatchNorm1d(4), torch.nn.Dropout(0.8))
        model.load_state_dict(source.state_dict())
        captured = []
        handle = model.register_forward_hook(lambda module, inputs, output: captured.append(output.detach().clone()))
        if cumulative:
            runner.recalibrate_cumulative(model, x)
        else:
            runner.recalibrate(model, x, torch.device("cpu"))
        handle.remove()
        models.append(model)
        activations.append(captured)
    assert all(torch.equal(a, b) for a, b in zip(*activations))
    assert not torch.equal(models[0][1].running_var, models[1][1].running_var)
    assert torch.equal(models[0][0].weight, models[1][0].weight)


def test_cumulative_still_rejects_models_without_bn():
    with pytest.raises(ValueError, match="BatchNorm"):
        runner.recalibrate_cumulative(torch.nn.Linear(2, 2), torch.zeros(2, 2))
