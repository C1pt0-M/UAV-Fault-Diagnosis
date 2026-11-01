"""Run with: python -m pytest -q tests/test_run_normal_adaptation_v1.py"""
import importlib.util
from pathlib import Path

import pytest
import torch

_SPEC = importlib.util.spec_from_file_location(
    "adaptation_runner", Path(__file__).resolve().parents[1] / "scripts/research/run_normal_adaptation_v1.py"
)
runner = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(runner)


def test_recalibration_changes_only_bn_statistics_and_disables_dropout():
    torch.manual_seed(7)
    model = torch.nn.Sequential(torch.nn.Conv1d(2, 3, 1), torch.nn.BatchNorm1d(3), torch.nn.Dropout(0.9))
    initial = {name: value.clone() for name, value in model.state_dict().items()}
    seen = []
    handle = model.register_forward_pre_hook(lambda mod, inputs: seen.append((mod[1].training, mod[2].training, torch.is_grad_enabled())))
    x = torch.randn(65, 2, 10) + 5
    result = runner.recalibrate(model, x, torch.device("cpu"))
    handle.remove()
    assert seen == [(True, False, False), (True, False, False)]
    assert result["1"]["num_batches_tracked"] == 2
    assert result["1"]["momentum"] == 0.1
    assert not torch.equal(initial["1.running_mean"], model[1].running_mean)
    for name, parameter in model.named_parameters():
        assert torch.equal(initial[name], parameter)
        assert not parameter.requires_grad
        assert parameter.grad is None
    assert not any(layer.training for layer in model.modules())
    second = torch.nn.Sequential(torch.nn.Conv1d(2, 3, 1), torch.nn.BatchNorm1d(3), torch.nn.Dropout(0.9))
    second.load_state_dict(initial)
    runner.recalibrate(second, x, torch.device("cpu"))
    assert all(torch.equal(a, b) for a, b in zip(model.state_dict().values(), second.state_dict().values()))


def test_recalibration_rejects_missing_batchnorm():
    with pytest.raises(ValueError, match="BatchNorm"):
        runner.recalibrate(torch.nn.Conv1d(2, 3, 1), torch.zeros(2, 2, 10), torch.device("cpu"))


def make_bundle():
    rows = [{"case_uid": f"case{i:02}", "row_count": "150", "output_sha256": str(i)} for i in range(10)]
    columns = [str(i) for i in range(26)]
    bundle = {"x": torch.zeros(10, 26, 150), "columns": columns,
              "protocol_sha256": runner.PROTOCOL_SHA,
              "samples": [{"case_uid": row["case_uid"], "start_row": 0, "end_row": 150,
                           "source_csv_sha256": row["output_sha256"]} for row in rows]}
    return bundle, rows, columns


def test_bundle_validates_role_membership_order_shape_and_finiteness():
    bundle, rows, columns = make_bundle()
    assert len(runner.validate_bundle(bundle, rows, columns)) == 10
    bundle["samples"][0]["case_uid"] = "fault_case_not_allowed"
    with pytest.raises(ValueError, match="allowed cases"):
        runner.validate_bundle(bundle, rows, columns)
    bundle, rows, columns = make_bundle()
    bundle["x"][0, 0, 0] = float("nan")
    with pytest.raises(ValueError, match="non-finite"):
        runner.validate_bundle(bundle, rows, columns)
    bundle, rows, columns = make_bundle()
    bundle["samples"] = bundle["samples"][::-1]
    with pytest.raises(ValueError, match="ordered"):
        runner.validate_bundle(bundle, rows, columns)
    bundle, rows, columns = make_bundle()
    bundle["samples"][0]["end_row"] = 149
    with pytest.raises(ValueError, match="span"):
        runner.validate_bundle(bundle, rows, columns)
    bundle, rows, columns = make_bundle()
    bundle["samples"][0]["source_csv_sha256"] = "tampered"
    with pytest.raises(ValueError, match="identity"):
        runner.validate_bundle(bundle, rows, columns)


def test_frozen_roles_and_hash_guard():
    root = Path(__file__).resolve().parents[1]
    artifacts = root / "artifacts/normal_adaptation_audit_20261029"
    protocol, rows = runner.load_roles(artifacts / "normal_adaptation_protocol_v1.json", artifacts / "real_case_roles_v1.csv")
    assert protocol["status"] == "FROZEN"
    assert len(rows) == 10
    assert all(row["role"] == "adapt_normal" for row in rows)
    with pytest.raises(ValueError, match="SHA-256"):
        runner.require_hash(artifacts / "real_case_roles_v1.csv", "wrong")


def test_source_validation_metrics(tmp_path):
    cache = tmp_path / "val.pt"
    x = torch.zeros(4, 26, 150)
    x[:, 0, 0] = torch.tensor([-2.0, 2.0, -2.0, 2.0])
    torch.save({"x": x, "y": torch.tensor([0, 0, 1, 1])}, cache)

    class Model(torch.nn.Module):
        def forward(self, batch):
            value = batch[:, 0, 0]
            return torch.stack([-value, value], dim=1)

    result = runner.validate_source(Model(), cache, torch.zeros(1, 26, 1), torch.ones(1, 26, 1), torch.device("cpu"), tmp_path / "probs.npz")
    assert result["confusion_matrix"] == [[1, 1], [1, 1]]
    assert result["accuracy"] == result["macro_f1"] == result["fault_recall"] == result["normal_window_fpr"] == 0.5
    torch.save({"x": x, "y": torch.zeros(4, dtype=torch.long)}, cache)
    result = runner.validate_source(Model(), cache, torch.zeros(1, 26, 1), torch.ones(1, 26, 1), torch.device("cpu"), tmp_path / "normal_probs.npz")
    assert result["fault_recall"] is None
    assert result["normal_window_fpr"] == 0.5
