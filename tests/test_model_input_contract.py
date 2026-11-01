"""Shared online input validation at discovery and checkpoint-loading entries."""

import copy
import json
import subprocess
import sys
from pathlib import Path

import pytest
import torch

from uav_dashboard_api.services.model_registry import list_registered_models
from uav_fault.model_contract import RFLY_SEQ_SCHEMAS, online_input_contract_errors
from uav_fault.models.cnn.lstm import OnlineCNN1D
from uav_fault.online.fault_detector import FaultDetector


def _contract():
    return (
        {
            "model": "cnn1d",
            "in_channels": 2,
            "num_classes": 2,
            "label_names": ["normal", "fault"],
            "hidden_size": 16,
            "num_layers": 1,
        },
        {"columns": ["roll", "pitch"], "mean": [0.0, 0.0], "std": [1.0, 1.0]},
    )


def _write_release(root, config, stats):
    model_dir = root / "online_test"
    model_dir.mkdir()
    (model_dir / "config.json").write_text(json.dumps(config), encoding="utf-8")
    (model_dir / "norm_stats.json").write_text(json.dumps(stats), encoding="utf-8")
    (model_dir / "best.pt").write_bytes(b"must not be read for invalid metadata")
    return model_dir


@pytest.mark.parametrize(
    "source,field,value,expected",
    [
        ("config", "in_channels", True, "in_channels must be a positive integer"),
        ("config", "in_channels", 2.5, "in_channels must be a positive integer"),
        ("config", "in_channels", "2", "in_channels must be a positive integer"),
        ("config", "in_channels", 3, "in_channels must match columns length"),
        ("config", "num_classes", 3, "num_classes must be 2"),
        ("config", "num_classes", 2.0, "num_classes must be 2"),
        ("config", "num_classes", float("inf"), "num_classes must be 2"),
        ("config", "num_classes", float("nan"), "num_classes must be 2"),
        ("config", "task", "offline_multiclass", "task must be online_binary"),
        ("config", "label_names", ["fault", "normal"], "label_names must be"),
        ("config", "label_names", [], "label_names must be"),
        ("stats", "columns", ["roll", "roll"], "columns must be"),
        ("stats", "columns", ["roll", " "], "columns must be"),
        ("stats", "columns", ["roll", " pitch"], "columns must be"),
        ("stats", "columns", ["roll", {}], "columns must be"),
        ("stats", "columns", 2, "columns must be"),
        ("stats", "mean", [float("nan"), 0], "mean must contain finite"),
        ("stats", "mean", [float("inf"), 0], "mean must contain finite"),
        ("stats", "mean", [1e39, 0], "mean must contain finite"),
        ("stats", "mean", ["0", 0], "mean must contain finite"),
        ("stats", "mean", [True, 0], "mean must contain finite"),
        ("stats", "mean", [[0], [0]], "mean must contain finite"),
        ("stats", "std", [float("inf"), 1], "std must contain finite"),
        ("stats", "std", [float("nan"), 1], "std must contain finite"),
        ("stats", "std", [0, 1], "std must contain finite"),
        ("stats", "std", [-1, 1], "std must contain finite"),
        ("stats", "std", [1e-50, 1], "std must contain finite"),
        ("stats", "std", [1], "std must match columns length"),
        ("config", "feature_count", 3, "config.feature_count must match"),
        ("stats", "feature_count", False, "norm_stats.feature_count must match"),
        ("config", "window_size", 0, "config.window_size must be"),
        ("stats", "window_size", 150.5, "norm_stats.window_size must be"),
        ("config", "protocol_name", {}, "config.protocol_name must be"),
        ("stats", "feature_schema", "", "norm_stats.feature_schema must be"),
        ("stats", "feature_schema", "rfly_seq_columns_v1", "feature_schema does not match columns order"),
    ],
)
def test_invalid_contract_is_rejected_by_registry_and_loader(
    tmp_path, monkeypatch, source, field, value, expected
):
    config, stats = _contract()
    (config if source == "config" else stats)[field] = value
    errors = online_input_contract_errors(config, stats)
    assert any(expected in error for error in errors)
    model_dir = _write_release(tmp_path, config, stats)

    metadata = list_registered_models({"MODEL_ROOT": str(tmp_path)})[0]
    assert metadata["input_contract_errors"] == errors
    assert metadata["capabilities"]["realtime_detection"] is False
    assert metadata["capabilities"]["fault_prob_curve"] is False
    assert list_registered_models({"MODEL_ROOT": str(tmp_path)}, capability="realtime_detection") == []

    def unexpected_checkpoint_read(*args, **kwargs):
        pytest.fail("invalid metadata must be rejected before checkpoint loading")

    monkeypatch.setattr(torch, "load", unexpected_checkpoint_read)
    with pytest.raises(ValueError, match="Invalid online model input contract") as exc:
        FaultDetector(str(model_dir))
    assert "; ".join(errors) in str(exc.value)


@pytest.mark.parametrize("field,left,right", [
    ("window_size", 150, 100),
    ("feature_schema", "custom_v1", "custom_v2"),
    ("protocol_name", "old_protocol", "new_protocol"),
])
def test_conflicting_version_declarations_are_rejected(field, left, right):
    config, stats = _contract()
    config[field], stats[field] = left, right
    assert f"{field} differs between config and norm_stats" in online_input_contract_errors(config, stats)


@pytest.mark.parametrize("config,stats", [(None, {}), ([], {}), ({}, None), ({}, [])])
def test_metadata_must_be_json_objects(config, stats):
    assert online_input_contract_errors(config, stats)


def test_legacy_metadata_and_custom_column_order_remain_supported(tmp_path):
    config, stats = _contract()
    config.pop("label_names")
    stats["columns"] = ["pitch", "roll"]
    assert online_input_contract_errors(config, stats) == []
    _write_release(tmp_path, config, stats)
    metadata = list_registered_models({"MODEL_ROOT": str(tmp_path)}, capability="realtime_detection")[0]
    assert metadata["input_contract_errors"] == []
    assert metadata["feature_count"] == 2


@pytest.mark.parametrize("schema", list(RFLY_SEQ_SCHEMAS))
def test_declared_schema_checks_order_without_modifying_metadata(schema):
    config, stats = _contract()
    columns = RFLY_SEQ_SCHEMAS[schema]
    config.update(in_channels=len(columns), feature_schema=schema, window_size=150)
    stats.update(columns=list(columns), mean=[0] * len(columns), std=[1] * len(columns), feature_schema=schema)
    original = copy.deepcopy((config, stats))
    assert online_input_contract_errors(config, stats) == []
    assert (config, stats) == original
    stats["columns"][0], stats["columns"][1] = stats["columns"][1], stats["columns"][0]
    assert any("columns order" in error for error in online_input_contract_errors(config, stats))


def test_26_channel_release_loads_and_predicts_with_checkpoint(tmp_path):
    config, stats = _contract()
    columns = RFLY_SEQ_SCHEMAS["rfly_seq_columns_v2"]
    config.update(in_channels=len(columns), feature_schema="rfly_seq_columns_v2", window_size=150)
    stats.update(columns=columns, mean=[0] * len(columns), std=[1] * len(columns), window_size=150)
    model_dir = _write_release(tmp_path, config, stats)
    model = OnlineCNN1D(in_channels=len(columns), hidden_size=16, num_layers=1, num_classes=2)
    torch.save(model.state_dict(), model_dir / "best.pt")
    detector = FaultDetector(str(model_dir), window_size=None)
    assert detector.columns == columns
    assert detector.window_size == 150
    result = detector.predict([{} for _ in range(150)])
    assert 0 <= result["fault_prob"] <= 1


def test_deployed_online_releases_pass_input_contract():
    root = Path(__file__).resolve().parents[1] / "assets" / "models"
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    for entry in manifest["models"]:
        if entry["runtime"] != "online":
            continue
        directory = root / entry["path"]
        config = json.loads((directory / "config.json").read_text(encoding="utf-8"))
        stats = json.loads((directory / "norm_stats.json").read_text(encoding="utf-8"))
        assert online_input_contract_errors(config, stats) == [], directory.name


def test_contract_and_registry_import_do_not_load_torch():
    subprocess.run([
        sys.executable, "-c",
        "import sys; from uav_dashboard_api.services import model_registry; "
        "assert 'torch' not in sys.modules; "
        "assert 'numpy' not in sys.modules",
    ], check=True)
