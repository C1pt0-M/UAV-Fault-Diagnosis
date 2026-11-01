"""Offline compatibility and manifest validation through runtime entry points."""

import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from uav_dashboard_api.services.model_registry import list_registered_models, resolve_registered_model
from uav_fault.data.build_sequence_cache import SEQ_COLUMNS, SEQ_LEN, load_and_resample
from uav_fault.model_contract import RFLY_SEQ_COLUMNS_V1

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "apps/dashboard/backend/scripts/offline_multiclass_infer.py"


def _json(path, data):
    path.write_text(json.dumps(data), encoding="utf-8")


def _offline_script():
    spec = importlib.util.spec_from_file_location("offline_contract_test", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _offline_release(root):
    directory = root / "offline_test"
    directory.mkdir()
    _json(directory / "config.json", {
        "model": "cnn_lstm_v2_normalproto", "in_channels": 31,
        "num_classes": 3, "label_names": ["normal", "motor", "sensor"],
    })
    (directory / "best.pt").write_bytes(b"unused checkpoint")
    return directory


def _stats():
    return {"columns": list(SEQ_COLUMNS), "mean": [0.0] * 31, "std": [1.0] * 31}


@pytest.mark.parametrize("field,value", [
    ("columns", list(reversed(SEQ_COLUMNS))),
    ("mean", [0] * 30),
    ("std", [1] * 32),
    ("mean", [float("nan")] * 31),
    ("std", [float("inf")] * 31),
    ("mean", [1e39] * 31),
    ("std", [-1] * 31),
    ("mean", [[0]] * 31),
    ("mean", [True] * 31),
])
def test_invalid_offline_stats_disable_registry_and_never_fall_back(tmp_path, field, value):
    model_dir = _offline_release(tmp_path)
    stats = _stats()
    stats[field] = value
    selected = model_dir / "norm_stats.json"
    _json(selected, stats)
    fallback = tmp_path / "fallback.json"
    _json(fallback, _stats())

    metadata = list_registered_models({"MODEL_ROOT": str(tmp_path)})[0]
    assert metadata["input_contract_errors"]
    assert not metadata["capabilities"]["offline_detection"]
    assert resolve_registered_model({"MODEL_ROOT": str(tmp_path)}, capability="offline_detection") is None
    with pytest.raises(ValueError) as exc:
        _offline_script().load_norm_stats([selected, fallback], SEQ_COLUMNS)
    assert str(selected) in str(exc.value)
    assert any(error in str(exc.value) for error in metadata["input_contract_errors"])


@pytest.mark.parametrize("content", ["{bad json", "[]", "null"])
def test_unreadable_or_nonobject_stats_do_not_fall_back(tmp_path, content):
    model_dir = _offline_release(tmp_path)
    selected = model_dir / "norm_stats.json"
    selected.write_text(content, encoding="utf-8")
    metadata = list_registered_models({"MODEL_ROOT": str(tmp_path)})[0]
    assert not metadata["capabilities"]["offline_detection"]
    with pytest.raises(ValueError):
        _offline_script().load_norm_stats([selected], SEQ_COLUMNS)


def test_legacy_stats_without_columns_keep_floor_and_candidate_priority(tmp_path):
    stats = _stats()
    stats.pop("columns")
    stats["std"][:3] = [0, 1e-8, 2]
    selected, fallback = tmp_path / "selected.json", tmp_path / "fallback.json"
    _json(selected, stats)
    _json(fallback, _stats())
    mean, std, source = _offline_script().load_norm_stats([tmp_path / "missing", None, selected, fallback], SEQ_COLUMNS)
    assert source == str(selected)
    np.testing.assert_array_equal(mean, np.zeros(31))
    np.testing.assert_array_equal(std[:3], [1, 1, 2])
    assert _offline_script().load_norm_stats([tmp_path / "missing", None], SEQ_COLUMNS) == (None, None, "")


@pytest.mark.parametrize("mode", ["model_stats", "configured_stats", "sample_zscore"])
def test_offline_entry_preserves_normalization_and_whole_log_preprocessing(tmp_path, monkeypatch, mode):
    module = _offline_script()
    model_dir = _offline_release(tmp_path)
    csv = tmp_path / "flight.csv"
    csv.write_text("gyro_x,accel_z\n0,-9.8\n1,-9.7\n2,-9.6\n3,-9.5\n", encoding="utf-8")
    configured = tmp_path / "configured.json"
    stats = _stats()
    stats["mean"] = [0.5] * 31
    stats["std"] = [2.0] * 31
    if mode == "model_stats":
        _json(model_dir / "norm_stats.json", stats)
        _json(configured, _stats())
        expected_source = str(model_dir / "norm_stats.json")
    elif mode == "configured_stats":
        _json(configured, stats)
        expected_source = str(configured)
    else:
        expected_source = "sample_zscore"

    inputs = []

    class CaptureModel(torch.nn.Module):
        def forward(self, x):
            inputs.append(x.clone())
            return torch.tensor([[0.0, 1.0, 2.0]])

    monkeypatch.setattr(module, "create_model", lambda *args, **kwargs: CaptureModel())
    monkeypatch.setattr(torch, "load", lambda *args, **kwargs: {})
    args = SimpleNamespace(repo_root=str(tmp_path), csv_path=str(csv), model_dir=str(model_dir),
                           output_json=str(tmp_path / "result.json"), norm_stats=str(configured), device="cpu")
    result = module.run_inference(args)
    assert result["debug"]["norm_source"] == expected_source
    assert inputs[0].shape == (1, 31, SEQ_LEN)
    raw = torch.from_numpy(load_and_resample(str(csv), seq_len=SEQ_LEN).T)
    if mode == "sample_zscore":
        std = raw.std(dim=1, keepdim=True)
        expected = (raw - raw.mean(dim=1, keepdim=True)) / torch.where(std < 1e-6, torch.ones_like(std), std)
    else:
        expected = (raw - 0.5) / 2.0
    torch.testing.assert_close(inputs[0][0], expected)
    assert result["labels"] == ["normal", "motor", "sensor"]


@pytest.mark.parametrize("channels", [26, 43, True, "31"])
def test_offline_entry_and_registry_reject_incompatible_channel_counts(tmp_path, channels):
    model_dir = _offline_release(tmp_path)
    config = json.loads((model_dir / "config.json").read_text())
    config["in_channels"] = channels
    _json(model_dir / "config.json", config)
    metadata = list_registered_models({"MODEL_ROOT": str(tmp_path)})[0]
    assert not metadata["capabilities"]["offline_detection"]
    csv = tmp_path / "flight.csv"
    csv.write_text("gyro_x\n0\n1\n", encoding="utf-8")
    args = SimpleNamespace(repo_root=str(tmp_path), csv_path=str(csv), model_dir=str(model_dir),
                           output_json=str(tmp_path / "result.json"), norm_stats="", device="cpu")
    with pytest.raises(ValueError, match="in_channels must be 31"):
        _offline_script().run_inference(args)
    assert not Path(args.output_json).exists()


@pytest.mark.parametrize("classes,expected", [
    (3, ["normal", "motor", "sensor"]),
    (4, ["normal", "motor", "propeller", "sensor"]),
    (5, ["normal", "battery_failure", "motor_failure", "prop_imbalance", "thrust_loss"]),
    (None, ["normal", "battery_failure", "motor_failure", "prop_imbalance", "thrust_loss"]),
    (6, ["normal", "battery_failure", "motor_failure", "prop_imbalance", "thrust_loss", "class_5"]),
])
def test_legacy_offline_class_labels_and_task_alias_remain_supported(tmp_path, monkeypatch, classes, expected):
    model_dir = _offline_release(tmp_path)
    config = {"model": "cnn_lstm_v2_normalproto", "in_channels": 31,
              "task": "offline_5class", "num_classes": classes}
    if classes is None:
        config.pop("num_classes")
    _json(model_dir / "config.json", config)
    module = _offline_script()
    result = _stubbed_prediction(module, model_dir, tmp_path, monkeypatch, len(expected))
    metadata = list_registered_models({"MODEL_ROOT": str(tmp_path)}, capability="offline_detection")[0]
    assert metadata["num_classes"] == len(expected)
    assert metadata["label_names"] == result["labels"] == expected


def test_deployed_offline_model_runs_in_isolated_subprocess(tmp_path):
    assert SEQ_COLUMNS == RFLY_SEQ_COLUMNS_V1
    csv, output = tmp_path / "flight.csv", tmp_path / "prediction.json"
    csv.write_text("gyro_x,accel_z\n0,-9.8\n1,-9.7\n2,-9.6\n", encoding="utf-8")
    model_dir = ROOT / "assets/models/transfer3_cnnlstm_normalproto"
    completed = subprocess.run([
        sys.executable, "-I", str(SCRIPT), "--repo-root", str(ROOT),
        "--model-dir", str(model_dir), "--csv-path", str(csv),
        "--output-json", str(output),
    ], capture_output=True, text=True, timeout=60)
    assert completed.returncode == 0, completed.stderr
    result = json.loads(output.read_text(encoding="utf-8"))
    assert result["debug"]["norm_source"] == str(model_dir / "norm_stats.json")
    assert result["labels"] == ["normal", "motor", "sensor"]
    assert sum(result["probs"].values()) == pytest.approx(1)


def _online_release(root):
    directory = root / "online_test"
    directory.mkdir()
    _json(directory / "config.json", {"model": "cnn1d", "in_channels": 2, "num_classes": 2})
    _json(directory / "norm_stats.json", {"columns": ["roll", "pitch"], "mean": [0, 0], "std": [1, 1]})
    (directory / "best.pt").write_bytes(b"metadata test")
    return directory


@pytest.mark.parametrize("manifest", [
    [], {}, {"version": 2, "models": []}, {"version": True, "models": []},
    {"version": 1, "models": {}}, {"version": 1, "models": [None]},
    {"version": 1, "models": [{}]},
])
def test_malformed_manifest_fails_instead_of_becoming_empty_registry(tmp_path, manifest):
    _json(tmp_path / "manifest.json", manifest)
    with pytest.raises(ValueError):
        list_registered_models({"MODEL_ROOT": str(tmp_path)})


@pytest.mark.parametrize("path", ["../outside", "online_test/../online_test", "/outside", "C:/outside", "C:outside", "\\\\server\\share", ".", "", 42])
def test_manifest_paths_cannot_escape_or_replace_model_root(tmp_path, path):
    _json(tmp_path / "manifest.json", {"version": 1, "models": [{"path": path}]})
    with pytest.raises(ValueError):
        list_registered_models({"MODEL_ROOT": str(tmp_path)})


@pytest.mark.parametrize("entry", [
    {"name": "different_name", "path": "online_test"},
    {"path": "online_test", "runtime": "invalid"},
    {"path": "online_test", "runtime": None},
    {"path": "online_test", "runtime": "offline"},
])
def test_manifest_identity_and_runtime_must_agree(tmp_path, entry):
    _online_release(tmp_path)
    _json(tmp_path / "manifest.json", {"version": 1, "models": [entry]})
    with pytest.raises(ValueError):
        list_registered_models({"MODEL_ROOT": str(tmp_path)})


def test_duplicate_manifest_entries_fail(tmp_path):
    _online_release(tmp_path)
    _json(tmp_path / "manifest.json", {"version": 1, "models": [{"path": "online_test"}] * 2})
    with pytest.raises(ValueError, match="Duplicate"):
        list_registered_models({"MODEL_ROOT": str(tmp_path)})


@pytest.mark.parametrize("filename", ["config.json", "best.pt"])
def test_manifest_missing_required_files_fail(tmp_path, filename):
    directory = _online_release(tmp_path)
    (directory / filename).unlink()
    _json(tmp_path / "manifest.json", {"version": 1, "models": [{"name": "online_test"}]})
    with pytest.raises(FileNotFoundError, match=filename):
        list_registered_models({"MODEL_ROOT": str(tmp_path)})


def test_manifest_symlink_cannot_escape_root(tmp_path):
    root, outside = tmp_path / "models", tmp_path / "outside"
    root.mkdir()
    outside.mkdir()
    try:
        (root / "online_test").symlink_to(outside, target_is_directory=True)
    except OSError:
        pytest.skip("directory symlink creation is not permitted on this host")
    _json(root / "manifest.json", {"version": 1, "models": [{"path": "online_test"}]})
    with pytest.raises(ValueError, match="inside MODEL_ROOT"):
        list_registered_models({"MODEL_ROOT": str(root)})


def test_legacy_manifest_omissions_and_whitelist_remain_supported(tmp_path):
    _online_release(tmp_path)
    _json(tmp_path / "manifest.json", {"version": 1, "models": [{"name": "online_test"}]})
    config = {"MODEL_ROOT": str(tmp_path)}
    assert [m["name"] for m in list_registered_models(config)] == ["online_test"]
    config["MODEL_WHITELIST"] = "other"
    assert list_registered_models(config) == []


def test_current_manifest_has_all_four_available_models():
    models = list_registered_models({"MODEL_ROOT": str(ROOT / "assets/models")})
    assert len(models) == 4
    assert all(not model["input_contract_errors"] for model in models)
    assert sum(model["capabilities"]["offline_detection"] for model in models) == 1
    assert sum(model["capabilities"]["realtime_detection"] for model in models) == 3


def _offline_args(root, model_dir):
    csv = root / "contract_input.csv"
    csv.write_text("gyro_x,accel_z\n0,-9.8\n1,-9.7\n2,-9.6\n", encoding="utf-8")
    return SimpleNamespace(repo_root=str(root), csv_path=str(csv), model_dir=str(model_dir),
                           output_json=str(root / "contract_output.json"), norm_stats="", device="cpu")


def _stubbed_prediction(module, model_dir, root, monkeypatch, classes=3):
    class OutputModel(torch.nn.Module):
        def forward(self, x):
            return torch.arange(classes, dtype=torch.float32).unsqueeze(0)

    monkeypatch.setattr(module, "create_model", lambda *args, **kwargs: OutputModel())
    monkeypatch.setattr(torch, "load", lambda *args, **kwargs: {})
    return module.run_inference(_offline_args(root, model_dir))


def _assert_offline_rejected(root, monkeypatch, config, stats):
    model_dir = _offline_release(root)
    _json(model_dir / "config.json", config)
    _json(model_dir / "norm_stats.json", stats)
    args = _offline_args(root, model_dir)
    Path(args.output_json).write_text("keep previous result", encoding="utf-8")
    metadata = list_registered_models({"MODEL_ROOT": str(root)})[0]
    assert metadata["input_contract_errors"]
    assert not metadata["capabilities"]["offline_detection"]
    assert resolve_registered_model({"MODEL_ROOT": str(root)}, capability="offline_detection") is None

    def unexpected_load(*args, **kwargs):
        pytest.fail("invalid metadata must be rejected before reading a checkpoint")

    monkeypatch.setattr(torch, "load", unexpected_load)
    with pytest.raises(ValueError) as exc:
        _offline_script().run_inference(args)
    assert all(error in str(exc.value) for error in metadata["input_contract_errors"])
    assert Path(args.output_json).read_text(encoding="utf-8") == "keep previous result"


@pytest.mark.parametrize("labels", [
    [], None, "normal,motor,sensor", ["normal", "fault"],
    ["normal", "motor", "motor"], ["normal", "motor", " motor "],
    ["normal", "motor", " "], ["normal", "motor", 2],
])
def test_invalid_offline_labels_rejected_without_fallback_or_output_overwrite(tmp_path, monkeypatch, labels):
    config = {"model": "cnn_lstm_v2_normalproto", "in_channels": 31,
              "num_classes": 3, "label_names": labels}
    _assert_offline_rejected(tmp_path, monkeypatch, config, _stats())


@pytest.mark.parametrize("labels,expected", [
    ([" normal ", " motor ", "sensor", "unused"], ["normal", "motor", "sensor"]),
    (["sensor", "normal", "motor"], ["sensor", "normal", "motor"]),
])
def test_explicit_label_order_trim_and_legacy_extra_labels_match_runtime(tmp_path, monkeypatch, labels, expected):
    model_dir = _offline_release(tmp_path)
    _json(model_dir / "config.json", {"model": "cnn_lstm_v2_normalproto", "in_channels": 31,
                                      "num_classes": 3, "label_names": labels})
    result = _stubbed_prediction(_offline_script(), model_dir, tmp_path, monkeypatch)
    metadata = list_registered_models({"MODEL_ROOT": str(tmp_path)}, capability="offline_detection")[0]
    assert metadata["label_names"] == result["labels"] == list(result["probs"]) == expected
    assert result["prediction"]["label"] == expected[-1]
    assert result["topk"][0]["label"] == expected[-1]


@pytest.mark.parametrize("source,field,value", [
    ("config", "feature_schema", "rfly_seq_columns_v2"),
    ("stats", "feature_schema", "rfly_seq_columns_v2"),
    ("config", "feature_schema", ""),
    ("stats", "feature_schema", {}),
    ("config", "protocol_name", None),
    ("stats", "protocol_name", " "),
    ("config", "protocol_name", " protocol_v1 "),
    ("config", "feature_count", 26),
    ("stats", "feature_count", True),
    ("stats", "feature_count", "31"),
])
def test_offline_feature_declarations_are_validated_by_both_entries(tmp_path, monkeypatch, source, field, value):
    config = {"model": "cnn_lstm_v2_normalproto", "in_channels": 31, "num_classes": 3}
    stats = _stats()
    (config if source == "config" else stats)[field] = value
    _assert_offline_rejected(tmp_path, monkeypatch, config, stats)


@pytest.mark.parametrize("field", ["feature_schema", "protocol_name"])
@pytest.mark.parametrize("configured", [False, True])
def test_mismatched_offline_versions_never_select_another_statistics_file(tmp_path, monkeypatch, field, configured):
    model_dir = _offline_release(tmp_path)
    config = {"model": "cnn_lstm_v2_normalproto", "in_channels": 31, "num_classes": 3, field: "custom_v1"}
    stats = _stats()
    stats[field] = "custom_v2"
    _json(model_dir / "config.json", config)
    selected = tmp_path / "configured.json" if configured else model_dir / "norm_stats.json"
    _json(selected, stats)
    fallback = tmp_path / "fallback.json"
    _json(fallback, _stats())
    expected_error = f"{field} differs between config and norm_stats"
    if not configured:
        metadata = list_registered_models({"MODEL_ROOT": str(tmp_path)})[0]
        assert expected_error in metadata["input_contract_errors"]
        assert not metadata["capabilities"]["offline_detection"]
    with pytest.raises(ValueError, match=expected_error):
        _offline_script().load_norm_stats([selected, fallback], SEQ_COLUMNS, model_cfg=config)
    args = _offline_args(tmp_path, model_dir)
    args.norm_stats = str(selected) if configured else ""
    monkeypatch.setattr(torch, "load", lambda *args, **kwargs: pytest.fail("checkpoint must not be read"))
    with pytest.raises(ValueError, match=expected_error):
        _offline_script().run_inference(args)
    assert not Path(args.output_json).exists()


@pytest.mark.parametrize("declaration", ["missing", "matching", "one_sided", "custom"])
def test_optional_offline_version_declarations_remain_compatible(tmp_path, monkeypatch, declaration):
    model_dir = _offline_release(tmp_path)
    config = {"model": "cnn_lstm_v2_normalproto", "in_channels": 31, "num_classes": 3}
    stats = _stats()
    stats.pop("columns")  # Historical positional statistics remain supported.
    if declaration == "matching":
        config.update(feature_schema="rfly_seq_columns_v1", protocol_name="legacy_protocol")
        stats.update(feature_schema="rfly_seq_columns_v1", protocol_name="legacy_protocol")
    elif declaration == "one_sided":
        config["feature_schema"] = "rfly_seq_columns_v1"
        stats["protocol_name"] = "legacy_protocol"
    elif declaration == "custom":
        config["feature_schema"] = stats["feature_schema"] = "custom_schema_v1"
    _json(model_dir / "config.json", config)
    _json(model_dir / "norm_stats.json", stats)
    metadata = list_registered_models({"MODEL_ROOT": str(tmp_path)}, capability="offline_detection")[0]
    assert metadata["input_contract_errors"] == []
    result = _stubbed_prediction(_offline_script(), model_dir, tmp_path, monkeypatch)
    assert result["debug"]["norm_source"] == str(model_dir / "norm_stats.json")


def test_invalid_config_version_is_rejected_even_when_all_statistics_are_missing(tmp_path, monkeypatch):
    model_dir = _offline_release(tmp_path)
    _json(model_dir / "config.json", {"model": "cnn_lstm_v2_normalproto", "in_channels": 31,
                                      "num_classes": 3, "feature_schema": "rfly_seq_columns_v2"})
    metadata = list_registered_models({"MODEL_ROOT": str(tmp_path)})[0]
    assert not metadata["capabilities"]["offline_detection"]
    monkeypatch.setattr(torch, "load", lambda *args, **kwargs: pytest.fail("checkpoint must not be read"))
    with pytest.raises(ValueError, match="feature_schema"):
        _offline_script().run_inference(_offline_args(tmp_path, model_dir))
