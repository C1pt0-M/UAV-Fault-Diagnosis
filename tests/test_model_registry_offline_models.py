from __future__ import annotations

import ast
import json
from pathlib import Path

from uav_dashboard_api.services.model_registry import list_registered_models

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def test_model_registry_includes_external_metrics_and_new_architectures(tmp_path: Path) -> None:
    model_root = tmp_path / "models"

    for name, model_name, architecture, macro_test, macro_external in [
        ("transfer3_cnnlstm_v2_normalproto", "cnn_lstm_v2_normalproto", "cnn_lstm_v2_normalproto", 0.987, 0.784),
    ]:
        model_dir = model_root / name
        model_dir.mkdir(parents=True)
        (model_dir / "best.pt").write_bytes(b"pt")
        _write_json(
            model_dir / "config.json",
            {
                "model": model_name,
                "architecture": architecture,
                "task": "offline_multiclass",
                "num_classes": 3,
                "label_names": ["normal", "motor", "sensor"],
                "best_val_f1": 0.95,
            },
        )
        _write_json(
            model_dir / "metrics_test.json",
            {
                "accuracy": 0.99,
                "macro_f1": macro_test,
                "weighted_f1": 0.98,
            },
        )
        _write_json(
            model_dir / "metrics_external_test.json",
            {
                "accuracy": 0.86,
                "macro_f1": macro_external,
                "weighted_f1": 0.85,
                "balanced_accuracy": 0.76,
            },
        )
        _write_json(
            model_dir / "norm_stats.json",
            {"columns": [f"feat_{i}" for i in range(31)]},
        )

    models = list_registered_models(
        {"MODEL_ROOT": str(model_root), "MODEL_WHITELIST": ""},
        task="offline_multiclass",
        capability="offline_detection",
    )
    names = [item["name"] for item in models]
    assert "transfer3_cnnlstm_v2_normalproto" in names

    top = next(item for item in models if item["name"] == "transfer3_cnnlstm_v2_normalproto")
    assert top["capabilities"]["offline_detection"] is True
    assert top["metrics"]["macro_f1"] == 0.987
    assert top["metrics_external"]["macro_f1"] == 0.784
    assert top["metrics_external"]["balanced_accuracy"] == 0.76


def test_offline_multiclass_infer_supports_transfer3_model_names() -> None:
    script_path = (
        PROJECT_ROOT
        / "apps"
        / "dashboard"
        / "backend"
        / "scripts"
        / "offline_multiclass_infer.py"
    )
    tree = ast.parse(script_path.read_text(encoding="utf-8"))

    create_model = None
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == "create_model":
            create_model = node
            break
    assert create_model is not None

    constants = {node.value for node in ast.walk(create_model) if isinstance(node, ast.Constant)}
    assert "cnn_lstm_v2_normalproto" in constants


def test_offline_infer_prefers_selected_model_norm_stats() -> None:
    script_path = (
        PROJECT_ROOT
        / "apps"
        / "dashboard"
        / "backend"
        / "scripts"
        / "offline_multiclass_infer.py"
    )
    source = script_path.read_text(encoding="utf-8")

    model_stats_index = source.index("model_dir / 'norm_stats.json'")
    configured_stats_index = source.index("Path(args.norm_stats) if args.norm_stats else None")
    assert model_stats_index < configured_stats_index


def test_model_registry_accepts_online_kg_mstca(tmp_path: Path) -> None:
    model_root = tmp_path / "models"
    model_dir = model_root / "online_detector_kg_mstca_rfly"
    model_dir.mkdir(parents=True)
    (model_dir / "best.pt").write_bytes(b"pt")

    _write_json(
        model_dir / "config.json",
        {
            "model": "kg_mstca",
            "task": "online_binary",
            "in_channels": 31,
            "num_classes": 2,
            "label_names": ["normal", "fault"],
            "best_val_f1": 0.91,
        },
    )
    _write_json(
        model_dir / "metrics_test.json",
        {
            "accuracy": 0.95,
            "macro_f1": 0.95,
            "weighted_f1": 0.95,
            "f1_fault": 0.95,
        },
    )
    _write_json(
        model_dir / "norm_stats.json",
        {
            "columns": [f"feat_{i}" for i in range(31)],
            "mean": [0.0] * 31,
            "std": [1.0] * 31,
            "window_size": 150,
        },
    )

    models = list_registered_models(
        {"MODEL_ROOT": str(model_root), "MODEL_WHITELIST": ""},
        task="online_binary",
        capability="realtime_detection",
    )

    names = [item["name"] for item in models]
    assert "online_detector_kg_mstca_rfly" in names
    meta = next(item for item in models if item["name"] == "online_detector_kg_mstca_rfly")
    assert meta["architecture"] == "kg_mstca"
    assert meta["capabilities"]["realtime_detection"] is True
    assert meta["metrics"]["f1_fault"] == 0.95
