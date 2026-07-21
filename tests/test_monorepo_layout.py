from __future__ import annotations

import importlib
import json
from pathlib import Path


def test_reorganized_packages_are_importable() -> None:
    assert importlib.import_module("uav_dashboard_api")
    assert importlib.import_module("uav_fault")
    assert importlib.import_module("rfly_bridge")
    assert importlib.util.find_spec("uav_fault.telemetry.schema_adapter")
    assert importlib.util.find_spec("rfly_bridge.rfly_live_bridge")


def test_runtime_model_manifest_lives_under_assets() -> None:
    manifest = Path(__file__).resolve().parents[1] / "assets" / "models" / "manifest.json"
    assert manifest.is_file()

    payload = json.loads(manifest.read_text(encoding="utf-8"))
    manifest_dirs = {
        str(entry["path"]).strip()
        for entry in payload.get("models", [])
        if isinstance(entry, dict) and str(entry.get("path", "")).strip()
    }
    actual_dirs = {
        item.name
        for item in manifest.parent.iterdir()
        if item.is_dir()
    }

    assert manifest_dirs
    assert manifest_dirs <= actual_dirs
    assert actual_dirs <= manifest_dirs


def test_scripts_are_layered_without_compatibility_shims() -> None:
    scripts_root = Path(__file__).resolve().parents[1] / "scripts"
    expected_dirs = {"data", "dev", "eval", "plots"}

    assert expected_dirs <= {item.name for item in scripts_root.iterdir() if item.is_dir()}

    moved_scripts = {
        "compare_offline_results.py": "eval",
        "eval_telemetry_live.py": "eval",
        "phase_a_input_gate.py": "data",
        "plot_misclassified_thrust_loss.py": "plots",
    }
    for script_name, bucket in moved_scripts.items():
        target = scripts_root / bucket / script_name
        assert target.is_file()
        assert not (scripts_root / script_name).exists()
