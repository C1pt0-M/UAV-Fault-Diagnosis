import importlib
import importlib.util
import sys
import types
from pathlib import Path

import pytest
from flask import Flask


BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

if "flask_cors" not in sys.modules:
    fake_flask_cors = types.ModuleType("flask_cors")
    fake_flask_cors.CORS = lambda *args, **kwargs: None
    sys.modules["flask_cors"] = fake_flask_cors


def _reload_config(monkeypatch, **env):
    keys = {
        "REPO_ROOT",
        "RUNTIME_DIR",
        "UPLOAD_DIR",
        "DERIVED_DIR",
        "RUNS_DIR",
        "REPORTS_DIR",
        "AI_RUNTIME_DIR",
        "AI_KNOWLEDGE_DIR",
        "MODEL_ROOT",
        "OFFLINE_NORM_STATS",
        "ML_PYTHON_BIN",
        "TELEMETRY_PYTHON_BIN",
        "FAULT_DATASET_ROOT",
        "ULG_PREPARE_SCRIPT",
        "PX4_ROOT",
    }
    for key in keys:
        monkeypatch.delenv(key, raising=False)
    for key, value in env.items():
        monkeypatch.setenv(key, value)

    sys.modules.pop("config", None)
    import uav_dashboard_api.config as config_module

    return importlib.reload(config_module)


def _load_bridge_module():
    module_name = "_test_bridge_module"
    sys.modules.pop(module_name, None)
    module_path = BACKEND_ROOT / "uav_dashboard_api" / "services" / "bridge.py"
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_config_blank_env_uses_repo_relative_defaults(monkeypatch, tmp_path: Path):
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    runtime_dir = tmp_path / "runtime"

    config_module = _reload_config(
        monkeypatch,
        REPO_ROOT=str(repo_root),
        RUNTIME_DIR=str(runtime_dir),
        AI_RUNTIME_DIR=str(runtime_dir / "ai"),
        MODEL_ROOT="",
        OFFLINE_NORM_STATS="",
        AI_KNOWLEDGE_DIR="",
        ML_PYTHON_BIN="",
        TELEMETRY_PYTHON_BIN="",
        FAULT_DATASET_ROOT="",
        ULG_PREPARE_SCRIPT="",
    )

    assert config_module.Config.MODEL_ROOT == str((repo_root / "assets" / "models").resolve())
    assert config_module.Config.OFFLINE_NORM_STATS == str(
        (repo_root / "assets" / "models" / "transfer3_cnnlstm_normalproto" / "norm_stats.json").resolve()
    )
    assert config_module.Config.AI_KNOWLEDGE_DIR == (
        repo_root / "knowledge_base"
    ).resolve()
    assert config_module.Config.ML_PYTHON_BIN == sys.executable
    assert config_module.Config.TELEMETRY_PYTHON_BIN == sys.executable
    assert config_module.Config.FAULT_DATASET_ROOT == ""
    assert config_module.Config.ULG_PREPARE_SCRIPT == ""


def test_run_ulg_to_csv_prefers_explicit_prepare_script(monkeypatch, tmp_path: Path):
    bridge = _load_bridge_module()

    script_path = tmp_path / "tools" / "build_fault_dataset.py"
    script_path.parent.mkdir(parents=True, exist_ok=True)
    script_path.write_text("# test stub\n", encoding="utf-8")

    calls = {}

    def fake_run(cmd, capture_output, text, cwd, check):
        calls["cmd"] = cmd
        calls["cwd"] = cwd
        return types.SimpleNamespace(returncode=0, stdout="ok", stderr="")

    monkeypatch.setattr(bridge.subprocess, "run", fake_run)

    app = Flask(__name__)
    app.config.update(
        ML_PYTHON_BIN="python-bin",
        ULG_PREPARE_SCRIPT=str(script_path),
        FAULT_DATASET_ROOT="",
        REPO_ROOT=str(tmp_path),
    )

    with app.app_context():
        result = bridge.run_ulg_to_csv("mission.ulg", "mission.csv")

    assert calls["cmd"] == [
        "python-bin",
        str(script_path),
        "--ulg_path",
        "mission.ulg",
        "--output_csv",
        "mission.csv",
    ]
    assert calls["cwd"] == str(tmp_path)
    assert result["output_csv"] == "mission.csv"


def test_run_ulg_to_csv_requires_prepare_script_configuration(tmp_path: Path):
    bridge = _load_bridge_module()

    app = Flask(__name__)
    app.config.update(
        ML_PYTHON_BIN="python-bin",
        ULG_PREPARE_SCRIPT="",
        FAULT_DATASET_ROOT="",
        REPO_ROOT=str(tmp_path),
    )

    with app.app_context():
        with pytest.raises(FileNotFoundError) as exc_info:
            bridge.run_ulg_to_csv("mission.ulg", "mission.csv")

    assert "ULG_PREPARE_SCRIPT" in str(exc_info.value)
 
 
 
def test_config_blank_env_exposes_px4_runtime_defaults(monkeypatch, tmp_path: Path): 
    for key in ( 
        "PX4_LAUNCH_MODE", 
        "PX4_READY_HOST", 
        "PX4_READY_PORT", 
        "TELEMETRY_MAVSDK_ADDRESS", 
        "TELEMETRY_MAVSDK_GRPC_PORT", 
        "TELEMETRY_PYMAV_ADDRESS", 
        "TELEMETRY_FORWARD_HOST", 
        "TELEMETRY_FORWARD_PORT", 
        "TELEMETRY_EMIT_RATE_HZ", 
        "TELEMETRY_ENABLE_PYMAV", 
        "TELEMETRY_START_FLIGHT_CONTROLLER", 
    ): 
        monkeypatch.delenv(key, raising=False) 
 
    repo_root = tmp_path / "repo" 
    repo_root.mkdir() 
    runtime_dir = tmp_path / "runtime" 
 
    config_module = _reload_config( 
        monkeypatch, 
        REPO_ROOT=str(repo_root), 
        RUNTIME_DIR=str(runtime_dir), 
        AI_RUNTIME_DIR=str(runtime_dir / "ai"), 
    ) 
 
    assert config_module.Config.PX4_LAUNCH_MODE == "managed" 
    assert config_module.Config.PX4_READY_HOST == "127.0.0.1" 
    assert config_module.Config.PX4_READY_PORT == 18570 
    assert config_module.Config.TELEMETRY_MAVSDK_ADDRESS == "udpin://0.0.0.0:14550" 
    assert config_module.Config.TELEMETRY_MAVSDK_GRPC_PORT == 50052 
    assert config_module.Config.TELEMETRY_PYMAV_ADDRESS == "udpin:0.0.0.0:14030" 
    assert config_module.Config.TELEMETRY_FORWARD_HOST == "127.0.0.1" 
    assert config_module.Config.TELEMETRY_FORWARD_PORT == 15550 
    assert config_module.Config.TELEMETRY_EMIT_RATE_HZ == 50 
    assert config_module.Config.TELEMETRY_ENABLE_PYMAV is True 
    assert config_module.Config.TELEMETRY_START_FLIGHT_CONTROLLER is True 
