import importlib
import os
import sys
import types
from pathlib import Path


BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

os.environ.setdefault("RUNTIME_DIR", str((BACKEND_ROOT / "runtime").resolve()))
os.environ.setdefault(
    "UPLOAD_DIR", str((BACKEND_ROOT / "runtime" / "uploads").resolve())
)
os.environ.setdefault(
    "DERIVED_DIR", str((BACKEND_ROOT / "runtime" / "derived").resolve())
)
os.environ.setdefault("RUNS_DIR", str((BACKEND_ROOT / "runtime" / "runs").resolve()))
os.environ.setdefault(
    "REPORTS_DIR", str((BACKEND_ROOT / "runtime" / "reports").resolve())
)
os.environ.setdefault(
    "AI_RUNTIME_DIR", str((BACKEND_ROOT / "runtime" / "ai").resolve())
)
REPO_ROOT = BACKEND_ROOT.parents[2]

os.environ.setdefault(
    "AI_KNOWLEDGE_DIR",
    str((REPO_ROOT / "knowledge_base").resolve()),
)
os.environ.setdefault("MODEL_ROOT", str((REPO_ROOT / "assets" / "models").resolve()))
os.environ.setdefault(
    "OFFLINE_NORM_STATS",
    str(
        (
            REPO_ROOT
            / "assets"
            / "models"
            / "transfer3_cnnlstm_normalproto"
            / "norm_stats.json"
        ).resolve()
    ),
)
os.environ.setdefault("ML_PYTHON_BIN", sys.executable)
os.environ.setdefault("TELEMETRY_PYTHON_BIN", sys.executable)
os.environ.setdefault("FAULT_DATASET_ROOT", "")
os.environ.setdefault("ULG_PREPARE_SCRIPT", "")

if "flask_cors" not in sys.modules:
    fake_flask_cors = types.ModuleType("flask_cors")
    fake_flask_cors.CORS = lambda *args, **kwargs: None
    sys.modules["flask_cors"] = fake_flask_cors

if "flask_sqlalchemy" not in sys.modules:
    fake_flask_sqlalchemy = types.ModuleType("flask_sqlalchemy")

    class _FakeSQLAlchemy:
        def __init__(self, *args, **kwargs):
            pass

        def init_app(self, app):
            return None

        def create_all(self):
            return None

    fake_flask_sqlalchemy.SQLAlchemy = _FakeSQLAlchemy
    sys.modules["flask_sqlalchemy"] = fake_flask_sqlalchemy

if "sqlalchemy" not in sys.modules:
    fake_sqlalchemy = types.ModuleType("sqlalchemy")
    fake_event = types.ModuleType("sqlalchemy.event")

    def _listens_for(*args, **kwargs):
        def _decorator(fn):
            return fn

        return _decorator

    fake_event.listens_for = _listens_for
    fake_sqlalchemy.event = fake_event
    sys.modules["sqlalchemy"] = fake_sqlalchemy
    sys.modules["sqlalchemy.event"] = fake_event

if "sqlalchemy.engine" not in sys.modules:
    fake_engine = types.ModuleType("sqlalchemy.engine")
    fake_engine.Engine = type("Engine", (), {})
    sys.modules["sqlalchemy.engine"] = fake_engine

if "uav_fault.data.build_sequence_cache" not in sys.modules:
    fake_build_sequence_cache = types.ModuleType("uav_fault.data.build_sequence_cache")
    fake_build_sequence_cache.SEQ_COLUMN_ALIASES = [
        ("time_s", ("trueTime", "Timestamp")),
        ("seq", ()),
        ("actuator_ctrl_1", ()),
        ("actuator_ctrl_2", ()),
        ("actuator_ctrl_3", ()),
        ("actuator_ctrl_4", ()),
        ("actuator_out_1", ()),
        ("actuator_out_2", ()),
        ("actuator_out_3", ()),
        ("actuator_out_4", ()),
        ("gyro_x", ()),
        ("gyro_y", ()),
        ("gyro_z", ()),
        ("accel_x", ()),
        ("accel_y", ()),
        ("accel_z", ()),
        ("baro_alt", ()),
        ("baro_temp", ()),
        ("baro_pressure", ()),
        ("vel_x", ()),
        ("vel_y", ()),
        ("vel_z", ()),
        ("mag_x", ()),
        ("mag_y", ()),
        ("mag_z", ()),
        ("att_q_w", ()),
        ("att_q_x", ()),
        ("att_q_y", ()),
        ("att_q_z", ()),
    ]
    fake_build_sequence_cache.SEQ_COLUMNS = [
        *[name for name, _ in fake_build_sequence_cache.SEQ_COLUMN_ALIASES],
        "actuator_mean",
        "actuator_std",
        "acc_norm",
        "actuator_effort",
    ]
    sys.modules["data.build_sequence_cache"] = fake_build_sequence_cache
    sys.modules["uav_fault.data.build_sequence_cache"] = fake_build_sequence_cache


def _pop_modules(*module_names: str) -> None:
    for module_name in module_names:
        sys.modules.pop(module_name, None)


def test_telemetry_manager_module_import_does_not_require_detection_runner(
    monkeypatch,
):
    trap_detection_runner = types.ModuleType("uav_dashboard_api.services.detection_runner")

    def _boom(name):
        raise AssertionError(
            f"telemetry_manager import should not access detection_runner.{name}"
        )

    trap_detection_runner.__getattr__ = _boom

    _pop_modules(
        "uav_dashboard_api.services.telemetry_manager",
        "uav_dashboard_api.services.detection_runner",
    )
    monkeypatch.setitem(
        sys.modules, "uav_dashboard_api.services.detection_runner", trap_detection_runner
    )

    module = importlib.import_module("uav_dashboard_api.services.telemetry_manager")

    assert hasattr(module, "TelemetryProcessManager")


def test_detection_runner_module_import_does_not_require_online_fault_detector(
    monkeypatch,
):
    trap_fault_detector = types.ModuleType("uav_fault.online.fault_detector")

    def _boom(name):
        raise AssertionError(
            f"detection_runner import should not access online.fault_detector.{name}"
        )

    trap_fault_detector.__getattr__ = _boom

    _pop_modules(
        "uav_dashboard_api.services.detection_runner",
        "uav_fault.online.fault_detector",
    )
    monkeypatch.setitem(sys.modules, "uav_fault.online.fault_detector", trap_fault_detector)

    module = importlib.import_module("uav_dashboard_api.services.detection_runner")

    assert hasattr(module, "DetectionRunner")


def test_detection_runner_instantiates_fault_detector_only_when_runner_is_created(
    monkeypatch,
):
    calls = []
    fake_fault_detector = types.ModuleType("uav_fault.online.fault_detector")

    class _FakeFaultDetector:
        def __init__(self, model_dir=None, device="cpu"):
            calls.append((model_dir, device))
            self.window_size = 123

        def predict(self, window):
            return None

    fake_fault_detector.FaultDetector = _FakeFaultDetector

    _pop_modules(
        "uav_dashboard_api.services.detection_runner",
        "uav_fault.online.fault_detector",
    )
    monkeypatch.setitem(sys.modules, "uav_fault.online.fault_detector", fake_fault_detector)

    module = importlib.import_module("uav_dashboard_api.services.detection_runner")
    runner = module.DetectionRunner("model-dir", device="cpu")

    assert calls == [("model-dir", "cpu")]
    assert runner.window_size == 123
