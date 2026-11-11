import sys
import types
from pathlib import Path

import bcrypt
import pytest

from uav_dashboard_api import create_app, db
from uav_dashboard_api.config import TestingConfig
from uav_dashboard_api.models.user import User
from uav_dashboard_api.utils.jwt_utils import JWTUtils


def _write_json(path: Path, payload: dict):
    import json

    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


def _create_model(root: Path, name: str, config: dict):
    model_dir = root / name
    model_dir.mkdir(parents=True, exist_ok=True)
    _write_json(model_dir / "config.json", config)
    (model_dir / "best.pt").write_bytes(b"checkpoint")


@pytest.fixture
def app(tmp_path: Path, monkeypatch):
    fake_fault_detector = types.ModuleType("uav_fault.online.fault_detector")
    fake_build_sequence_cache = types.ModuleType("uav_fault.data.build_sequence_cache")

    class _DummyFaultDetector:
        def __init__(self, model_dir=None, device="cpu"):
            self.model_dir = model_dir
            self.device = device
            self.window_size = 150

        def predict(self, window):
            return None

    fake_fault_detector.FaultDetector = _DummyFaultDetector
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
    monkeypatch.setitem(sys.modules, "uav_fault.online.fault_detector", fake_fault_detector)
    monkeypatch.setitem(sys.modules, "data.build_sequence_cache", fake_build_sequence_cache)
    monkeypatch.setitem(
        sys.modules, "uav_fault.data.build_sequence_cache", fake_build_sequence_cache
    )

    model_root = tmp_path / "models"
    runtime_dir = tmp_path / "runtime"
    model_root.mkdir()
    _create_model(
        model_root, "lstm", {"model": "cnn_lstm_v2_normalproto", "in_channels": 31, "num_classes": 3, "label_names": ["normal", "motor", "sensor"]}
    )

    class LocalTestingConfig(TestingConfig):
        SQLALCHEMY_DATABASE_URI = f"sqlite:///{tmp_path / 'dashboard_flows.db'}"
        MODEL_ROOT = str(model_root)
        MODEL_WHITELIST = ()
        DEFAULT_OFFLINE_MODEL = "lstm"
        DEFAULT_ONLINE_MODEL = "online_detector_alpha"
        ONLINE_MODEL_PRIORITY = ()
        RUNTIME_DIR = runtime_dir
        UPLOAD_DIR = runtime_dir / "uploads"
        DERIVED_DIR = runtime_dir / "derived"
        RUNS_DIR = runtime_dir / "runs"
        REPORTS_DIR = runtime_dir / "reports"
        ENABLE_BLIND_TEST = True
        TELEMETRY_ALLOWED_BACKENDS = ("replay", "px4", "rfly")

    flask_app = create_app(LocalTestingConfig)
    flask_app.config["TESTING"] = True
    with flask_app.app_context():
        db.create_all()
        yield flask_app
        from uav_dashboard_api.services.task_manager import shutdown_executor
        from uav_dashboard_api.services.telemetry_manager import shutdown_telemetry_manager

        shutdown_executor(flask_app, wait=True)
        shutdown_telemetry_manager(flask_app)
        db.session.remove()
        db.drop_all()


@pytest.fixture
def client(app):
    return app.test_client()


@pytest.fixture
def auth_header(app):
    with app.app_context():
        user = User(
            username="flow_user",
            password=bcrypt.hashpw(b"test123", bcrypt.gensalt()).decode("utf-8"),
            real_name="Flow User",
            email="flow@example.com",
            role="admin",
            status="active",
        )
        db.session.add(user)
        db.session.commit()
        return {"Authorization": f"Bearer {JWTUtils.generate_token(user.id)}"}
