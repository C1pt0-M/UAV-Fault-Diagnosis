from __future__ import annotations

import sys
import json
from pathlib import Path
from types import SimpleNamespace

import bcrypt
import pytest


BACKEND_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = BACKEND_ROOT.parents[2]
SRC_ROOT = PROJECT_ROOT / "packages" / "uav_fault"
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


def _manager(tmp_path: Path, config_overrides: dict | None = None):
    config = {
        "REPO_ROOT": str(PROJECT_ROOT),
        "TELEMETRY_PYTHON_BIN": sys.executable,
        "RFLY_FORWARD_HOST": "127.0.0.1",
        "RFLY_FORWARD_PORT": 15550,
        "RFLY_EMIT_RATE_HZ": 50,
        "RFLY_RPM_SCALE": 10000.0,
        "RFLY_TEST_ID": 1,
        "RFLY_UDP_PORT": 20100,
        "RFLY_UDP_MODE": 2,
        "RFLY_MAV_IP": "127.0.0.1",
        "RFLY_PYMAV_ADDRESS": "udpin:0.0.0.0:14030",
        "RFLY_ENABLE_PYMAV": True,
        "RFLY_RECEIVER_TIMEOUT_S": 60.0,
        "RFLY_PX4MAVCTRL_PATH": "",
        "RFLY_BRIDGE_SCRIPT": "",
    }
    config.update(config_overrides or {})
    return SimpleNamespace(
        app=SimpleNamespace(config=config),
        _receiver=None,
        _receiver_thread=None,
        _startup_phase=None,
        _timeout_reason=None,
        _on_backend_frame=lambda frame, window, source_backend=None: None,
        _receiver_worker=lambda: None,
        _popen_session_kwargs=lambda: {},
        _stop_process=lambda proc, name: None,
        _proc_status=lambda proc: "running" if proc is not None else None,
    )


def test_rfly_backend_uses_default_px4mavctrl_path_when_config_is_empty(tmp_path, monkeypatch):
    from uav_dashboard_api.services import telemetry_backends
    from uav_dashboard_api.services.telemetry_backends import RflyScenarioBackend

    default_px4mavctrl = tmp_path / "RflySimSDK" / "ctrl" / "PX4MavCtrlV4.py"
    default_px4mavctrl.parent.mkdir(parents=True)
    default_px4mavctrl.write_text("class PX4MavCtrler:\n    pass\n", encoding="utf-8")
    monkeypatch.setattr(
        telemetry_backends,
        "DEFAULT_RFLY_PX4MAVCTRL_PATH",
        default_px4mavctrl,
    )

    backend = RflyScenarioBackend(
        _manager(tmp_path),
        session_id="TS-RFLY-001",
        csv_path=str(tmp_path / "session.csv"),
        sim_time=60,
        options={},
    )

    backend.prepare()

    assert backend._px4mavctrl_path == default_px4mavctrl


def test_rfly_backend_starts_receiver_and_bridge_process(tmp_path, monkeypatch):
    from uav_dashboard_api.services import telemetry_backends
    from uav_dashboard_api.services.telemetry_backends import RflyScenarioBackend

    px4mavctrl = tmp_path / "PX4MavCtrlV4.py"
    px4mavctrl.write_text("class PX4MavCtrler:\n    pass\n", encoding="utf-8")
    manager = _manager(tmp_path, {"RFLY_PX4MAVCTRL_PATH": str(px4mavctrl)})

    receiver_calls = []

    class FakeReceiver:
        def __init__(self, **kwargs):
            receiver_calls.append(kwargs)
            self.callbacks = []
            self._sock = SimpleNamespace(close=lambda: None)

    thread_calls = []

    class FakeThread:
        def __init__(self, target=None, name=None, daemon=None, args=(), kwargs=None):
            self.target = target
            self.name = name
            self.daemon = daemon
            self.args = args
            self.kwargs = kwargs or {}
            self.started = False

        def start(self):
            self.started = True
            thread_calls.append(self.name)

        def is_alive(self):
            return self.started

        def join(self, timeout=None):
            self.started = False

    popen_calls = []

    class FakeProc:
        pid = 1234

    def fake_popen(cmd, stdout=None, stderr=None, cwd=None, env=None, **kwargs):
        popen_calls.append({"cmd": cmd, "cwd": cwd, "env": env})
        return FakeProc()

    monkeypatch.setattr(telemetry_backends, "TelemetryReceiver", FakeReceiver)
    monkeypatch.setattr(telemetry_backends.threading, "Thread", FakeThread)
    monkeypatch.setattr(telemetry_backends.subprocess, "Popen", fake_popen)

    backend = RflyScenarioBackend(
        manager,
        session_id="TS-RFLY-002",
        csv_path=str(tmp_path / "session.csv"),
        sim_time=60,
        options={},
    )

    backend.prepare()
    backend.start()

    assert receiver_calls[0]["port"] == 15550
    assert receiver_calls[0]["csv_path"].endswith("session.csv")
    assert receiver_calls[0]["auto_stop"] == 0
    assert thread_calls == ["telemetry-rfly-receiver"]
    cmd = popen_calls[0]["cmd"]
    assert cmd[:2] == [
        sys.executable,
        str(PROJECT_ROOT / "packages" / "rfly_bridge" / "rfly_bridge" / "run_rfly_live_bridge.py"),
    ]
    assert "--px4mavctrl" in cmd
    assert str(px4mavctrl) in cmd
    assert "--port" in cmd
    assert "15550" in cmd
    assert "--duration" in cmd
    assert "60.0" in cmd
    assert popen_calls[0]["cwd"] == str(PROJECT_ROOT)


@pytest.fixture
def rfly_api_app(tmp_path: Path):
    from uav_dashboard_api import create_app, db
    from uav_dashboard_api.models.user import User
    from uav_dashboard_api.config import TestingConfig

    class LocalConfig(TestingConfig):
        SQLALCHEMY_DATABASE_URI = f"sqlite:///{tmp_path / 'rfly_api.db'}"
        SECRET_KEY = "test-secret"
        JWT_SECRET_KEY = "test-jwt"
        TELEMETRY_ALLOWED_BACKENDS = ("rfly", "replay")
        ENABLE_TELEMETRY = True
        MODEL_ROOT = str(PROJECT_ROOT / "models")
        RUNTIME_DIR = tmp_path / "runtime"
        UPLOAD_DIR = RUNTIME_DIR / "uploads"
        DERIVED_DIR = RUNTIME_DIR / "derived"
        RUNS_DIR = RUNTIME_DIR / "runs"
        REPORTS_DIR = RUNTIME_DIR / "reports"
        AI_RUNTIME_DIR = RUNTIME_DIR / "ai"

    app = create_app(LocalConfig)
    app.config["TESTING"] = True
    with app.app_context():
        db.create_all()
        user = User(
            username="rfly_user",
            password=bcrypt.hashpw(b"test123", bcrypt.gensalt()).decode("utf-8"),
            real_name="Rfly User",
            email="rfly@example.com",
            role="admin",
            status="active",
        )
        db.session.add(user)
        db.session.commit()
        yield app
        db.session.remove()
        db.drop_all()


def test_start_collection_api_accepts_rfly_backend(rfly_api_app, monkeypatch, tmp_path):
    from uav_dashboard_api.services.telemetry_manager import telemetry_manager
    from uav_dashboard_api.utils.jwt_utils import JWTUtils

    px4mavctrl = tmp_path / "PX4MavCtrlV4.py"
    px4mavctrl.write_text("class PX4MavCtrler:\n    pass\n", encoding="utf-8")
    captured = {}

    def fake_start_collection(**kwargs):
        captured.update(kwargs)
        return {
            "session_id": "TS-RFLY-API",
            "status": "running",
            "scenario_backend": "rfly",
        }

    monkeypatch.setattr(telemetry_manager, "start_collection", fake_start_collection)
    token = JWTUtils.generate_token(1)
    client = rfly_api_app.test_client()

    response = client.post(
        "/api/telemetry/start-collection",
        headers={"Authorization": f"Bearer {token}"},
        json={
            "backend": "rfly",
            "sim_time": 60,
            "rfly_px4mavctrl_path": str(px4mavctrl),
        },
    )

    assert response.status_code == 200
    assert response.get_json()["data"]["scenario_backend"] == "rfly"
    assert captured["backend"] == "rfly"
    assert captured["rfly_options"]["px4mavctrl_path"] == str(px4mavctrl)


def test_rfly_diagnostics_reports_not_ready_when_backend_is_not_allowed(rfly_api_app):
    from uav_dashboard_api.utils.jwt_utils import JWTUtils

    rfly_api_app.config["TELEMETRY_ALLOWED_BACKENDS"] = ("replay",)
    token = JWTUtils.generate_token(1)
    client = rfly_api_app.test_client()

    response = client.get(
        "/api/telemetry/rfly/diagnostics",
        headers={"Authorization": f"Bearer {token}"},
    )

    assert response.status_code == 200
    data = response.get_json()["data"]
    assert data["ready"] is False
    assert data["summary"]["status"] == "not_ready"
    assert "backend_not_allowed" in {issue["code"] for issue in data["issues"]}


def test_rfly_diagnostics_accepts_override_and_masks_server_paths(rfly_api_app, tmp_path):
    from uav_dashboard_api.utils.jwt_utils import JWTUtils

    px4mavctrl = tmp_path / "PX4MavCtrlV4.py"
    px4mavctrl.write_text("class PX4MavCtrler:\n    pass\n", encoding="utf-8")
    token = JWTUtils.generate_token(1)
    client = rfly_api_app.test_client()

    response = client.post(
        "/api/telemetry/rfly/diagnostics",
        headers={"Authorization": f"Bearer {token}"},
        json={"rfly_px4mavctrl_path": str(px4mavctrl)},
    )

    assert response.status_code == 200
    data = response.get_json()["data"]
    assert data["ready"] is True
    assert data["summary"]["status"] == "ready"
    assert data["checks"]["px4mavctrl"]["ok"] is True
    assert data["checks"]["px4mavctrl"]["name"] == "PX4MavCtrlV4.py"
    assert str(px4mavctrl) not in json.dumps(data, ensure_ascii=False)


def test_rfly_diagnostics_uses_default_px4mavctrl_path_when_config_is_empty(
    rfly_api_app, tmp_path, monkeypatch
):
    from uav_dashboard_api.services import rfly_diagnostics
    from uav_dashboard_api.utils.jwt_utils import JWTUtils

    px4mavctrl = tmp_path / "RflySimSDK" / "ctrl" / "PX4MavCtrlV4.py"
    px4mavctrl.parent.mkdir(parents=True)
    px4mavctrl.write_text("class PX4MavCtrler:\n    pass\n", encoding="utf-8")
    monkeypatch.setattr(
        rfly_diagnostics,
        "DEFAULT_RFLY_PX4MAVCTRL_PATH",
        px4mavctrl,
    )
    rfly_api_app.config["RFLY_PX4MAVCTRL_PATH"] = ""
    token = JWTUtils.generate_token(1)
    client = rfly_api_app.test_client()

    response = client.get(
        "/api/telemetry/rfly/diagnostics",
        headers={"Authorization": f"Bearer {token}"},
    )

    assert response.status_code == 200
    data = response.get_json()["data"]
    assert data["checks"]["px4mavctrl"]["ok"] is True
    assert data["checks"]["px4mavctrl"]["name"] == "PX4MavCtrlV4.py"


def test_telemetry_status_exposes_rfly_diagnostics_summary(rfly_api_app, monkeypatch):
    from uav_dashboard_api.services.telemetry_manager import telemetry_manager
    from uav_dashboard_api.utils.jwt_utils import JWTUtils

    monkeypatch.setattr(telemetry_manager, "get_status", lambda: {"is_collecting": False})
    token = JWTUtils.generate_token(1)
    client = rfly_api_app.test_client()

    response = client.get(
        "/api/telemetry/status",
        headers={"Authorization": f"Bearer {token}"},
    )

    assert response.status_code == 200
    capabilities = response.get_json()["data"]["telemetry_capabilities"]
    assert "rfly_ready" in capabilities
    assert capabilities["rfly_diagnostics_summary"]["status"] in {"ready", "not_ready"}


def test_rfly_transport_status_is_connected_when_bridge_and_receiver_are_running():
    from uav_dashboard_api.services.telemetry_manager import TelemetryProcessManager

    mgr = TelemetryProcessManager()
    mgr._reset_runtime_state()
    mgr._backend_name = "rfly"

    class LiveThread:
        def is_alive(self):
            return True

    mgr._receiver_thread = LiveThread()
    mgr._backend = SimpleNamespace(
        status=lambda: {
            "rfly_bridge": "running",
            "receiver": "running",
            "collector": None,
            "px4_sitl": None,
        }
    )

    assert mgr.get_status()["transport_status"] == "connected"


def test_rfly_backend_frame_is_canonicalized_and_exposes_proxy_contract(monkeypatch):
    from uav_dashboard_api.services.telemetry_manager import TelemetryProcessManager

    mgr = TelemetryProcessManager()
    mgr._reset_runtime_state()
    mgr._backend_name = "rfly"
    mgr._realtime_buffer.clear()

    window = [
        {
            "seq": idx,
            "time_s": float(idx) / 50.0,
            "actuator_out_1": 1300.0,
            "actuator_out_2": 1310.0,
            "actuator_out_3": 1290.0,
            "actuator_out_4": 1305.0,
            "actuator_source": "proxy",
            "att_q_w": 1.0,
            "att_q_x": 0.0,
            "att_q_y": 0.0,
            "att_q_z": 0.0,
            "vel_x": 0.1,
            "vel_y": 0.2,
            "vel_z": -0.3,
            "accel_x": 0.0,
            "accel_y": 0.0,
            "accel_z": -9.8,
            "gyro_x": 0.0,
            "gyro_y": 0.0,
            "gyro_z": 0.0,
            "baro_alt": 12.0,
            "baro_temp": 25.0,
            "baro_pressure": 101325.0,
        }
        for idx in range(150)
    ]

    mgr._on_backend_frame(window[-1], window, source_backend="rfly")

    assert mgr._feature_coverage["mapped_count"] > 0
    assert mgr._input_contract_status["actuator_source"] == "proxy"
    assert mgr._input_contract_status["status"] in {"healthy", "degraded"}
    assert mgr._realtime_buffer[-1]["motor1"] == 1300.0
