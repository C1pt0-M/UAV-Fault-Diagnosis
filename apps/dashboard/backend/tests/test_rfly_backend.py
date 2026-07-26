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
        "RFLY_PYTHON_BIN": sys.executable,
        "RFLY_FORWARD_HOST": "127.0.0.1",
        "RFLY_FORWARD_PORT": 15550,
        "RFLY_INFERENCE_LOCATION": "center",
        "RFLY_MEC_LISTEN_HOST": "127.0.0.1",
        "RFLY_MEC_LISTEN_PORT": 15552,
        "RFLY_MEC_FORWARD_HOST": "127.0.0.1",
        "RFLY_MEC_FORWARD_PORT": 15550,
        "RFLY_MEC_SCRIPT": "",
        "RFLY_MEC_INFERENCE_STRIDE": 5,
        "RFLY_MEC_STARTUP_WAIT_S": 0.0,
        "RFLY_EMIT_RATE_HZ": 50,
        "RFLY_RPM_SCALE": 10000.0,
        "RFLY_TEST_ID": 1,
        "RFLY_UDP_PORT": 20100,
        "RFLY_UDP_MODE": 2,
        "RFLY_MAV_IP": "127.0.0.1",
        "RFLY_PYMAV_ADDRESS": "udpin:0.0.0.0:14031",
        "RFLY_ENABLE_PYMAV": True,
        "RFLY_RECEIVER_TIMEOUT_S": 60.0,
        "RFLY_PX4MAVCTRL_PATH": "",
        "RFLY_BRIDGE_SCRIPT": "",
        "RFLY_BRIDGE_MODE": "attach",
        "RFLY_AUTO_START_SIM": False,
        "RFLY_POWERSHELL_BIN": "pwsh",
        "RFLY_SIM_START_SCRIPT": "scripts/start_rflysim_sitl_qgc.ps1",
        "RFLY_SIM_STOP_SCRIPT": "scripts/stop_rflysim_sitl.ps1",
        "RFLY_SITL_FRAME": "iris",
        "RFLY_QGC_UDP_PORT": 14600,
        "RFLY_DASHBOARD_PYMAV_UDP_PORT": 14031,
        "RFLY_SIM_MAP": "LowGPU",
        "RFLY_BUAA_AUTOTEST_DIR": str(tmp_path / "AutoTestAPI"),
        "RFLY_BUAA_UNIQUE_INDEX": 1419000188,
        "RFLY_BUAA_STARTUP_WAIT_S": 25.0,
        "RFLY_BUAA_READY_TIMEOUT_S": 90.0,
    }
    config.update(config_overrides or {})
    return SimpleNamespace(
        app=SimpleNamespace(config=config),
        on_transport_lifecycle=lambda snapshot, session_id, backend: None,
        start_transport_processing=lambda: None,
        stop_transport_processing=lambda drain=True: None,
        submit_transport_frame=lambda frame, window, **kwargs: None,
        on_transport_finished=lambda session_id, backend: None,
        _resolve_online_model_meta=lambda requested_model=None: {
            "name": requested_model or "online_detector_kg_mstca_rfly",
            "path": str(PROJECT_ROOT / "assets" / "models" / "online_detector_kg_mstca_rfly"),
            "window_size": 150,
        },
    )


def test_rfly_backend_builds_lowgpu_sim_start_command(tmp_path):
    from uav_dashboard_api.services.telemetry_backends import RflyScenarioBackend

    manager = _manager(
        tmp_path,
        {
            "RFLY_AUTO_START_SIM": True,
        },
    )
    backend = RflyScenarioBackend(
        manager,
        session_id="TS-RFLY-SIM-CMD",
        csv_path=str(tmp_path / "session.csv"),
        sim_time=30,
        options={},
    )

    cmd = backend._build_sim_start_cmd()

    assert cmd[0] == "pwsh"
    assert cmd[cmd.index("-File") + 1].endswith("start_rflysim_sitl_qgc.ps1")
    assert cmd[cmd.index("-Map") + 1] == "LowGPU"
    assert cmd[cmd.index("-QgcUdpPort") + 1] == "14600"
    assert cmd[cmd.index("-DashboardPymavUdpPort") + 1] == "14031"


def test_rfly_backend_mec_mode_builds_edge_port_chain(tmp_path):
    from uav_dashboard_api.services.telemetry_backends import RflyScenarioBackend

    px4mavctrl = tmp_path / "PX4MavCtrlV4.py"
    px4mavctrl.write_text("class PX4MavCtrler:\n    pass\n", encoding="utf-8")
    manager = _manager(
        tmp_path,
        {
            "RFLY_PX4MAVCTRL_PATH": str(px4mavctrl),
            "RFLY_INFERENCE_LOCATION": "mec",
            "RUNTIME_DIR": str(tmp_path / "runtime"),
        },
    )
    backend = RflyScenarioBackend(
        manager,
        session_id="TS-MEC-PORTS",
        csv_path=str(tmp_path / "session.csv"),
        sim_time=30,
        options={"inference_location": "mec"},
    )

    backend.prepare()
    backend._open_mec_log()
    mec_cmd = backend._build_mec_cmd()
    bridge_cmd = backend._build_bridge_cmd()

    assert mec_cmd[mec_cmd.index("--listen-port") + 1] == "15552"
    assert mec_cmd[mec_cmd.index("--forward-port") + 1] == "15550"
    assert bridge_cmd[bridge_cmd.index("--port") + 1] == "15552"


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

        def stop(self):
            self._sock.close()

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

        def poll(self):
            return None

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
    assert cmd[cmd.index("--duration") + 1] == "0.0"
    assert popen_calls[0]["cwd"] == str(PROJECT_ROOT)


def test_rfly_backend_prefers_dedicated_rfly_python_bin(tmp_path):
    from uav_dashboard_api.services.telemetry_backends import RflyScenarioBackend

    px4mavctrl = tmp_path / "PX4MavCtrlV4.py"
    px4mavctrl.write_text("class PX4MavCtrler:\n    pass\n", encoding="utf-8")
    manager = _manager(
        tmp_path,
        {
            "RFLY_PX4MAVCTRL_PATH": str(px4mavctrl),
            "TELEMETRY_PYTHON_BIN": "telemetry-python.exe",
            "RFLY_PYTHON_BIN": "rfly-python.exe",
        },
    )
    backend = RflyScenarioBackend(
        manager,
        session_id="TS-RFLY-PY",
        csv_path=str(tmp_path / "session.csv"),
        sim_time=30,
        options={},
    )

    backend.prepare()

    assert backend._build_bridge_cmd()[0] == "rfly-python.exe"


def test_rfly_backend_buaa_mode_builds_autotest_bridge_cmd(tmp_path):
    from uav_dashboard_api.services.telemetry_backends import RflyScenarioBackend

    autotest_dir = tmp_path / "AutoTestAPI"
    autotest_dir.mkdir()
    manager = _manager(
        tmp_path,
        {
            "RFLY_BRIDGE_MODE": "buaa",
            "RFLY_BUAA_AUTOTEST_DIR": str(autotest_dir),
            "RFLY_BUAA_UNIQUE_INDEX": 1419000188,
            "RFLY_PX4MAVCTRL_PATH": "",
        },
    )
    backend = RflyScenarioBackend(
        manager,
        session_id="TS-RFLY-BUAA",
        csv_path=str(tmp_path / "session.csv"),
        sim_time=30,
        options={},
    )

    backend.prepare()
    cmd = backend._build_bridge_cmd()

    assert cmd[1].endswith("run_buaa_realtime_bridge.py")
    assert "--autotest-dir" in cmd
    assert str(autotest_dir) in cmd
    assert "--unique-index" in cmd
    assert "1419000188" in cmd
    assert "--startup-wait" in cmd
    assert cmd[cmd.index("--ready-timeout") + 1] == "90.0"
    assert "--manual-control" in cmd
    assert "25.0" in cmd
    assert "--sim-map" in cmd
    assert "LowGPU" in cmd
    assert "--duration" in cmd
    assert cmd[cmd.index("--duration") + 1] == "0.0"
    assert backend._receiver_timeout_s() == 0.0
    assert "--px4mavctrl" not in cmd


def test_rfly_automatic_mode_keeps_planned_duration_and_receiver_timeout(tmp_path):
    from uav_dashboard_api.services.telemetry_backends import RflyScenarioBackend

    autotest_dir = tmp_path / "AutoTestAPI"
    autotest_dir.mkdir()
    manager = _manager(
        tmp_path,
        {
            "RFLY_BRIDGE_MODE": "buaa",
            "RFLY_BUAA_AUTO_CONTROL": True,
            "RFLY_BUAA_AUTOTEST_DIR": str(autotest_dir),
            "RFLY_RECEIVER_TIMEOUT_S": 75.0,
        },
    )
    backend = RflyScenarioBackend(
        manager,
        session_id="TS-RFLY-AUTO",
        csv_path=str(tmp_path / "session.csv"),
        sim_time=30,
        options={},
    )

    backend.prepare()
    cmd = backend._build_bridge_cmd()

    assert "--manual-control" not in cmd
    assert cmd[cmd.index("--duration") + 1] == "30.0"
    assert backend._receiver_timeout_s() == 75.0


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
            "rfly_inference_location": "mec",
        },
    )

    assert response.status_code == 200
    assert response.get_json()["data"]["scenario_backend"] == "rfly"
    assert captured["backend"] == "rfly"
    assert captured["rfly_options"]["px4mavctrl_path"] == str(px4mavctrl)
    assert captured["rfly_options"] == {
        "px4mavctrl_path": str(px4mavctrl),
        "inference_location": "mec",
        "inference_model": None,
    }
    assert captured["rfly_options"]["inference_location"] == "mec"


def test_start_collection_api_rejects_invalid_rfly_inference_location(
    rfly_api_app, tmp_path
):
    from uav_dashboard_api.utils.jwt_utils import JWTUtils

    px4mavctrl = tmp_path / "PX4MavCtrlV4.py"
    px4mavctrl.write_text("class PX4MavCtrler:\n    pass\n", encoding="utf-8")
    token = JWTUtils.generate_token(1)
    client = rfly_api_app.test_client()

    response = client.post(
        "/api/telemetry/start-collection",
        headers={"Authorization": f"Bearer {token}"},
        json={
            "backend": "rfly",
            "sim_time": 60,
            "rfly_px4mavctrl_path": str(px4mavctrl),
            "rfly_inference_location": "cloud",
        },
    )

    assert response.status_code == 400
    assert "rfly_inference_location" in response.get_json()["message"]


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


def test_rfly_diagnostics_buaa_mode_checks_autotest_dir_not_sdk_px4mavctrl(
    rfly_api_app, tmp_path
):
    from uav_dashboard_api.utils.jwt_utils import JWTUtils

    autotest_dir = tmp_path / "AutoTestAPI"
    autotest_dir.mkdir()
    rfly_api_app.config["RFLY_BRIDGE_MODE"] = "buaa"
    rfly_api_app.config["RFLY_BUAA_AUTOTEST_DIR"] = str(autotest_dir)
    rfly_api_app.config["RFLY_PX4MAVCTRL_PATH"] = ""
    token = JWTUtils.generate_token(1)
    client = rfly_api_app.test_client()

    response = client.get(
        "/api/telemetry/rfly/diagnostics",
        headers={"Authorization": f"Bearer {token}"},
    )

    assert response.status_code == 200
    data = response.get_json()["data"]
    assert data["checks"]["bridge_mode"]["mode"] == "buaa"
    assert data["checks"]["sim_map"]["map"] == "LowGPU"
    assert data["checks"]["px4mavctrl"]["ok"] is True
    assert data["checks"]["buaa_autotest"]["ok"] is True


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

    mgr._backend = SimpleNamespace(
        is_collecting=lambda: True,
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

    mgr.process_transport_frame(window[-1], window, source_backend="rfly")

    assert mgr._feature_coverage["mapped_count"] > 0
    assert mgr._input_contract_status["actuator_source"] == "proxy"
    assert mgr._input_contract_status["status"] in {"healthy", "degraded"}
    assert mgr._realtime_buffer[-1]["motor1"] == 1300.0


def test_rfly_backend_blocks_detection_when_critical_signals_are_zero(monkeypatch):
    from uav_dashboard_api.services.telemetry_manager import TelemetryProcessManager

    mgr = TelemetryProcessManager()
    mgr._reset_runtime_state()
    mgr._backend_name = "rfly"
    mgr._detecting = True

    calls = []

    class FakeDetector:
        window_size = 150

        def update(self, *, frame, window, frame_idx):
            calls.append((frame, window, frame_idx))
            return {"fault_prob": 1.0, "label": "fault", "confidence": 1.0}, []

    mgr._detector = FakeDetector()
    window = [
        {
            "seq": idx,
            "time_s": float(idx) / 50.0,
            "actuator_out_1": 0.0,
            "actuator_out_2": 0.0,
            "actuator_out_3": 0.0,
            "actuator_out_4": 0.0,
            "att_q_w": 1.0,
            "att_q_x": 0.0,
            "att_q_y": 0.0,
            "att_q_z": 0.0,
            "vel_x": 0.0,
            "vel_y": 0.0,
            "vel_z": 0.0,
            "accel_x": 0.0,
            "accel_y": 0.0,
            "accel_z": 0.0,
            "gyro_x": 0.0,
            "gyro_y": 0.0,
            "gyro_z": 0.0,
            "baro_alt": 0.0,
            "baro_temp": 25.0,
            "baro_pressure": 101325.0,
        }
        for idx in range(150)
    ]

    mgr.process_transport_frame(window[-1], window, source_backend="rfly")

    assert calls == []
    assert mgr._input_contract_status["status"] == "degraded"
    assert mgr._input_contract_status["reason"] == "untrusted_zero_signals"
    assert mgr._input_contract_status["detection_blocked"] is True
    assert "actuator_outputs" in mgr._input_contract_status["zero_signal_groups"]
    assert "imu_accel_z" in mgr._input_contract_status["zero_signal_groups"]


def test_rfly_backend_allows_detection_for_proxy_outputs_with_valid_imu(monkeypatch):
    from uav_dashboard_api.services.telemetry_manager import TelemetryProcessManager

    mgr = TelemetryProcessManager()
    mgr._reset_runtime_state()
    mgr._realtime_buffer.clear()
    monkeypatch.setattr(mgr, "_get_online_inference_stride", lambda: 2)
    mgr._backend_name = "rfly"
    mgr._detecting = True

    calls = []

    class FakeDetector:
        window_size = 150

        def update(self, *, frame, window, frame_idx):
            calls.append((frame, window, frame_idx))
            return {"fault_prob": 0.2, "label": "normal", "confidence": 0.8}, []

    mgr._detector = FakeDetector()
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
            "vel_y": 0.0,
            "vel_z": -0.2,
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

    mgr.process_transport_frame(window[-1], window, source_backend="rfly")
    assert calls == []
    mgr.process_transport_frame(window[-1], window, source_backend="rfly")

    assert len(calls) == 1
    assert len(mgr._realtime_buffer) == 2
    assert mgr._input_contract_status["status"] == "degraded"
    assert mgr._input_contract_status["actuator_source"] == "proxy"
    assert mgr._input_contract_status["detection_blocked"] is False
    assert mgr._current_fault_prob == 0.2
