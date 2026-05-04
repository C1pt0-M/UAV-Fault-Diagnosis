import os
from pathlib import Path
import sys
import types

import pytest


BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

try:
    import flask_cors  # noqa: F401
except ImportError:
    fake_flask_cors = types.ModuleType("flask_cors")
    fake_flask_cors.CORS = lambda *args, **kwargs: None
    sys.modules["flask_cors"] = fake_flask_cors

try:
    import flask_sqlalchemy  # noqa: F401
except ImportError:
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

try:
    import sqlalchemy  # noqa: F401
except ImportError:
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

try:
    import sqlalchemy.engine  # noqa: F401
except ImportError:
    fake_engine = types.ModuleType("sqlalchemy.engine")
    fake_engine.Engine = type("Engine", (), {})
    sys.modules["sqlalchemy.engine"] = fake_engine

if "app.services.detection_runner" not in sys.modules:
    fake_detection_runner = types.ModuleType("app.services.detection_runner")

    class _FakeDetectionRunner:
        def __init__(self, model_dir=None, device="cpu"):
            self.model_dir = model_dir
            self.device = device
            self.window_size = 150

    fake_detection_runner.DetectionRunner = _FakeDetectionRunner
    sys.modules["app.services.detection_runner"] = fake_detection_runner

if "data.build_sequence_cache" not in sys.modules:
    fake_build_sequence_cache = types.ModuleType("data.build_sequence_cache")
    fake_build_sequence_cache.SEQ_COLUMN_ALIASES = [
        ("time_s", ("trueTime", "Timestamp")),
        ("seq", ()),
        ("actuator_ctrl_1", ()),
        ("actuator_ctrl_2", ()),
        ("actuator_ctrl_3", ()),
        ("actuator_ctrl_4", ()),
        ("actuator_out_1", ("_actuator_outputs_0_output[0]",)),
        ("actuator_out_2", ("_actuator_outputs_0_output[1]",)),
        ("actuator_out_3", ("_actuator_outputs_0_output[2]",)),
        ("actuator_out_4", ("_actuator_outputs_0_output[3]",)),
        ("gyro_x", ()),
        ("gyro_y", ()),
        ("gyro_z", ()),
        ("accel_x", ()),
        ("accel_y", ()),
        ("accel_z", ()),
        ("baro_alt", ("_vehicle_air_data_0_baro_alt_meter",)),
        ("baro_temp", ()),
        ("baro_pressure", ()),
        ("vel_x", ()),
        ("vel_y", ()),
        ("vel_z", ()),
        ("mag_x", ()),
        ("mag_y", ()),
        ("mag_z", ()),
        ("att_q_w", ("_vehicle_attitude_0_q[0]",)),
        ("att_q_x", ("_vehicle_attitude_0_q[1]",)),
        ("att_q_y", ("_vehicle_attitude_0_q[2]",)),
        ("att_q_z", ("_vehicle_attitude_0_q[3]",)),
    ]
    fake_build_sequence_cache.SEQ_COLUMNS = [
        *[name for name, _ in fake_build_sequence_cache.SEQ_COLUMN_ALIASES],
        "actuator_mean",
        "actuator_std",
        "acc_norm",
        "actuator_effort",
    ]
    sys.modules["data.build_sequence_cache"] = fake_build_sequence_cache

from uav_dashboard_api.services.telemetry_manager import TelemetryProcessManager


@pytest.fixture(autouse=True)
def isolated_runtime_env(tmp_path, monkeypatch):
    runtime_dir = (tmp_path / "runtime").resolve()
    runtime_dir.mkdir(parents=True, exist_ok=True)
    env = {
        "RUNTIME_DIR": runtime_dir,
        "UPLOAD_DIR": runtime_dir / "uploads",
        "DERIVED_DIR": runtime_dir / "derived",
        "RUNS_DIR": runtime_dir / "runs",
        "REPORTS_DIR": runtime_dir / "reports",
        "AI_RUNTIME_DIR": runtime_dir / "ai",
        "AI_KNOWLEDGE_DIR": (BACKEND_ROOT.parent.parent / "knowledge_base").resolve(),
        "MODEL_ROOT": (BACKEND_ROOT.parent.parent / "models").resolve(),
        "OFFLINE_NORM_STATS": (
            BACKEND_ROOT.parent.parent
            / "data"
            / "processed"
            / "seq_cache"
            / "norm_stats.json"
        ).resolve(),
        "ML_PYTHON_BIN": sys.executable,
        "TELEMETRY_PYTHON_BIN": sys.executable,
        "FAULT_DATASET_ROOT": "",
        "ULG_PREPARE_SCRIPT": "",
    }
    for key, value in env.items():
        monkeypatch.setenv(key, str(value))
    manager = TelemetryProcessManager()
    manager._reset_runtime_state()
    manager._realtime_buffer.clear()
    manager._px4_proc = None
    manager._collector_proc = None
    manager._flight_proc = None
    manager._receiver = None
    manager._receiver_thread = None
    manager.app = None
    yield runtime_dir
    manager._reset_runtime_state()
    manager._realtime_buffer.clear()
    manager._px4_proc = None
    manager._collector_proc = None
    manager._flight_proc = None
    manager._receiver = None
    manager._receiver_thread = None
    manager.app = None


def test_test_runtime_dirs_are_isolated_from_real_dashboard_runtime(isolated_runtime_env):
    assert Path(os.environ["RUNTIME_DIR"]).resolve() == isolated_runtime_env
    assert Path(os.environ["RUNTIME_DIR"]).resolve() != (BACKEND_ROOT / "runtime").resolve()


def test_get_status_exposes_predictive_warning_fields():
    mgr = TelemetryProcessManager()
    mgr._reset_runtime_state()
    mgr._current_fault_prob = 0.62
    mgr._risk_score = 0.58
    mgr._risk_trend = "rising"
    mgr._warning_stage = "warning"
    mgr._warning_reason = "椋庨櫓杩炵画涓婂崌锛屽紓甯告鐜囧凡瓒呰繃棰勮闃堝€?
    mgr._preview_started_at = 10.0
    mgr._top_recommendations = [
        {"title": "缁撴潫褰撳墠鏋舵鍚庢墽琛岀绾胯瘖鏂?, "priority": "P1"}
    ]
    mgr._last_frame_time_s = 13.4

    status = mgr.get_status()

    assert status["current_fault_prob"] == 0.62
    assert status["risk_score"] == 0.58
    assert status["risk_trend"] == "rising"
    assert status["warning_stage"] == "warning"
    assert "寮傚父姒傜巼宸茶秴杩囬璀﹂槇鍊? in status["warning_reason"]
    assert status["preview_lead_time_s"] == 3.4
    assert status["top_recommendations"][0]["priority"] == "P1"


def test_predictive_warning_transitions_from_preview_to_confirmed():
    mgr = TelemetryProcessManager()
    mgr._reset_runtime_state()
    mgr._detecting = True

    class _FakeEvent:
        def __init__(self):
            self.start_t = 11.0
            self.end_t = 13.0
            self.max_fault_prob = 0.92
            self.mean_fault_prob = 0.86
            self.n_points = 6

    class _FakeDetector:
        window_size = 1
        aggregator = type("Agg", (), {"onset_th": 0.6})()

        def __init__(self):
            self.calls = 0

        def update(self, frame, window, frame_idx):
            self.calls += 1
            if self.calls == 1:
                return (
                    {
                        "label": "fault",
                        "confidence": 0.66,
                        "fault_prob": 0.66,
                    },
                    [],
                )
            return (
                {
                    "label": "fault",
                    "confidence": 0.92,
                    "fault_prob": 0.92,
                },
                [_FakeEvent()],
            )

    mgr._detector = _FakeDetector()
    mgr._record_fault_event = lambda event: mgr._fault_events.append(event)
    frame = {
        "time_s": 10.0,
        "seq": 1,
        "motor1": 0.2,
        "motor2": 0.2,
        "motor3": 0.2,
        "motor4": 0.2,
    }

    mgr._on_backend_frame(frame, [frame], source_backend="px4")
    status_preview = mgr.get_status()
    assert status_preview["warning_stage"] in {"watch", "warning"}
    assert status_preview["latest_diagnoses"][0]["source"] == "realtime_preview"
    assert (
        status_preview["latest_diagnoses"][0]["evidence_summary"]["source"] == "online"
    )
    assert status_preview["latest_diagnoses"][0]["evidence_summary"]["top_evidence"]

    frame2 = {
        "time_s": 13.0,
        "seq": 2,
        "motor1": 0.3,
        "motor2": 0.3,
        "motor3": 0.3,
        "motor4": 0.3,
    }
    mgr._on_backend_frame(frame2, [frame2], source_backend="px4")
    status_confirmed = mgr.get_status()
    assert status_confirmed["warning_stage"] == "critical"
    assert status_confirmed["fault_events_count"] == 1
    assert status_confirmed["latest_diagnoses"][0]["source"] == "realtime"
    assert status_confirmed["latest_diagnoses"][0]["lead_time_s"] == 3.0
    assert status_confirmed["preview_lead_time_s"] == 3.0
    assert (
        status_confirmed["latest_diagnoses"][0]["evidence_summary"]["source"]
        == "online"
    )
    assert status_confirmed["latest_diagnoses"][0]["evidence_summary"]["top_evidence"]


def test_get_status_exposes_last_frame_time_and_input_contract_status():
    mgr = TelemetryProcessManager()
    mgr._reset_runtime_state()
    mgr._session_id = "TS-TEST"
    mgr._backend_name = "px4"
    mgr._start_time = 100.0
    mgr._planned_duration_s = 120.0
    mgr._frame_count = 3
    mgr._feature_coverage = {"mapped_count": 10, "default_count": 2}
    mgr._last_frame_time_s = 12.34
    mgr._last_frame_wall_time = 1234567890.5
    mgr._input_contract_status = {
        "status": "degraded",
        "reason": "missing_features",
    }
    mgr._startup_phase = "collecting"
    mgr._timeout_reason = "collector_no_frames"

    status = mgr.get_status()

    assert status["last_frame_time_s"] == 12.34
    assert status["last_frame_wall_time"] == 1234567890.5
    assert status["input_contract_status"] == {
        "status": "degraded",
        "reason": "missing_features",
    }
    assert status["startup_phase"] == "collecting"
    assert status["timeout_reason"] == "collector_no_frames"


def test_realtime_data_uses_canonical_frame_values_after_px4_frame_callback():
    mgr = TelemetryProcessManager()
    mgr._reset_runtime_state()
    mgr._realtime_buffer.clear()
    mgr._backend_name = "px4"

    frame = {
        "time_s": 1.0,
        "seq": 1,
        "motor1": 0.11,
        "motor2": 0.22,
        "motor3": 0.33,
        "motor4": 0.44,
        "roll": 0.1,
        "pitch": 0.2,
        "yaw": 0.3,
        "baro_alt_meter": 12.5,
    }
    window = [frame]

    mgr._on_backend_frame(frame, window, source_backend="px4")
    data = mgr.get_realtime_data()

    assert data["times"] == [1.0]
    assert data["motor1"] == [0.11]
    assert data["motor4"] == [0.44]
    assert data["roll"] == [0.1]
    assert data["pitch"] == [0.2]
    assert data["yaw"] == [0.3]
    assert data["pos_z"] == [12.5]


def test_realtime_data_derives_attitude_from_replay_quaternion_frame():
    mgr = TelemetryProcessManager()
    mgr._reset_runtime_state()
    mgr._realtime_buffer.clear()
    mgr._backend_name = "replay"

    frame = {
        "time_s": 1.5,
        "_actuator_outputs_0_output[0]": 1200.0,
        "_actuator_outputs_0_output[1]": 1190.0,
        "_actuator_outputs_0_output[2]": 1210.0,
        "_actuator_outputs_0_output[3]": 1185.0,
        "_vehicle_attitude_0_q[0]": 0.9553364891,
        "_vehicle_attitude_0_q[1]": 0.0,
        "_vehicle_attitude_0_q[2]": 0.0,
        "_vehicle_attitude_0_q[3]": 0.2955202067,
        "_vehicle_air_data_0_baro_alt_meter": 15.2,
    }

    mgr._on_backend_frame(frame, [frame], source_backend="replay")
    data = mgr.get_realtime_data()

    assert data["times"] == [1.5]
    assert data["motor1"] == [1200.0]
    assert abs(data["roll"][0]) < 1e-3
    assert abs(data["pitch"][0]) < 1e-3
    assert abs(data["yaw"][0] - 0.6) < 1e-3
    assert data["pos_z"] == [15.2]


def test_receiver_worker_marks_timeout_reason_when_no_frames_arrive():
    mgr = TelemetryProcessManager()
    mgr._reset_runtime_state()
    mgr._session_id = "TS-RX"
    mgr._start_time = 100.0

    class _FakeReceiver:
        def start(self):
            raise TimeoutError("receiver timeout")

    mgr._receiver = _FakeReceiver()
    mgr._receiver_worker()

    status = mgr.get_status()
    assert status["timeout_reason"] == "receiver_timeout"
    assert status["startup_phase"] == "receiver_timeout"


def test_start_detection_marks_startup_phase_detecting():
    mgr = TelemetryProcessManager()
    mgr._reset_runtime_state()
    mgr._session_id = "TS-DET"
    mgr._start_time = 100.0
    mgr._planned_duration_s = 120.0
    mgr._backend_name = "px4"

    mgr._backend = object()
    mgr._receiver = object()

    class _LiveThread:
        def is_alive(self):
            return True

    mgr._receiver_thread = _LiveThread()

    mgr._resolve_online_model_meta = lambda requested_model=None: {
        "name": "online_detector_alpha",
        "path": "fake_model_dir",
        "task": "online_binary",
        "architecture": "cnn1d",
        "num_classes": 2,
        "label_names": ["normal", "fault"],
        "window_size": 150,
        "feature_count": 31,
        "metrics": {},
    }

    original_runner = globals().get("DetectionRunner")
    from uav_dashboard_api.services import telemetry_manager as telemetry_manager_module

    class _FakeRunner:
        def __init__(self, model_dir, device="cpu"):
            self.window_size = 150

    telemetry_manager_module.DetectionRunner = _FakeRunner
    try:
        mgr.start_detection()
        status = mgr.get_status()
        assert status["startup_phase"] == "detecting"
    finally:
        telemetry_manager_module.DetectionRunner = original_runner


def test_start_detection_wraps_detector_init_error_as_runtime_error():
    mgr = TelemetryProcessManager()
    mgr._reset_runtime_state()
    mgr._session_id = "TS-DET-ERR"
    mgr._start_time = 100.0
    mgr._planned_duration_s = 120.0
    mgr._backend_name = "replay"

    class _ReplayBackend:
        pass

    mgr._backend = _ReplayBackend()
    mgr._receiver = object()

    class _LiveThread:
        def is_alive(self):
            return True

    mgr._backend._thread = _LiveThread()
    mgr._resolve_online_model_meta = lambda requested_model=None: {
        "name": "online_detector_CNN",
        "path": "bad_model_dir",
        "task": "online_binary",
        "architecture": "cnn1d",
        "num_classes": 2,
        "label_names": ["normal", "fault"],
        "window_size": 150,
        "feature_count": 31,
        "metrics": {},
    }

    from uav_dashboard_api.services import telemetry_manager as telemetry_manager_module

    original_runner = telemetry_manager_module.DetectionRunner

    class _BoomRunner:
        def __init__(self, model_dir, device="cpu"):
            raise KeyError("hidden_size")

    telemetry_manager_module.DetectionRunner = _BoomRunner
    try:
        try:
            mgr.start_detection()
            assert False, "expected RuntimeError"
        except RuntimeError as exc:
            assert "鍦ㄧ嚎妫€娴嬪惎鍔ㄥけ璐? in str(exc)
            assert "online_detector_CNN" in str(exc)
            assert "hidden_size" in str(exc)
        assert mgr._detecting is False
        assert mgr._active_model is None
    finally:
        telemetry_manager_module.DetectionRunner = original_runner


def test_on_backend_frame_adds_preview_alert_when_fault_prob_reaches_onset_threshold_but_no_finalized_event():
    mgr = TelemetryProcessManager()
    mgr._reset_runtime_state()
    mgr._detecting = True

    class _FakeDetector:
        window_size = 1
        aggregator = type("Agg", (), {"onset_th": 0.6})()

        def update(self, frame, window, frame_idx):
            return (
                {
                    "label": "fault",
                    "confidence": 0.68,
                    "fault_prob": 0.68,
                    "raw_logits": [0.32, 0.68],
                },
                [],
            )

    mgr._detector = _FakeDetector()
    frame = {
        "time_s": 3.5,
        "seq": 1,
        "motor1": 0.2,
        "motor2": 0.2,
        "motor3": 0.2,
        "motor4": 0.2,
    }
    mgr._on_backend_frame(frame, [frame], source_backend="px4")

    status = mgr.get_status()
    assert status["fault_events_count"] == 1
    assert status["latest_diagnoses"][0]["label"] == "fault"
    assert status["latest_diagnoses"][0]["probability"] == 0.68


def test_start_px4_uses_configured_sim_target_and_autostart(monkeypatch):
    mgr = TelemetryProcessManager()
    mgr._reset_runtime_state()

    class _FakeApp:
        config = {
            "PX4_SIM_TARGET": "gz_x500",
            "PX4_SYS_AUTOSTART": "4005",
        }

    calls = []

    def _fake_run(*args, **kwargs):
        class _Result:
            returncode = 1

        return _Result()

    class _FakeProc:
        pid = 12345

    def _fake_popen(cmd, stdout=None, stderr=None, env=None, **kwargs):
        calls.append(
            {
                "cmd": cmd,
                "env": env,
            }
        )
        return _FakeProc()

    monkeypatch.setattr("app.services.telemetry_manager.subprocess.run", _fake_run)
    monkeypatch.setattr("app.services.telemetry_manager.subprocess.Popen", _fake_popen)

    mgr.app = _FakeApp()
    mgr._start_px4()

    assert len(calls) == 1
    assert calls[0]["cmd"] == [
        "bash",
        "-c",
        f"make -C {mgr._px4_root} px4_sitl gz_x500",
    ]
    assert calls[0]["env"]["PX4_SYS_AUTOSTART"] == "4005"
    assert calls[0]["env"]["HEADLESS"] == "1"


def test_resolve_telemetry_python_bin_prefers_runtime_python_over_legacy_linux_path(
    monkeypatch,
):
    mgr = TelemetryProcessManager()
    mgr._reset_runtime_state()
    mgr._telemetry_python_bin = None

    class _FakeApp:
        config = {
            "TELEMETRY_PYTHON_BIN": "",
            "ML_PYTHON_BIN": "",
        }

    checked = []
    legacy_linux = "/home/jack/miniconda3/envs/fault-dataset/bin/python"

    monkeypatch.setattr(
        "app.services.telemetry_manager.os.path.exists",
        lambda path: path in {legacy_linux, sys.executable},
    )
    monkeypatch.setattr(
        mgr,
        "_python_supports_telemetry",
        lambda python_bin: checked.append(python_bin) or True,
    )

    mgr.app = _FakeApp()
    resolved = mgr._resolve_telemetry_python_bin()

    assert resolved == sys.executable
    assert checked[0] == sys.executable


def test_start_px4_uses_configured_px4_root(monkeypatch):
    mgr = TelemetryProcessManager()
    mgr._reset_runtime_state()

    class _FakeApp:
        config = {
            "PX4_ROOT": "/tmp/custom-px4",
            "PX4_SIM_TARGET": "gz_x500",
            "PX4_SYS_AUTOSTART": "4005",
        }

    calls = []

    def _fake_run(*args, **kwargs):
        class _Result:
            returncode = 1

        return _Result()

    class _FakeProc:
        pid = 12345

    def _fake_popen(cmd, stdout=None, stderr=None, env=None, **kwargs):
        calls.append({"cmd": cmd, "env": env})
        return _FakeProc()

    monkeypatch.setattr("app.services.telemetry_manager.subprocess.run", _fake_run)
    monkeypatch.setattr("app.services.telemetry_manager.subprocess.Popen", _fake_popen)

    mgr.app = _FakeApp()
    mgr._start_px4()

    assert calls[0]["cmd"] == [
        "bash",
        "-c",
        "make -C /tmp/custom-px4 px4_sitl gz_x500",
    ]
 
 
 
def test_wait_for_px4_uses_configured_ready_endpoint(monkeypatch): 
    mgr = TelemetryProcessManager() 
    mgr._reset_runtime_state() 
 
    class _FakeApp: 
        config = {"PX4_READY_HOST": "10.8.0.9", "PX4_READY_PORT": 19001} 
 
    calls = [] 
 
    class _Conn: 
        def __enter__(self): 
            return self 
 
        def __exit__(self, exc_type, exc, tb): 
            return False 
 
    def fake_create_connection(address, timeout=0): 
        calls.append({"address": address, "timeout": timeout}) 
        return _Conn() 
 
    monkeypatch.setattr("app.services.telemetry_manager.socket.create_connection", fake_create_connection) 
    mgr.app = _FakeApp() 
 
    assert mgr._wait_for_px4(timeout=1) is True 
    assert calls == [{"address": ("10.8.0.9", 19001), "timeout": 2}] 
 
 
def test_build_collector_command_uses_runtime_config(): 
    mgr = TelemetryProcessManager() 
    mgr._reset_runtime_state() 
 
    class _FakeApp: 
        config = { 
            "TELEMETRY_MAVSDK_ADDRESS": "udp://10.0.0.5:14550", 
            "TELEMETRY_MAVSDK_GRPC_PORT": 50062, 
            "TELEMETRY_PYMAV_ADDRESS": "udpin:0.0.0.0:14031", 
            "TELEMETRY_FORWARD_HOST": "127.0.0.1", 
            "TELEMETRY_FORWARD_PORT": 16660, 
            "TELEMETRY_EMIT_RATE_HZ": 25, 
            "TELEMETRY_ENABLE_PYMAV": False, 
        } 
 
    mgr.app = _FakeApp() 
 
    assert mgr._build_collector_cmd("python-bin") == [ 
        "python-bin", 
        "-m", 
        "telemetry.collector", 
        "--host", 
        "127.0.0.1", 
        "--port", 
        "16660", 
        "--emit-rate", 
        "25", 
        "--mavsdk-address", 
        "udp://10.0.0.5:14550", 
        "--grpc-port", 
        "50062", 
        "--pymav-address", 
        "udpin:0.0.0.0:14031", 
        "--no-pymav", 
    ] 
