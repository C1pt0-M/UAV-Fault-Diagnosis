import os
from pathlib import Path
import sys
import time
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

if "uav_dashboard_api.services.detection_runner" not in sys.modules:
    fake_detection_runner = types.ModuleType("uav_dashboard_api.services.detection_runner")

    class _FakeDetectionRunner:
        def __init__(self, model_dir=None, device="cpu"):
            self.model_dir = model_dir
            self.device = device
            self.window_size = 150

    fake_detection_runner.DetectionRunner = _FakeDetectionRunner
    sys.modules["uav_dashboard_api.services.detection_runner"] = fake_detection_runner

if "uav_fault.data.build_sequence_cache" not in sys.modules:
    fake_build_sequence_cache = types.ModuleType("uav_fault.data.build_sequence_cache")
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
    sys.modules["uav_fault.data.build_sequence_cache"] = fake_build_sequence_cache

from uav_dashboard_api.services.telemetry_manager import TelemetryProcessManager
from uav_dashboard_api.services.telemetry_backends import PX4ScenarioBackend


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
    manager._stop_live_processing_pipeline(drain=False)
    manager._reset_runtime_state()
    manager._realtime_buffer.clear()
    manager._px4_proc = None
    manager._collector_proc = None
    manager._flight_proc = None
    manager._receiver = None
    manager._receiver_thread = None
    manager.app = None


def test_live_pipeline_decouples_frame_ingest_from_slow_inference(monkeypatch):
    mgr = TelemetryProcessManager()
    mgr._reset_runtime_state()
    mgr._backend_name = "rfly"
    mgr.app = types.SimpleNamespace(config={"ONLINE_INFERENCE_STRIDE": 5})

    class SlowDetector:
        window_size = 1
        aggregator = types.SimpleNamespace(onset_th=0.6)

        def update(self, *, frame, window, frame_idx):
            time.sleep(0.04)
            return ({"fault_prob": 0.1, "label": "normal", "confidence": 0.9}, [])

    mgr._detector = SlowDetector()
    mgr._detecting = True
    monkeypatch.setattr(
        mgr, "_canonicalize_frame", lambda frame, backend: (dict(frame), None)
    )
    mgr._start_live_processing_pipeline()
    started = time.perf_counter()
    for index in range(500):
        mgr._submit_backend_frame(
            {"seq": index, "time_s": index / 50.0, "accel_z": -9.8},
            source_backend="rfly",
        )
    submit_elapsed = time.perf_counter() - started
    mgr._stop_live_processing_pipeline(drain=True)

    assert submit_elapsed < 0.2
    assert mgr._frame_count == 500
    assert mgr._frame_queue_drops == 0
    assert mgr._inference_queue_drops > 0


def test_old_transport_callback_cannot_submit_into_new_session():
    mgr = TelemetryProcessManager()
    mgr._reset_runtime_state()
    mgr._clear_work_queue(mgr._frame_queue)
    old_backend = object()
    active_backend = object()
    mgr._backend = active_backend
    mgr._backend_name = "rfly"
    mgr._session_id = "TS-NEW"

    mgr.submit_transport_frame(
        {"seq": 1},
        [],
        source_backend="rfly",
        session_id="TS-OLD",
        backend=old_backend,
    )

    assert mgr._frame_queue.empty()


def test_live_pipeline_canonicalizes_each_frame_once(monkeypatch):
    mgr = TelemetryProcessManager()
    mgr._reset_runtime_state()
    mgr._backend_name = "rfly"
    calls = []

    def canonicalize(frame, backend):
        calls.append(frame["seq"])
        return dict(frame), None

    monkeypatch.setattr(mgr, "_canonicalize_frame", canonicalize)
    mgr._start_live_processing_pipeline()
    for index in range(200):
        mgr._submit_backend_frame(
            {"seq": index, "time_s": index / 50.0},
            source_backend="rfly",
        )
    mgr._stop_live_processing_pipeline(drain=True)

    assert calls == list(range(200))
    assert mgr._frame_count == 200


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
    mgr._warning_reason = "风险持续上升，异常概率已超过预警阈值。"
    mgr._preview_started_at = 10.0
    mgr._top_recommendations = [
        {"title": "结束当前架次后执行离线诊断", "priority": "P1"}
    ]
    mgr._last_frame_time_s = 13.4

    status = mgr.get_status()

    assert status["current_fault_prob"] == 0.62
    assert status["risk_score"] == 0.58
    assert status["risk_trend"] == "rising"
    assert status["warning_stage"] == "warning"
    assert "异常概率已超过预警阈值" in status["warning_reason"]
    assert status["preview_lead_time_s"] == 3.4
    assert status["top_recommendations"][0]["priority"] == "P1"


def test_network_link_status_updates_from_5g_frame_fields():
    mgr = TelemetryProcessManager()
    mgr._reset_runtime_state()
    mgr._backend_name = "rfly"
    mgr._network_link_status = mgr._initial_network_link_status(
        enabled=True,
        profile="enhanced_5g",
        status="running",
    )

    base_frame = {
        "time_s": 0.02,
        "actuator_out_1": 1300.0,
        "actuator_out_2": 1300.0,
        "actuator_out_3": 1300.0,
        "actuator_out_4": 1300.0,
        "att_q_w": 1.0,
        "att_q_x": 0.0,
        "att_q_y": 0.0,
        "att_q_z": 0.0,
        "vel_x": 0.1,
        "vel_y": 0.0,
        "vel_z": -0.1,
        "accel_x": 0.0,
        "accel_y": 0.0,
        "accel_z": -9.8,
    }
    first = {
        **base_frame,
        "seq": 1,
        "net_seq": 1,
        "net_profile": "enhanced_5g",
        "net_delay_ms": 18.0,
        "net_e2e_latency_ms": 20.0,
    }
    second = {
        **base_frame,
        "seq": 3,
        "net_seq": 3,
        "time_s": 0.06,
        "net_profile": "enhanced_5g",
        "net_delay_ms": 22.0,
        "net_e2e_latency_ms": 24.0,
    }

    mgr.process_transport_frame(first, [first], source_backend="rfly")
    mgr.process_transport_frame(second, [first, second], source_backend="rfly")

    status = mgr.get_status()["network_link"]
    assert status["enabled"] is True
    assert status["profile"] == "enhanced_5g"
    assert status["status"] == "active"
    assert status["rx_packets"] == 2
    assert status["lost_packets"] == 1
    assert status["loss_rate"] == pytest.approx(1 / 3)
    assert status["data_integrity"] == pytest.approx(2 / 3)
    assert status["avg_latency_ms"] == pytest.approx(20.0)
    assert status["latest_latency_ms"] == pytest.approx(22.0)


def test_network_link_ignores_non_contiguous_upstream_sequence_numbers():
    mgr = TelemetryProcessManager()
    mgr._reset_runtime_state()
    mgr._network_link_status = mgr._initial_network_link_status(
        enabled=True,
        profile="enhanced_5g",
        status="active",
    )
    base_frame = {
        "time_s": 0.02,
        "motor1": 0.1,
        "motor2": 0.1,
        "motor3": 0.1,
        "motor4": 0.1,
        "accel_x": 0.0,
        "accel_y": 0.0,
        "accel_z": -9.8,
        "net_profile": "enhanced_5g",
        "net_delay_ms": 20.0,
    }

    first = {**base_frame, "seq": 100, "net_seq": 1}
    second = {**base_frame, "seq": 250, "net_seq": 2, "time_s": 0.04}
    mgr.process_transport_frame(first, [first], source_backend="rfly")
    mgr.process_transport_frame(second, [first, second], source_backend="rfly")

    status = mgr.get_status()["network_link"]
    assert status["rx_packets"] == 2
    assert status["lost_packets"] == 0
    assert status["loss_rate"] == 0.0


def test_network_link_status_prefers_emulator_metrics_over_receiver_backlog(monkeypatch):
    mgr = TelemetryProcessManager()
    mgr._reset_runtime_state()
    mgr._network_link_status = mgr._initial_network_link_status(
        enabled=True,
        profile="enhanced_5g",
        status="active",
    )
    mgr._network_link_status.update(
        avg_latency_ms=304.0,
        p95_latency_ms=410.0,
        loss_rate=0.2,
        data_integrity=0.8,
    )
    monkeypatch.setattr(mgr, "_backend", types.SimpleNamespace(
        status=lambda: {
            "network_link": "running",
            "network_metrics": {
                "avg_latency_ms": 29.4,
                "p95_latency_ms": 32.0,
                "rx_packets": 1000,
                "tx_packets": 994,
                "lost_packets": 6,
                "loss_rate": 0.006,
                "data_integrity": 0.994,
                "queue_len": 1,
            },
        }
    ))

    status = mgr.get_status()["network_link"]

    assert status["avg_latency_ms"] == 29.4
    assert status["p95_latency_ms"] == 32.0
    assert status["loss_rate"] == 0.006
    assert status["data_integrity"] == 0.994


def test_predictive_warning_transitions_from_preview_to_confirmed():
    mgr = TelemetryProcessManager()
    mgr._reset_runtime_state()
    mgr._detecting = True
    original_record_fault_event = mgr._record_fault_event

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
    try:
        frame = {
            "time_s": 10.0,
            "seq": 1,
            "motor1": 1300.0,
            "motor2": 1305.0,
            "motor3": 1295.0,
            "motor4": 1310.0,
            "accel_z": -9.8,
            "vx": 0.1,
            "baro_alt_meter": 12.0,
        }

        mgr.process_transport_frame(frame, [frame], source_backend="px4")
        status_preview = mgr.get_status()
        assert status_preview["warning_stage"] in {"watch", "warning"}
        assert status_preview["latest_diagnoses"][0]["source"] == "realtime_preview"
        assert (
            status_preview["latest_diagnoses"][0]["evidence_summary"]["source"]
            == "online"
        )
        assert status_preview["latest_diagnoses"][0]["evidence_summary"]["top_evidence"]

        frame2 = {
            "time_s": 13.0,
            "seq": 2,
            "motor1": 1320.0,
            "motor2": 1315.0,
            "motor3": 1305.0,
            "motor4": 1325.0,
            "accel_z": -9.6,
            "vx": 0.2,
            "baro_alt_meter": 12.5,
        }
        mgr.process_transport_frame(frame2, [frame2], source_backend="px4")
        status_confirmed = mgr.get_status()
        assert status_confirmed["warning_stage"] == "critical"
        assert status_confirmed["fault_events_count"] == 1
        assert status_confirmed["latest_diagnoses"][0]["source"] == "realtime"
        assert status_confirmed["latest_diagnoses"][0]["lead_time_s"] == 3.0
    finally:
        mgr._record_fault_event = original_record_fault_event
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

    mgr.process_transport_frame(frame, window, source_backend="px4")
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
        "actuator_out_1": 1200.0,
        "actuator_out_2": 1190.0,
        "actuator_out_3": 1210.0,
        "actuator_out_4": 1185.0,
        "att_q_w": 0.9553364891,
        "att_q_x": 0.0,
        "att_q_y": 0.0,
        "att_q_z": 0.2955202067,
        "baro_alt": 15.2,
    }

    mgr.process_transport_frame(frame, [frame], source_backend="replay")
    data = mgr.get_realtime_data()

    assert data["times"] == [1.5]
    assert data["motor1"] == [1200.0]
    assert abs(data["roll"][0]) < 1e-3
    assert abs(data["pitch"][0]) < 1e-3
    assert abs(data["yaw"][0] - 0.6) < 1e-3
    assert data["pos_z"] == [15.2]


def test_realtime_data_exposes_5g_link_trend_series():
    mgr = TelemetryProcessManager()
    mgr._reset_runtime_state()
    mgr._realtime_buffer.clear()
    mgr._backend_name = "rfly"

    frame = {
        "time_s": 2.0,
        "seq": 10,
        "net_profile": "enhanced_5g",
        "net_delay_ms": 24.5,
        "net_e2e_latency_ms": 29.75,
    }
    mgr.process_transport_frame(frame, [frame], source_backend="rfly")
    data = mgr.get_realtime_data()

    assert data["net_latency_ms"] == [24.5]
    assert data["net_loss_rate"] == [0.0]
    assert data["net_data_integrity"] == [1.0]


def test_mec_result_is_consumed_without_local_detector():
    mgr = TelemetryProcessManager()
    mgr._reset_runtime_state()
    mgr._backend_name = "rfly"
    mgr._detecting = True
    mgr._detector = None
    mgr._mec_status = mgr._initial_mec_status(
        enabled=True, location="mec", status="running"
    )

    frame = {
        "seq": 1,
        "time_s": 3.0,
        "accel_z": -9.8,
        "mec_enabled": 1,
        "mec_inference_location": "mec",
        "mec_inference_count": 1,
        "mec_fault_prob": 0.72,
        "mec_label": "fault",
        "mec_confidence": 0.72,
        "mec_inference_ms": 4.2,
        "mec_avg_inference_ms": 4.0,
        "mec_p95_inference_ms": 5.1,
        "mec_decision_latency_ms": 31.2,
        "mec_avg_decision_latency_ms": 29.8,
        "mec_p95_decision_latency_ms": 35.4,
        "mec_event_seq": 0,
        "mec_events": [],
    }

    mgr.process_transport_frame(frame, [frame], source_backend="rfly")

    status = mgr.get_status()
    assert status["current_fault_prob"] == 0.72
    assert status["mec"]["location"] == "mec"
    assert status["mec"]["latest_inference_ms"] == 4.2
    assert status["inference"] == {
        "location": "mec",
        "status": "active",
        "latest_inference_ms": 4.2,
        "avg_inference_ms": 4.0,
        "p95_inference_ms": 5.1,
        "latest_decision_latency_ms": 31.2,
        "avg_decision_latency_ms": 29.8,
        "p95_decision_latency_ms": 35.4,
        "inference_count": 1,
        "inference_queue_drops": 0,
    }


def test_center_inference_status_tracks_runtime_and_data_to_decision_latency():
    mgr = TelemetryProcessManager()
    mgr._reset_runtime_state()
    mgr._center_inference_count = 1000

    mgr._record_center_inference_metrics(
        canonical={"net_tx_ms": 970.0},
        inference_ms=2.0,
        completed_ms=1000,
    )
    mgr._record_center_inference_metrics(
        canonical={"net_tx_ms": 960.0},
        inference_ms=4.0,
        completed_ms=1000,
    )
    mgr._record_center_inference_metrics(
        canonical={"net_tx_ms": 950.0},
        inference_ms=8.0,
        completed_ms=1000,
    )

    status = mgr.get_status()["inference"]
    assert status["location"] == "center"
    assert status["status"] == "active"
    assert status["latest_inference_ms"] == 8.0
    assert status["avg_inference_ms"] == 4.667
    assert status["p95_inference_ms"] == 4.0
    assert status["latest_decision_latency_ms"] == 50.0
    assert status["avg_decision_latency_ms"] == 40.0
    assert status["p95_decision_latency_ms"] == 40.0
    assert status["inference_count"] == 1003


def test_start_detection_marks_startup_phase_detecting():
    mgr = TelemetryProcessManager()
    mgr._reset_runtime_state()
    mgr._session_id = "TS-DET"
    mgr._start_time = 100.0
    mgr._planned_duration_s = 120.0
    mgr._backend_name = "px4"

    mgr._backend = types.SimpleNamespace(is_collecting=lambda: True)

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

    from uav_dashboard_api.services import telemetry_manager as telemetry_manager_module
    original_runner = telemetry_manager_module.DetectionRunner
    original_get_runner_cls = TelemetryProcessManager._get_detection_runner_cls

    class _FakeRunner:
        def __init__(self, model_dir, device="cpu"):
            self.window_size = 150

    telemetry_manager_module.DetectionRunner = _FakeRunner
    TelemetryProcessManager._get_detection_runner_cls = staticmethod(lambda: _FakeRunner)
    try:
        mgr.start_detection()
        status = mgr.get_status()
        assert status["startup_phase"] == "detecting"
    finally:
        telemetry_manager_module.DetectionRunner = original_runner
        TelemetryProcessManager._get_detection_runner_cls = original_get_runner_cls


def test_start_detection_wraps_detector_init_error_as_runtime_error():
    mgr = TelemetryProcessManager()
    mgr._reset_runtime_state()
    mgr._session_id = "TS-DET-ERR"
    mgr._start_time = 100.0
    mgr._planned_duration_s = 120.0
    mgr._backend_name = "replay"

    class _ReplayBackend:
        def is_collecting(self):
            return True

    mgr._backend = _ReplayBackend()
    mgr._resolve_online_model_meta = lambda requested_model=None: {
        "name": "online_detector_cnn1d_rfly",
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
    original_get_runner_cls = TelemetryProcessManager._get_detection_runner_cls

    class _BoomRunner:
        def __init__(self, model_dir, device="cpu"):
            raise KeyError("hidden_size")

    telemetry_manager_module.DetectionRunner = _BoomRunner
    TelemetryProcessManager._get_detection_runner_cls = staticmethod(lambda: _BoomRunner)
    try:
        with pytest.raises(RuntimeError) as exc_info:
            mgr.start_detection()
        error_text = str(exc_info.value)
        assert "online_detector_cnn1d_rfly" in error_text
        assert "hidden_size" in error_text
        assert mgr._detecting is False
        assert mgr._active_model is None
    finally:
        telemetry_manager_module.DetectionRunner = original_runner
        TelemetryProcessManager._get_detection_runner_cls = original_get_runner_cls


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
        "motor1": 1300.0,
        "motor2": 1305.0,
        "motor3": 1295.0,
        "motor4": 1310.0,
        "accel_z": -9.8,
        "vx": 0.1,
        "baro_alt_meter": 12.0,
    }
    mgr.process_transport_frame(frame, [frame], source_backend="px4")

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

    monkeypatch.setattr("uav_dashboard_api.services.telemetry_backends.subprocess.run", _fake_run)
    monkeypatch.setattr("uav_dashboard_api.services.telemetry_backends.subprocess.Popen", _fake_popen)

    mgr.app = _FakeApp()
    backend = PX4ScenarioBackend(
        mgr,
        session_id="TS-PX4-START",
        csv_path="session.csv",
        sim_time=30,
        options={},
    )
    backend._start_px4()

    assert len(calls) == 1
    assert calls[0]["cmd"] == [
        "bash",
        "-c",
        f"make -C {backend._get_px4_root()} px4_sitl gz_x500",
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
        "uav_dashboard_api.services.telemetry_manager.os.path.exists",
        lambda path: path in {legacy_linux, sys.executable},
    )
    monkeypatch.setattr(
        mgr,
        "_python_supports_telemetry",
        lambda python_bin: checked.append(python_bin) or True,
    )

    mgr.app = _FakeApp()
    resolved = mgr.resolve_telemetry_python_bin()

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

    monkeypatch.setattr("uav_dashboard_api.services.telemetry_backends.subprocess.run", _fake_run)
    monkeypatch.setattr("uav_dashboard_api.services.telemetry_backends.subprocess.Popen", _fake_popen)

    mgr.app = _FakeApp()
    backend = PX4ScenarioBackend(
        mgr,
        session_id="TS-PX4-ROOT",
        csv_path="session.csv",
        sim_time=30,
        options={},
    )
    backend._start_px4()

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
 
    monkeypatch.setattr("uav_dashboard_api.services.telemetry_backends.socket.create_connection", fake_create_connection)
    mgr.app = _FakeApp() 
    backend = PX4ScenarioBackend(
        mgr,
        session_id="TS-PX4-WAIT",
        csv_path="session.csv",
        sim_time=30,
        options={},
    )
    backend._startup_cancelled = lambda generation: False
 
    assert backend._wait_for_px4(timeout=1, generation=1) is True
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
    backend = PX4ScenarioBackend(
        mgr,
        session_id="TS-PX4-COLLECTOR",
        csv_path="session.csv",
        sim_time=30,
        options={},
    )
 
    assert backend._build_collector_cmd("python-bin") == [
        "python-bin", 
        "-m", 
        "uav_fault.telemetry.collector",
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


def test_runtime_paths_point_to_monorepo_package_roots():
    mgr = TelemetryProcessManager()
    mgr._reset_runtime_state()

    repo_root = BACKEND_ROOT.parents[2]

    assert mgr._src_dir == repo_root
    assert repo_root in mgr._pythonpath_dirs
    assert repo_root / "packages" / "uav_fault" in mgr._pythonpath_dirs


def test_px4_backend_uses_packaged_flight_controller_module():
    mgr = TelemetryProcessManager()
    mgr._reset_runtime_state()
    mgr._blind_test_ground_truth = {
        "fault_type": "motor_failure",
        "fault_delay_s": 12,
        "effectiveness": 0.2,
        "motor_index": 3,
    }
    backend = PX4ScenarioBackend(
        mgr,
        session_id="TS-PX4-001",
        csv_path="session.csv",
        sim_time=42,
        options={},
    )

    cmd = backend._flight_cmd("python-bin")

    assert cmd[:4] == [
        "python-bin",
        "-m",
        "uav_fault.telemetry.flight_controller",
        "--sim-time",
    ]
    assert "--fault-type" in cmd
    assert "motor_failure" in cmd


@pytest.mark.skipif(os.name != "nt", reason="Windows process-group behavior")
def test_stop_process_sends_ctrl_break_for_graceful_child_cleanup():
    import signal

    events = []

    class FakeProcess:
        pid = 4321

        def poll(self):
            return None

        def send_signal(self, sig):
            events.append(("signal", sig))

        def wait(self, timeout):
            events.append(("wait", timeout))
            return 0

    PX4ScenarioBackend._stop_process(FakeProcess(), "rfly_bridge")

    assert events == [
        ("signal", signal.CTRL_BREAK_EVENT),
        ("wait", 15),
    ]


def test_px4_backend_sets_timeout_reason_when_px4_never_becomes_ready(
    isolated_runtime_env, monkeypatch
):
    mgr = TelemetryProcessManager()
    mgr._reset_runtime_state()
    mgr.app = types.SimpleNamespace(
        config={
            "RUNTIME_DIR": isolated_runtime_env,
            "UPLOAD_DIR": isolated_runtime_env / "uploads",
            "DERIVED_DIR": isolated_runtime_env / "derived",
            "RUNS_DIR": isolated_runtime_env / "runs",
            "REPORTS_DIR": isolated_runtime_env / "reports",
        }
    )

    class _FakeThread:
        def __init__(self, target=None, name=None, daemon=None, args=(), kwargs=None):
            self._target = target
            self._name = name
            self._args = args
            self._kwargs = kwargs or {}

        def start(self):
            if self._target and self._name == "px4-transport-startup":
                self._target(*self._args, **self._kwargs)

        def is_alive(self):
            return False

        def join(self, timeout=None):
            return None

    monkeypatch.setattr("uav_dashboard_api.services.telemetry_backends.threading.Thread", _FakeThread)
    monkeypatch.setattr("uav_dashboard_api.services.telemetry_manager.threading.Thread", _FakeThread)
    monkeypatch.setattr(mgr, "_start_live_processing_pipeline", lambda: None)
    monkeypatch.setattr(
        mgr,
        "_stop_live_processing_pipeline",
        lambda drain=True: None,
    )
    monkeypatch.setattr(
        "uav_dashboard_api.services.telemetry_backends.PX4ScenarioBackend._start_px4",
        lambda self: None,
    )
    monkeypatch.setattr(
        mgr, "resolve_telemetry_python_bin", lambda: "/usr/bin/python3"
    )
    monkeypatch.setattr(
        "uav_dashboard_api.services.telemetry_backends.PX4ScenarioBackend._wait_for_px4",
        lambda self, timeout, generation: False,
    )
    monkeypatch.setattr(
        "uav_dashboard_api.services.telemetry_backends.TelemetryReceiver",
        lambda *args, **kwargs: types.SimpleNamespace(
            callbacks=[], stop=lambda: None
        ),
    )
    monkeypatch.setattr(mgr, "_write_log", lambda *args, **kwargs: None)
    monkeypatch.setattr(mgr, "_create_session_record", lambda *args, **kwargs: None)

    result = mgr.start_collection(
        drone_id="DRONE-PX4-READY-001",
        session_type="simulation",
        sim_time=30,
        backend="px4",
    )
    status = mgr.get_status()

    assert result["scenario_backend"] == "px4"
    assert status["startup_phase"] == "px4_not_ready"
    assert status["timeout_reason"] == "px4_not_ready"
    assert status["transport_status"] == "degraded"
