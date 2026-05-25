import csv
import io
import json
import sys
import time
import types
from datetime import datetime
from pathlib import Path

import bcrypt
import pytest
from flask import Flask

from uav_dashboard_api import create_app, db
from uav_dashboard_api.models.asset import Asset
from uav_dashboard_api.models.run_record import RunRecord
from uav_dashboard_api.models.telemetry_session import TelemetryFaultEvent, TelemetrySession
from uav_dashboard_api.models.user import User
from uav_dashboard_api.utils.path_store import resolve_local_path
from uav_dashboard_api.utils.jwt_utils import JWTUtils
from uav_dashboard_api.config import TestingConfig


REPO_ROOT = Path(__file__).resolve().parents[4]
FRONTEND_SRC = REPO_ROOT / "apps" / "dashboard" / "frontend" / "src"


def _write_json(path: Path, payload: dict):
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
    monkeypatch.setitem(
        sys.modules, "data.build_sequence_cache", fake_build_sequence_cache
    )
    monkeypatch.setitem(
        sys.modules, "uav_fault.data.build_sequence_cache", fake_build_sequence_cache
    )

    model_root = tmp_path / "models"
    runtime_dir = tmp_path / "runtime"
    model_root.mkdir()
    _create_model(
        model_root, "lstm", {"model": "lstm", "in_channels": 48, "num_classes": 4}
    )

    class LocalTestingConfig(TestingConfig):
        SQLALCHEMY_DATABASE_URI = f"sqlite:///{tmp_path / 'test_main_flows.db'}"
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

    app = create_app(LocalTestingConfig)
    app.config["TESTING"] = True
    with app.app_context():
        db.create_all()
        yield app
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


def test_offline_prepare_run_report_main_flow(app, client, auth_header, monkeypatch):
    from uav_dashboard_api.routes import dashboard as dashboard_routes

    def fake_submit(app_obj, run_id, worker_fn, *args, **kwargs):
        with app_obj.app_context():
            run = db.session.get(RunRecord, run_id)
            run.status = "success"
            run.progress = 1.0
            run.finished_at = datetime.now()
            if run.task == "convert_ulg":
                asset = db.session.get(Asset, run.asset_id)
                prepared = (
                    Path(app_obj.config["DERIVED_DIR"])
                    / f"asset_{asset.id}"
                    / "prepared.csv"
                )
                prepared.parent.mkdir(parents=True, exist_ok=True)
                prepared.write_text(
                    "timestamp,motor1,battery_voltage\n0,0.1,11.1\n", encoding="utf-8"
                )
                meta = asset.get_meta()
                meta["prepared_csv_path"] = str(prepared)
                asset.set_meta(meta)
                run.set_result({"asset_id": asset.id, "output_csv": str(prepared)})
            else:
                run.set_result(
                    {
                        "prediction": {"label": "motor", "confidence": 0.91},
                        "probs": {"motor": 0.91, "normal": 0.09},
                        "severity": {
                            "level": "heavy",
                            "score": 0.88,
                            "method": "rule_v1",
                        },
                        "recommendations": [
                                    {"title": "立即检查电机", "priority": "P1"}
                        ],
                        "diagnosis": {
                            "source": "offline",
                            "label": "motor",
                            "probability": 0.91,
                            "severity": "heavy",
                            "decision": {
                                "episode_id": "ep_test_001",
                                "state": "DECIDED",
                                "fault_label": "motor",
                                "severity": "heavy",
                                "evidence_summary": {
                                    "source": "offline",
                                    "fault_label": "motor",
                                    "severity": {
                                        "level": "heavy",
                                        "score": 0.88,
                                        "method": "rule_v1",
                                    },
                                    "top_evidence": [
                                        {"metric": "motor_imbalance", "status": "used"}
                                    ],
                                },
                                "recommendations": [
                                    {"title": "立即检查电机", "priority": "P1"}
                                ],
                            },
                        },
                        "unified_decision": {
                            "episode_id": "ep_test_001",
                            "state": "DECIDED",
                            "fault_label": "motor",
                            "severity": "heavy",
                            "evidence_summary": {
                                "source": "offline",
                                "fault_label": "motor",
                                "severity": {
                                    "level": "heavy",
                                    "score": 0.88,
                                    "method": "rule_v1",
                                },
                                "top_evidence": [
                                    {"metric": "motor_imbalance", "status": "used"}
                                ],
                            },
                            "recommendations": [
                                    {"title": "立即检查电机", "priority": "P1"}
                            ],
                        },
                    }
                )
            db.session.commit()

    monkeypatch.setattr(dashboard_routes, "submit_background_run", fake_submit)
    upload = client.post(
        "/api/assets/upload",
        headers=auth_header,
        data={"file": (io.BytesIO(b"ULGDATA"), "mission.ulg")},
    )
    asset_id = upload.get_json()["data"]["id"]
    assert upload.status_code == 201
    assert (
        client.post(f"/api/assets/{asset_id}/prepare", headers=auth_header).status_code
        == 202
    )
    asset_detail = client.get(
        f"/api/assets/{asset_id}", headers=auth_header
    ).get_json()["data"]
    assert asset_detail["meta"]["prepared_csv_path"]

    run_resp = client.post(
        "/api/runs",
        headers=auth_header,
        json={"task": "offline_multiclass", "asset_id": asset_id, "model": "lstm"},
    )
    run_id = run_resp.get_json()["data"]["id"]
    assert run_resp.status_code == 202
    assert run_resp.get_json()["data"]["task"] == "offline_multiclass"
    result = client.get(f"/api/runs/{run_id}/result", headers=auth_header).get_json()[
        "data"
    ]
    assert result["diagnosis"]["label"] == "motor"
    assert result["diagnosis"]["severity"] == "heavy"
    assert result["unified_decision"]["state"] == "DECIDED"
    assert result["diagnosis"]["decision"]["fault_label"] == "motor"
    assert result["unified_decision"]["episode_id"] == "ep_test_001"
    assert result["unified_decision"]["evidence_summary"]["source"] == "offline"

    report = client.post(
        "/api/reports/diagnosis", headers=auth_header, json={"run_id": run_id}
    )
    report_id = report.get_json()["data"]["id"]
    assert report.status_code == 201
    report_detail = client.get(
        f"/api/reports/diagnosis/{report_id}", headers=auth_header
    ).get_json()["data"]
    assert report_detail["diagnosis"]["label"] == "motor"
    assert report_detail["diagnosis"]["severity"] == "heavy"
    assert report_detail["diagnosis"]["model"]["task"] == "offline_multiclass"
    assert report_detail["summary"]["decision"]["state"] == "DECIDED"
    assert report_detail["summary"]["decision"]["episode_id"] == "ep_test_001"


def test_offline_task_requires_canonical_name_and_filters_legacy_records(
    app, client, auth_header, monkeypatch
):
    from uav_dashboard_api.routes import dashboard as dashboard_routes

    def fake_submit(app_obj, run_id, worker_fn, *args, **kwargs):
        with app_obj.app_context():
            run = db.session.get(RunRecord, run_id)
            run.status = "success"
            run.progress = 1.0
            run.finished_at = datetime.now()
            run.set_result(
                {
                    "prediction": {"label": "sensor", "confidence": 0.87},
                    "probs": {"sensor": 0.87, "normal": 0.13},
                }
            )
            db.session.commit()

    monkeypatch.setattr(dashboard_routes, "submit_background_run", fake_submit)
    upload = client.post(
        "/api/assets/upload",
        headers=auth_header,
        data={"file": (io.BytesIO(b"timestamp,a\n0,1\n"), "mission.csv")},
    )
    asset_id = upload.get_json()["data"]["id"]
    with app.app_context():
        asset = db.session.get(Asset, asset_id)
        meta = asset.get_meta()
        meta["prepared_csv_path"] = asset.stored_path
        asset.set_meta(meta)
        db.session.commit()

    legacy_submit = client.post(
        "/api/runs",
        headers=auth_header,
        json={"task": "offline_5class", "asset_id": asset_id, "model": "lstm"},
    )
    assert legacy_submit.status_code == 400
    assert "不支持的任务类型" in legacy_submit.get_json()["message"]

    run_resp = client.post(
        "/api/runs",
        headers=auth_header,
        json={"task": "offline_multiclass", "asset_id": asset_id, "model": "lstm"},
    )
    assert run_resp.status_code == 202
    run_id = run_resp.get_json()["data"]["id"]
    assert run_resp.get_json()["data"]["task"] == "offline_multiclass"

    with app.app_context():
        legacy_run = RunRecord(
            task="offline_5class", status="success", progress=1.0, asset_id=asset_id
        )
        legacy_run.set_result({"prediction": {"label": "motor", "confidence": 0.8}})
        db.session.add(legacy_run)
        db.session.commit()
        legacy_run_id = legacy_run.id

    legacy_report = client.post(
        "/api/reports/diagnosis", headers=auth_header, json={"run_id": legacy_run_id}
    )
    assert legacy_report.status_code == 400
    assert legacy_report.get_json()["message"] == "仅支持离线分类诊断任务"

    legacy_list = client.get("/api/runs?task=offline_5class", headers=auth_header)
    canonical_list = client.get(
        "/api/runs?task=offline_multiclass", headers=auth_header
    )
    assert legacy_list.status_code == 200
    assert canonical_list.status_code == 200
    assert {item["id"] for item in legacy_list.get_json()["data"]} == {legacy_run_id}
    assert {item["id"] for item in canonical_list.get_json()["data"]} == {run_id}
    assert run_id in {item["id"] for item in canonical_list.get_json()["data"]}

    overview = client.get("/api/overview", headers=auth_header)
    assert overview.status_code == 200
    fault_distribution = overview.get_json()["data"]["fault_distribution"]
    assert fault_distribution["sensor"] == 1
    assert fault_distribution["motor"] == 0


def test_offline_run_accepts_telemetry_session_bridge(
    app, client, auth_header, monkeypatch, tmp_path: Path
):
    from uav_dashboard_api.routes import dashboard as dashboard_routes

    submit_calls = []

    def fake_submit(app_obj, run_id, worker_fn, *args, **kwargs):
        submit_calls.append((run_id, worker_fn.__name__, args, kwargs))

    monkeypatch.setattr(dashboard_routes, "submit_background_run", fake_submit)
    csv_path = tmp_path / "telemetry_bridge.csv"
    csv_path.write_text("timestamp,motor1\n0,1\n", encoding="utf-8")

    with app.app_context():
        db.session.add(
            TelemetrySession(
                id="TS-BRIDGE-001",
                session_type="simulation",
                csv_path=str(csv_path),
                fault_events_count=1,
                status="completed",
            )
        )
        db.session.commit()

    response = client.post(
        "/api/runs",
        headers=auth_header,
        json={"task": "offline_multiclass", "session_id": "TS-BRIDGE-001", "model": "lstm"},
    )

    assert response.status_code == 202
    payload = response.get_json()["data"]
    assert payload["task"] == "offline_multiclass"
    assert payload["asset_id"] is not None
    assert payload["params"]["session_id"] == "TS-BRIDGE-001"
    assert payload["params"]["source_type"] == "telemetry_session"
    assert payload["params"]["model"] == "lstm"
    assert submit_calls == [
        (
            payload["id"],
            "_offline_5class_worker",
            (payload["asset_id"], "lstm"),
            {},
        )
    ]

    with app.app_context():
        run = db.session.get(RunRecord, payload["id"])
        assert run is not None
        assert run.asset_id == payload["asset_id"]
        assert run.get_params()["session_id"] == "TS-BRIDGE-001"
        assert run.get_params()["source_type"] == "telemetry_session"


def test_offline_run_rejects_missing_telemetry_session(app, client, auth_header):
    response = client.post(
        "/api/runs",
        headers=auth_header,
        json={"task": "offline_multiclass", "session_id": "TS-MISSING-001", "model": "lstm"},
    )

    assert response.status_code == 404
    assert response.get_json()["message"] == "遥测会话不存在"


def test_offline_run_rejects_telemetry_session_without_csv(app, client, auth_header):
    with app.app_context():
        db.session.add(
            TelemetrySession(
                id="TS-NO-CSV-001",
                session_type="simulation",
                status="completed",
            )
        )
        db.session.commit()

    response = client.post(
        "/api/runs",
        headers=auth_header,
        json={"task": "offline_multiclass", "session_id": "TS-NO-CSV-001", "model": "lstm"},
    )

    assert response.status_code == 409
    assert response.get_json()["message"] == "session_csv_unavailable"


def test_offline_run_rejects_telemetry_session_with_missing_csv_file(
    app, client, auth_header, tmp_path: Path
):
    missing_csv = tmp_path / "missing_telemetry_bridge.csv"

    with app.app_context():
        db.session.add(
            TelemetrySession(
                id="TS-MISSING-CSV-001",
                session_type="simulation",
                csv_path=str(missing_csv),
                status="completed",
            )
        )
        db.session.commit()

    response = client.post(
        "/api/runs",
        headers=auth_header,
        json={
            "task": "offline_multiclass",
            "session_id": "TS-MISSING-CSV-001",
            "model": "lstm",
        },
    )

    assert response.status_code == 409
    assert response.get_json()["message"] == "session_csv_unavailable"


def test_telemetry_receiver_persists_live_baro_temp_and_actuator_controls(
    tmp_path: Path, monkeypatch
):
    src_root = REPO_ROOT / "packages" / "uav_fault"
    if str(src_root) not in sys.path:
        sys.path.insert(0, str(src_root))

    from uav_fault.telemetry.receiver import TelemetryReceiver

    payload = {
        "seq": 12,
        "time_s": 1.25,
        "motor1": 1200,
        "motor2": 1190,
        "motor3": 1210,
        "motor4": 1180,
        "baro_temp": 24.5,
        "actuator_ctrl_1": 0.11,
        "actuator_ctrl_2": -0.22,
        "actuator_ctrl_3": 0.33,
        "actuator_ctrl_4": 0.44,
    }

    class _FakeSocket:
        def __init__(self, packets):
            self._packets = list(packets)

        def settimeout(self, timeout):
            self.timeout = timeout

        def bind(self, addr):
            self.addr = addr

        def recvfrom(self, size):
            if self._packets:
                return self._packets.pop(0), ("127.0.0.1", 15550)
            raise KeyboardInterrupt

        def close(self):
            return None

    monkeypatch.setattr(
        "uav_fault.telemetry.receiver.socket.socket",
        lambda *args, **kwargs: _FakeSocket([json.dumps(payload).encode("utf-8")]),
    )

    rows = []
    csv_path = tmp_path / "receiver_live.csv"
    receiver = TelemetryReceiver(
        port=15550,
        csv_path=str(csv_path),
        window_size=4,
        auto_stop=0.0,
        recv_timeout=0.1,
        trim_on_stop=False,
    )
    receiver.callbacks.append(lambda row, window: rows.append((row, list(window))))

    receiver.start()

    with csv_path.open(newline="", encoding="utf-8") as f:
        saved_rows = list(csv.DictReader(f))

    assert len(saved_rows) == 1
    assert saved_rows[0]["baro_temp"] == "24.5"
    assert saved_rows[0]["actuator_ctrl_1"] == "0.11"
    assert saved_rows[0]["actuator_ctrl_2"] == "-0.22"
    assert saved_rows[0]["actuator_ctrl_3"] == "0.33"
    assert saved_rows[0]["actuator_ctrl_4"] == "0.44"
    assert rows[0][0]["baro_temp"] == 24.5
    assert rows[0][0]["actuator_ctrl_1"] == 0.11
    assert rows[0][1][0]["baro_temp"] == 24.5
    assert rows[0][1][0]["actuator_ctrl_4"] == 0.44


def test_telemetry_start_detection_status_and_session_detail_flow(
    app, client, auth_header, monkeypatch
):
    from uav_dashboard_api.services.telemetry_manager import telemetry_manager

    diagnosis = {
        "source": "realtime",
        "label": "fault",
        "probability": 0.96,
        "confidence": 0.96,
        "severity": "heavy",
        "recommendations": [{"title": "立即执行人工复核", "priority": "P0"}],
        "model": {"name": "online_detector_alpha"},
        "time_range": {"start_time": 1.0, "end_time": 2.5, "duration": 1.5},
    }
    with app.app_context():
        session = TelemetrySession(
            id="TS-000001",
            session_type="simulation",
            scenario_backend="replay",
            schema_version="rfly_seq_v1",
            source_path="/tmp/replay.csv",
            start_time=datetime.now(),
            end_time=datetime.now(),
            duration_seconds=12.0,
            frame_count=120,
            has_detection=True,
            fault_events_count=1,
            status="completed",
        )
        session.set_meta(
            {"replay_metrics": {"pred_event_count": 1, "hit": 1, "false_alarm": 0}}
        )
        event = TelemetryFaultEvent(
            id="FE-000001",
            session_id="TS-000001",
            start_time=1.0,
            end_time=2.5,
            duration=1.5,
            max_fault_prob=0.96,
            mean_fault_prob=0.9,
            data_points=18,
            severity="critical",
        )
        event.set_diagnosis(diagnosis)
        db.session.add(session)
        db.session.add(event)
        db.session.commit()

    detection_calls = []

    monkeypatch.setattr(
        telemetry_manager,
        "start_collection",
        lambda **kwargs: {"session_id": "TS-000001", "collector_status": "running"},
    )
    monkeypatch.setattr(
        telemetry_manager,
        "start_detection",
        lambda requested_model=None: (
            detection_calls.append(requested_model)
            or {
                "detector_status": "running",
                "model": {"name": requested_model or "online_detector_alpha"},
            }
        ),
    )
    monkeypatch.setattr(
        telemetry_manager,
        "get_status",
        lambda: {
            "is_collecting": True,
            "is_detecting": True,
            "is_blind_test": False,
            "current_session_id": "TS-000001",
            "uptime_seconds": 12.0,
            "planned_duration_s": 120.0,
            "frame_count": 120,
            "fault_events_count": 1,
            "fault_events": [
                {
                    "id": "FE-000001",
                    "start_time": 1.0,
                    "end_time": 2.5,
                    "duration": 1.5,
                    "max_fault_prob": 0.96,
                    "severity": "heavy",
                    "diagnosis": diagnosis,
                }
            ],
            "latest_diagnoses": [diagnosis],
            "online_model": {"name": "online_detector_alpha"},
            "transport_status": "connected",
            "business_status": "detecting",
            "scenario_backend": "replay",
            "schema_version": "rfly_seq_v1",
            "feature_coverage": {"mapped": ["roll"], "derived": ["yaw"], "default": []},
            "replay_metrics": {"pred_event_count": 1, "hit": 1, "false_alarm": 0},
            "processes": {
                "collector": "running",
                "receiver": "running",
                "detector": "running",
                "px4_sitl": "running",
            },
        },
    )

    assert (
        client.post(
            "/api/telemetry/start-collection",
            headers=auth_header,
            json={
                "drone_id": "DRONE-001",
                "sim_time": 0,
                "backend": "replay",
                "replay_path": "/tmp/replay.csv",
            },
        ).status_code
        == 200
    )
    assert (
        client.post(
            "/api/telemetry/start-detection",
            headers=auth_header,
            json={"model": "online_detector_beta"},
        ).status_code
        == 200
    )
    assert detection_calls == ["online_detector_beta"]
    status = client.get("/api/telemetry/status", headers=auth_header).get_json()["data"]
    assert status["transport_status"] == "connected"
    assert status["business_status"] == "detecting"
    assert status["scenario_backend"] == "replay"
    assert status["schema_version"] == "rfly_seq_v1"
    assert status["replay_metrics"]["hit"] == 1
    assert status["fault_events"][0]["diagnosis"]["severity"] == "heavy"

    detail = client.get(
        "/api/telemetry/sessions/TS-000001", headers=auth_header
    ).get_json()["data"]
    assert detail["scenario_backend"] == "replay"
    assert detail["schema_version"] == "rfly_seq_v1"
    assert detail["meta"]["replay_metrics"]["pred_event_count"] == 1


def test_telemetry_replay_session_detail_persists_session_fields(
    app, client, auth_header
):
    diagnosis = {
        "source": "realtime",
        "label": "fault",
        "probability": 0.98,
        "severity": "heavy",
        "time_range": {"start_time": 11.2, "end_time": 14.8, "duration": 3.6},
    }
    started_at = datetime(2026, 3, 21, 10, 0, 0)
    ended_at = datetime(2026, 3, 21, 10, 0, 19)
    with app.app_context():
        session = TelemetrySession(
            id="TS-REPLAY-002",
            session_type="simulation",
            scenario_backend="replay",
            schema_version="rfly_seq_v1",
            source_path="/tmp/rfly/replay.csv",
            start_time=started_at,
            end_time=ended_at,
            duration_seconds=18.5,
            frame_count=925,
            csv_path="/tmp/runtime/replay_session.csv",
            has_detection=True,
            fault_events_count=1,
            status="completed",
            created_by="tester",
        )
        session.set_meta(
            {
                "replay_metrics": {
                    "pred_event_count": 1,
                    "hit": 1,
                    "false_alarm": 0,
                    "detection_latency_s": 0.6,
                }
            }
        )
        event = TelemetryFaultEvent(
            id="FE-REPLAY-002",
            session_id="TS-REPLAY-002",
            start_time=1.0,
            end_time=2.0,
            duration=1.0,
            max_fault_prob=0.98,
            mean_fault_prob=0.93,
            data_points=25,
            severity="critical",
        )
        event.set_diagnosis(diagnosis)
        db.session.add(session)
        db.session.add(event)
        db.session.commit()

    response = client.get("/api/telemetry/sessions/TS-REPLAY-002", headers=auth_header)

    assert response.status_code == 200
    detail = response.get_json()["data"]
    assert detail["source_path"] == "/tmp/rfly/replay.csv"
    assert detail["start_time"] == "2026-03-21 10:00:00"
    assert detail["end_time"] == "2026-03-21 10:00:19"
    assert detail["duration_seconds"] == 18.5
    assert detail["frame_count"] == 925
    assert detail["csv_path"] == "/tmp/runtime/replay_session.csv"
    assert detail["has_detection"] is True
    assert detail["fault_events_count"] == 1
    assert detail["status"] == "completed"
    assert detail["created_by"] == "tester"
    assert detail["meta"]["replay_metrics"]["detection_latency_s"] == 0.6
    assert detail["latest_diagnosis"]["label"] == "fault"
    assert detail["fault_events"][0]["severity"] == "heavy"
    assert detail["fault_events"][0]["severity_legacy"] == "critical"
    assert detail["fault_events"][0]["start_time"] == 11.2
    assert detail["fault_events"][0]["end_time"] == 14.8
    assert detail["fault_events"][0]["duration"] == 3.6


def test_telemetry_replay_session_detail_builds_fault_evidence_timeline(
    app, client, auth_header
):
    diagnosis = {
        "source": "realtime",
        "label": "fault",
        "probability": 0.98,
        "severity": "heavy",
        "lead_time_s": 1.2,
        "time_range": {"start_time": 11.2, "end_time": 14.8, "duration": 3.6},
        "recommendations": [{"title": "立即执行人工复核", "priority": "P0"}],
        "evidence_summary": {
            "top_evidence": [
                {"kind": "signal", "name": "fault_probability", "value": 0.98, "status": "observed"}
            ]
        },
    }
    with app.app_context():
        db.session.add(TelemetrySession(id="TS-EVIDENCE-001", session_type="simulation", fault_events_count=1, status="completed"))
        event = TelemetryFaultEvent(
            id="FE-EVIDENCE-001", session_id="TS-EVIDENCE-001", start_time=11.2, end_time=14.8, duration=3.6, severity="critical"
        )
        event.set_diagnosis(diagnosis)
        db.session.add(event)
        db.session.commit()

    response = client.get("/api/telemetry/sessions/TS-EVIDENCE-001", headers=auth_header)

    assert response.status_code == 200
    detail = response.get_json()["data"]
    timeline = detail["fault_events"][0]["evidence_timeline"]
    assert [item["stage"] for item in timeline] == ["preview", "confirmed", "window_end", "action"]
    assert timeline[0]["ts"] == pytest.approx(10.0)
    assert timeline[1]["ts"] == pytest.approx(11.2)
    assert timeline[1]["evidence"][0]["name"] == "fault_probability"
    assert timeline[3]["status"] == "recommended"
    assert "立即执行人工复核" in timeline[3]["description"]


def test_telemetry_frontend_session_detail_dialog_displays_replay_metrics_contract():
    dialog = FRONTEND_SRC / "modules" / "monitoring" / "components" / "MonitoringSessionDialog.vue"
    content = dialog.read_text(encoding="utf-8")

    assert "monitoringCopy.sessionDetailTitle" in content
    assert "sessionDetail.fault_events_count" in content
    assert "monitoringCopy.faultEventsLabel" in content


def test_telemetry_frontend_exposes_fault_handoff_contract():
    monitoring_page = FRONTEND_SRC / "modules" / "monitoring" / "MonitoringPage.vue"
    content = monitoring_page.read_text(encoding="utf-8")

    assert "faultHandoffPayload" in content
    assert "faultHandoffPayload.fault_events_count" in content
    assert "faultHandoffPayload.session_id" in content


def test_telemetry_frontend_displays_runtime_explainability_fields():
    monitoring_copy = FRONTEND_SRC / "modules" / "monitoring" / "monitoringView.js"
    content = monitoring_copy.read_text(encoding="utf-8")

    assert "modelPlaceholder" in content
    assert "faultEventsLabel" in content
    assert "unknownFault" in content
    assert "replayAssetRequired" in content


def test_telemetry_frontend_uses_current_monitoring_model_and_replay_contract():
    control_panel = FRONTEND_SRC / "modules" / "monitoring" / "components" / "MonitoringControlPanel.vue"
    dashboard = FRONTEND_SRC / "modules" / "monitoring" / "composables" / "useMonitoringDashboard.js"
    control_content = control_panel.read_text(encoding="utf-8")
    dashboard_content = dashboard.read_text(encoding="utf-8")

    assert "monitoring-control--model" in control_content
    assert "selectedReplayAssetId" in control_content
    assert "assetApi.list('all')" in dashboard_content
    assert "selectedBackend.value === 'replay'" in dashboard_content
    assert "runStatus.fault_events" in dashboard_content


def test_telemetry_frontend_prompts_for_fault_classification_after_stop():
    monitoring_page = FRONTEND_SRC / "modules" / "monitoring" / "MonitoringPage.vue"
    dashboard = FRONTEND_SRC / "modules" / "monitoring" / "composables" / "useMonitoringDashboard.js"
    page_content = monitoring_page.read_text(encoding="utf-8")
    dashboard_content = dashboard.read_text(encoding="utf-8")

    assert "faultHandoffPayload" in page_content
    assert "fault_events_count" in page_content
    assert "session_id" in page_content
    assert "router.push" in dashboard_content
    assert "session_id: sessionId" in dashboard_content
    assert "auto: '1'" in dashboard_content


def test_fault_detection_frontend_supports_session_auto_route_contract():
    diagnosis_workspace = (
        FRONTEND_SRC / "modules" / "diagnosis" / "composables" / "useDiagnosisWorkspace.js"
    )
    content = diagnosis_workspace.read_text(encoding="utf-8")

    assert "route.query.session_id" in content
    assert "route.query.auto === '1'" in content
    assert "session_id: sessionId" in content
    assert "assetApi.list('all')" in content


def test_run_api_supports_session_id_for_offline_detection_contract():
    api_file = FRONTEND_SRC / "shared" / "services" / "runs.js"
    content = api_file.read_text(encoding="utf-8")

    assert "session_id: payload.sessionId" in content
    assert "asset_id: payload.assetId" in content


def test_analysis_fault_prob_curve_creates_missing_derived_asset_dir(
    app, client, auth_header, tmp_path: Path, monkeypatch
):
    csv_path = tmp_path / "fault_prob_curve_case.csv"
    csv_path.write_text(
        "time_s,roll,pitch,yaw\n"
        "0.0,0.1,0.0,0.0\n"
        "0.1,0.2,0.0,0.0\n"
        "0.2,0.3,0.0,0.0\n",
        encoding="utf-8",
    )

    with app.app_context():
        user = User.query.filter_by(username="flow_user").first()
        asset = Asset(
            filename=csv_path.name,
            kind="csv",
            stored_path=str(csv_path),
            size_bytes=csv_path.stat().st_size,
            sha256="test-sha256-fault-prob-curve",
            created_by=user.id,
        )
        db.session.add(asset)
        db.session.commit()
        asset_id = asset.id

    class _Completed:
        returncode = 0
        stdout = '{"time_s": 0.1, "fault_prob": 0.25}\n{"time_s": 0.2, "fault_prob": 0.8}\n'
        stderr = ""

    monkeypatch.setattr(
        "uav_dashboard_api.routes.dashboard.resolve_registered_model",
        lambda *args, **kwargs: {
            "name": "online_detector_cnn1d_rfly",
            "path": str(tmp_path / "fake_model"),
            "task": "online_binary",
            "architecture": "cnn1d",
            "label_names": ["normal", "fault"],
        },
    )
    monkeypatch.setattr(
        "subprocess.run",
        lambda *args, **kwargs: _Completed(),
    )

    resp = client.get(
        f"/api/analysis/fault_prob_curve?asset_id={asset_id}",
        headers=auth_header,
    )

    assert resp.status_code == 200
    payload = resp.get_json()
    assert payload["success"] is True
    assert payload["data"] == [
        {"time_s": 0.1, "fault_prob": 0.25},
        {"time_s": 0.2, "fault_prob": 0.8},
    ]

    cache_path = (
        Path(app.config["DERIVED_DIR"])
        / f"asset_{asset_id}"
        / "fault_prob_curve_online_detector_cnn1d_rfly_v2.json"
    )
    assert cache_path.exists()
    assert json.loads(cache_path.read_text(encoding="utf-8")) == payload["data"]


def test_analysis_fields_accepts_legacy_linux_runtime_csv_path(
    app, client, auth_header
):
    csv_dir = Path(app.config["UPLOAD_DIR"]) / "csv"
    csv_dir.mkdir(parents=True, exist_ok=True)
    csv_name = "20260326143545_2a2be49b_Case_2508000005.csv"
    csv_path = csv_dir / csv_name
    csv_path.write_text(
        "time_s,motor1,motor2\n"
        "0.0,1200,1190\n"
        "0.1,1210,1200\n",
        encoding="utf-8",
    )

    with app.app_context():
        user = User.query.filter_by(username="flow_user").first()
        asset = Asset(
            filename=csv_name,
            kind="csv",
            stored_path=(
                "/home/jack/UAV-Fault-Diagnosis/apps/dashboard/backend/"
                f"runtime/uploads/csv/{csv_name}"
            ),
            size_bytes=csv_path.stat().st_size,
            sha256="legacy-linux-runtime-csv-sha256",
            created_by=user.id,
        )
        db.session.add(asset)
        db.session.commit()
        asset_id = asset.id

    response = client.get(
        f"/api/analysis/fields?asset_id={asset_id}",
        headers=auth_header,
    )

    assert response.status_code == 200
    payload = response.get_json()
    assert payload["success"] is True
    fields = {
        item["field"]
        for group in payload["data"]["field_groups"]
        for item in group["fields"]
    }
    assert "motor1" in fields
    assert payload["data"]["time"]["key"] == "time_s"


def test_telemetry_frontend_only_prompts_after_confirmed_fault_events_contract():
    dashboard = FRONTEND_SRC / "modules" / "monitoring" / "composables" / "useMonitoringDashboard.js"
    page = FRONTEND_SRC / "modules" / "monitoring" / "MonitoringPage.vue"
    content = dashboard.read_text(encoding="utf-8")
    page_content = page.read_text(encoding="utf-8")

    assert "Number(response.data?.fault_events_count || 0) <= 0" in content
    assert "openFaultHandoffDialog(response.data)" in content
    assert "faultHandoffDialogVisible.value = true" in content
    assert "faultHandoffPayload" in page_content


def test_telemetry_frontend_keeps_blind_reveal_before_fault_classification_contract():
    control_panel = FRONTEND_SRC / "modules" / "monitoring" / "components" / "MonitoringControlPanel.vue"
    dashboard = FRONTEND_SRC / "modules" / "monitoring" / "composables" / "useMonitoringDashboard.js"
    control_content = control_panel.read_text(encoding="utf-8")
    dashboard_content = dashboard.read_text(encoding="utf-8")

    assert "v-if=\"blindTestEnabled\"" in control_content
    assert "await telemetryApi.startBlindTest" in dashboard_content
    assert "startPolling()" in dashboard_content


def test_telemetry_replay_session_auto_finalize_persists_runtime_fields(
    app, auth_header, tmp_path: Path
):
    from uav_dashboard_api.services.telemetry_manager import telemetry_manager

    replay_path = tmp_path / "replay_auto_finalize.csv"
    replay_path.write_text(
        "time_s,roll,pitch,yaw,is_fault_active\n"
        "0.0,1.0,0.0,0.0,0\n"
        "0.1,1.5,0.1,0.0,1\n"
        "0.2,1.8,0.1,0.0,1\n",
        encoding="utf-8",
    )

    class _FakeThread:
        _deferred = []
        _active_replay = None

        def __init__(self, target=None, name=None, daemon=None, args=(), kwargs=None):
            self._target = target
            self._args = args
            self._kwargs = kwargs or {}
            self._name = name
            self._alive = False

        def _run_now(self):
            self._alive = True
            is_replay = self._name == "telemetry-replay"
            if is_replay:
                _FakeThread._active_replay = self
            try:
                if self._target:
                    self._target(*self._args, **self._kwargs)
            finally:
                self._alive = False
                if is_replay:
                    _FakeThread._active_replay = None
                    deferred = list(_FakeThread._deferred)
                    _FakeThread._deferred.clear()
                    for thread in deferred:
                        thread._run_now()

        def start(self):
            if (
                getattr(self._target, "__name__", None) == "_auto_finalize"
                and _FakeThread._active_replay is not None
            ):
                _FakeThread._deferred.append(self)
                return
            self._run_now()

        def is_alive(self):
            return self._alive

        def join(self, timeout=None):
            return None

    original_thread = telemetry_manager._receiver_thread
    original_backend = telemetry_manager._backend
    original_session_id = telemetry_manager._session_id
    original_start_time = telemetry_manager._start_time
    original_planned_duration = telemetry_manager._planned_duration_s
    original_frame_count = telemetry_manager._frame_count
    original_fault_events = telemetry_manager._fault_events
    original_detecting = telemetry_manager._detecting
    original_active_model = telemetry_manager._active_model
    original_backend_name = telemetry_manager._backend_name
    original_backend_options = telemetry_manager._backend_options
    original_schema_version = telemetry_manager._schema_version
    original_feature_coverage = telemetry_manager._feature_coverage
    original_replay_metrics = telemetry_manager._replay_metrics

    try:
        telemetry_manager._receiver_thread = None
        telemetry_manager._backend = None
        telemetry_manager._session_id = None
        telemetry_manager._start_time = None
        telemetry_manager._planned_duration_s = None
        telemetry_manager._frame_count = 0
        telemetry_manager._fault_events = []
        telemetry_manager._detecting = False
        telemetry_manager._active_model = None
        telemetry_manager._backend_name = None
        telemetry_manager._backend_options = {}
        telemetry_manager._schema_version = "rfly_seq_v1"
        telemetry_manager._feature_coverage = None
        telemetry_manager._replay_metrics = None

        with pytest.MonkeyPatch.context() as mp:
            mp.setattr("uav_dashboard_api.services.telemetry_backends.threading.Thread", _FakeThread)
            mp.setattr("uav_dashboard_api.services.telemetry_manager.threading.Thread", _FakeThread)
            mp.setattr(
                "uav_dashboard_api.services.telemetry_backends.time.sleep",
                lambda *_args, **_kwargs: None,
            )
            mp.setattr(
                "uav_dashboard_api.services.telemetry_manager.time.sleep",
                lambda *_args, **_kwargs: None,
            )

            result = telemetry_manager.start_collection(
                drone_id="DRONE-REPLAY-INT-001",
                session_type="simulation",
                sim_time=3,
                backend="replay",
                replay_path=str(replay_path),
            )

        assert result["scenario_backend"] == "replay"
        assert result["replay_path"] == str(replay_path)

        deadline = time.time() + 1.0
        session = None
        while time.time() < deadline:
            with app.app_context():
                session = db.session.get(TelemetrySession, result["session_id"])
                if session is not None and session.status == "completed":
                    break
            time.sleep(0.01)

        assert session is not None
        assert session.drone_id == "DRONE-REPLAY-INT-001"
        assert session.session_type == "simulation"
        assert session.scenario_backend == "replay"
        assert session.schema_version == "rfly_seq_v1"
        assert session.source_path == str(replay_path)
        assert session.status == "completed"
        assert session.frame_count == 3
        assert session.csv_path == result["csv_path"]
        assert Path(resolve_local_path(session.csv_path, app.config)).exists()
        assert session.duration_seconds is not None
        assert session.duration_seconds >= 0
        assert session.fault_events_count == 0
        assert session.get_meta()["backend"] == "replay"
        assert session.get_meta()["replay_metrics"]["gt_has_labels"] == 1
        assert session.get_meta()["replay_metrics"]["gt_event_count"] == 1
        assert session.get_meta()["replay_metrics"]["pred_event_count"] == 0
    finally:
        telemetry_manager._receiver_thread = original_thread
        telemetry_manager._backend = original_backend
        telemetry_manager._session_id = original_session_id
        telemetry_manager._start_time = original_start_time
        telemetry_manager._planned_duration_s = original_planned_duration
        telemetry_manager._frame_count = original_frame_count
        telemetry_manager._fault_events = original_fault_events
        telemetry_manager._detecting = original_detecting
        telemetry_manager._active_model = original_active_model
        telemetry_manager._backend_name = original_backend_name
        telemetry_manager._backend_options = original_backend_options
        telemetry_manager._schema_version = original_schema_version
        telemetry_manager._feature_coverage = original_feature_coverage
        telemetry_manager._replay_metrics = original_replay_metrics


'''
def test_telemetry_session_detail_returns_404_when_missing(app, client, auth_header):
    response = client.get("/api/telemetry/sessions/TS-NOT-FOUND", headers=auth_header)

    assert response.status_code == 404
    payload = response.get_json()
    assert payload["success"] is False
    assert isinstance(payload["message"], str)
    assert payload["message"]
    assert payload["data"] is None


'''


def test_telemetry_session_detail_returns_404_when_missing(app, client, auth_header):
    response = client.get("/api/telemetry/sessions/TS-NOT-FOUND", headers=auth_header)

    assert response.status_code == 404
    payload = response.get_json()
    assert payload["success"] is False
    assert payload["data"] is None


def test_telemetry_start_collection_replay_success(
    app, client, auth_header, monkeypatch, tmp_path: Path
):
    from uav_dashboard_api.services.telemetry_manager import telemetry_manager

    replay_path = tmp_path / "replay.csv"
    replay_path.write_text("time_s,roll\n0.0,0.1\n", encoding="utf-8")
    start_calls = []

    def fake_start_collection(**kwargs):
        start_calls.append(kwargs)
        return {
            "session_id": "TS-REPLAY-001",
            "collector_status": "running",
            "scenario_backend": "replay",
        }

    monkeypatch.setattr(telemetry_manager, "start_collection", fake_start_collection)

    response = client.post(
        "/api/telemetry/start-collection",
        headers=auth_header,
        json={
            "drone_id": "DRONE-REPLAY-001",
            "session_type": "simulation",
            "sim_time": 45,
            "backend": "replay",
            "replay_path": str(replay_path),
        },
    )

    assert response.status_code == 200
    payload = response.get_json()
    assert payload["success"] is True
    assert isinstance(payload["message"], str)
    assert payload["message"]
    assert payload["data"]["session_id"] == "TS-REPLAY-001"
    assert payload["data"]["scenario_backend"] == "replay"
    assert start_calls == [
        {
            "drone_id": "DRONE-REPLAY-001",
            "session_type": "simulation",
            "sim_time": 45.0,
            "backend": "replay",
            "replay_path": str(replay_path),
            "rfly_options": {"px4mavctrl_path": None},
        }
    ]


def test_telemetry_stop_collection_success(app, client, auth_header, monkeypatch):
    from uav_dashboard_api.services.telemetry_manager import telemetry_manager

    stop_calls = []

    def fake_stop_collection():
        stop_calls.append(True)
        return {
            "session_id": "TS-STOP-001",
            "duration_seconds": 12.3,
            "frame_count": 120,
            "fault_events_count": 1,
            "scenario_backend": "replay",
            "replay_metrics": {"pred_event_count": 1, "hit": 1},
        }

    monkeypatch.setattr(telemetry_manager, "stop_collection", fake_stop_collection)

    response = client.post("/api/telemetry/stop-collection", headers=auth_header)

    assert response.status_code == 200
    payload = response.get_json()
    assert payload["success"] is True
    assert isinstance(payload["message"], str)
    assert payload["message"]
    assert payload["data"]["session_id"] == "TS-STOP-001"
    assert payload["data"]["scenario_backend"] == "replay"
    assert payload["data"]["frame_count"] == 120
    assert payload["data"]["fault_events_count"] == 1
    assert payload["data"]["replay_metrics"]["pred_event_count"] == 1
    assert stop_calls == [True]


def test_telemetry_status_returns_runtime_snapshots_without_leaking_mutable_state(app):
    from uav_dashboard_api.services.telemetry_manager import telemetry_manager

    original_feature_coverage = telemetry_manager._feature_coverage
    original_replay_metrics = telemetry_manager._replay_metrics

    try:
        telemetry_manager._feature_coverage = {
            "mapped": ["roll"],
            "derived": ["yaw"],
            "default": ["pos_z"],
        }
        telemetry_manager._replay_metrics = {"hit": 1, "details": {"latencies": [0.2]}}

        status = telemetry_manager.get_status()
        status["feature_coverage"]["mapped"].append("pitch")
        status["replay_metrics"]["details"]["latencies"].append(0.8)

        latest_status = telemetry_manager.get_status()
        assert latest_status["feature_coverage"]["mapped"] == ["roll"]
        assert latest_status["replay_metrics"]["details"]["latencies"] == [0.2]
    finally:
        telemetry_manager._feature_coverage = original_feature_coverage
        telemetry_manager._replay_metrics = original_replay_metrics


def test_telemetry_status_api_returns_last_frame_and_input_contract(
    app, client, auth_header, monkeypatch
):
    from uav_dashboard_api.services.telemetry_manager import telemetry_manager

    monkeypatch.setattr(
        telemetry_manager,
        "get_status",
        lambda: {
            "is_collecting": True,
            "is_detecting": False,
            "is_blind_test": False,
            "current_session_id": "TS-STATUS-001",
            "uptime_seconds": 8.0,
            "planned_duration_s": 120.0,
            "frame_count": 42,
            "fault_events_count": 0,
            "fault_events": [],
            "latest_diagnoses": [],
            "online_model": None,
            "transport_status": "connected",
            "business_status": "collecting",
            "scenario_backend": "px4",
            "schema_version": "rfly_seq_v1",
            "feature_coverage": {"mapped": ["time_s"], "derived": [], "default": []},
            "last_frame_time_s": 12.34,
            "input_contract_status": {
                "status": "degraded",
                "reason": "missing_features",
            },
            "replay_metrics": None,
            "processes": {
                "collector": "running",
                "receiver": "running",
                "detector": None,
                "px4_sitl": "running",
            },
        },
    )

    response = client.get("/api/telemetry/status", headers=auth_header)

    assert response.status_code == 200
    payload = response.get_json()
    assert payload["success"] is True
    assert payload["data"]["last_frame_time_s"] == 12.34
    assert payload["data"]["input_contract_status"] == {
        "status": "degraded",
        "reason": "missing_features",
    }


def test_wait_for_px4_returns_false_when_socket_never_becomes_ready(monkeypatch):
    from uav_dashboard_api.services.telemetry_manager import telemetry_manager

    original_px4_proc = telemetry_manager._px4_proc

    class _FakeProc:
        def poll(self):
            return None

    timeline = {"now": 0.0}

    def _fake_time():
        return timeline["now"]

    def _fake_sleep(seconds):
        timeline["now"] += seconds

    def _fake_connect(*args, **kwargs):
        raise ConnectionRefusedError

    try:
        telemetry_manager._px4_proc = _FakeProc()
        monkeypatch.setattr("uav_dashboard_api.services.telemetry_manager.time.time", _fake_time)
        monkeypatch.setattr("uav_dashboard_api.services.telemetry_manager.time.sleep", _fake_sleep)
        monkeypatch.setattr(
            "uav_dashboard_api.services.telemetry_manager.socket.create_connection", _fake_connect
        )

        assert telemetry_manager._wait_for_px4(timeout=20) is False
    finally:
        telemetry_manager._px4_proc = original_px4_proc


def test_px4_backend_sets_timeout_reason_when_px4_never_becomes_ready(
    app, monkeypatch, tmp_path: Path
):
    from uav_dashboard_api.services.telemetry_manager import telemetry_manager

    original_state = {
        name: getattr(telemetry_manager, name)
        for name in (
            "_receiver_thread",
            "_backend",
            "_session_id",
            "_start_time",
            "_planned_duration_s",
            "_frame_count",
            "_fault_events",
            "_detecting",
            "_active_model",
            "_backend_name",
            "_backend_options",
            "_schema_version",
            "_feature_coverage",
            "_replay_metrics",
            "_startup_phase",
            "_timeout_reason",
        )
    }

    replay_path = tmp_path / "dummy.csv"
    replay_path.write_text("time_s,roll\n0.0,0.1\n", encoding="utf-8")

    class _FakeThread:
        def __init__(self, target=None, name=None, daemon=None, args=(), kwargs=None):
            self._target = target
            self._args = args
            self._kwargs = kwargs or {}

        def start(self):
            if self._target:
                self._target(*self._args, **self._kwargs)

        def is_alive(self):
            return False

        def join(self, timeout=None):
            return None

    monkeypatch.setattr("uav_dashboard_api.services.telemetry_backends.threading.Thread", _FakeThread)
    monkeypatch.setattr("uav_dashboard_api.services.telemetry_manager.threading.Thread", _FakeThread)
    monkeypatch.setattr(telemetry_manager, "_start_px4", lambda: None)
    monkeypatch.setattr(
        telemetry_manager, "_resolve_telemetry_python_bin", lambda: "/usr/bin/python3"
    )
    monkeypatch.setattr(telemetry_manager, "_wait_for_px4", lambda timeout=90: False)
    monkeypatch.setattr(
        "uav_dashboard_api.services.telemetry_backends.TelemetryReceiver",
        lambda *args, **kwargs: types.SimpleNamespace(callbacks=[]),
    )
    monkeypatch.setattr(telemetry_manager, "_write_log", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        telemetry_manager, "_create_session_record", lambda *args, **kwargs: None
    )

    try:
        result = telemetry_manager.start_collection(
            drone_id="DRONE-PX4-READY-001",
            session_type="simulation",
            sim_time=30,
            backend="px4",
        )
        status = telemetry_manager.get_status()

        assert result["scenario_backend"] == "px4"
        assert status["startup_phase"] == "px4_not_ready"
        assert status["timeout_reason"] == "px4_not_ready"
        assert status["transport_status"] == "degraded"
    finally:
        for name, value in original_state.items():
            setattr(telemetry_manager, name, value)


def test_telemetry_stop_collection_replay_stop_failure_still_clears_runtime_and_finalizes_session(
    app, monkeypatch, tmp_path: Path
):
    from uav_dashboard_api.services.telemetry_manager import telemetry_manager

    replay_path = tmp_path / "replay_stop_fail.csv"
    replay_path.write_text("time_s,roll\n0.0,0.1\n", encoding="utf-8")
    original_receiver_thread = telemetry_manager._receiver_thread
    original_backend = telemetry_manager._backend
    original_session_id = telemetry_manager._session_id
    original_start_time = telemetry_manager._start_time
    original_planned_duration = telemetry_manager._planned_duration_s
    original_frame_count = telemetry_manager._frame_count
    original_fault_events = telemetry_manager._fault_events
    original_detecting = telemetry_manager._detecting
    original_active_model = telemetry_manager._active_model
    original_backend_name = telemetry_manager._backend_name
    original_backend_options = telemetry_manager._backend_options
    original_schema_version = telemetry_manager._schema_version
    original_feature_coverage = telemetry_manager._feature_coverage
    original_replay_metrics = telemetry_manager._replay_metrics

    def fake_start(self):
        class _StoppedThread:
            def is_alive(self):
                return False

        self._thread = _StoppedThread()

    def fail_stop(self):
        raise RuntimeError("replay stop failed")

    monkeypatch.setattr(
        "uav_dashboard_api.services.telemetry_manager.ReplayScenarioBackend.start", fake_start
    )
    monkeypatch.setattr(
        "uav_dashboard_api.services.telemetry_manager.ReplayScenarioBackend.stop", fail_stop
    )

    try:
        result = telemetry_manager.start_collection(
            drone_id="DRONE-REPLAY-STOP-FAIL",
            session_type="simulation",
            sim_time=30,
            backend="replay",
            replay_path=str(replay_path),
        )

        with pytest.raises(RuntimeError, match="replay stop failed"):
            telemetry_manager.stop_collection()

        status = telemetry_manager.get_status()
        assert result["session_id"] == result["session_id"]
        assert status["is_collecting"] is False
        assert status["current_session_id"] is None
        assert status["scenario_backend"] is None
        assert status["planned_duration_s"] is None
        with app.app_context():
            session = db.session.get(TelemetrySession, result["session_id"])
            assert session is not None
            assert session.status == "completed"
    finally:
        telemetry_manager._receiver_thread = original_receiver_thread
        telemetry_manager._backend = original_backend
        telemetry_manager._session_id = original_session_id
        telemetry_manager._start_time = original_start_time
        telemetry_manager._planned_duration_s = original_planned_duration
        telemetry_manager._frame_count = original_frame_count
        telemetry_manager._fault_events = original_fault_events
        telemetry_manager._detecting = original_detecting
        telemetry_manager._active_model = original_active_model
        telemetry_manager._backend_name = original_backend_name
        telemetry_manager._backend_options = original_backend_options
        telemetry_manager._schema_version = original_schema_version
        telemetry_manager._feature_coverage = original_feature_coverage
        telemetry_manager._replay_metrics = original_replay_metrics


def test_telemetry_stop_collection_finalize_race_only_finalizes_once(
    app, monkeypatch, tmp_path: Path
):
    import threading

    from uav_dashboard_api.services.telemetry_manager import telemetry_manager

    replay_path = tmp_path / "replay_finalize_race.csv"
    replay_path.write_text("time_s,roll\n0.0,0.1\n", encoding="utf-8")
    state = {
        name: getattr(telemetry_manager, name)
        for name in (
            "_receiver_thread",
            "_backend",
            "_session_id",
            "_start_time",
            "_planned_duration_s",
            "_frame_count",
            "_fault_events",
            "_detecting",
            "_active_model",
            "_backend_name",
            "_backend_options",
            "_schema_version",
            "_feature_coverage",
            "_replay_metrics",
        )
    }
    entered_stop = threading.Event()
    release_stop = threading.Event()
    finalize_calls, results, errors = [], [], []

    class _StoppedThread:
        def is_alive(self):
            return False

    def fake_backend_stop(self):
        entered_stop.set()
        release_stop.wait(timeout=1)

    def call_stop():
        try:
            results.append(telemetry_manager.stop_collection())
        except Exception as e:
            errors.append(e)

    monkeypatch.setattr(
        "uav_dashboard_api.services.telemetry_manager.ReplayScenarioBackend.start",
        lambda self: setattr(self, "_thread", _StoppedThread()),
    )
    monkeypatch.setattr(
        "uav_dashboard_api.services.telemetry_manager.ReplayScenarioBackend.stop", fake_backend_stop
    )
    monkeypatch.setattr(
        telemetry_manager, "_collect_replay_metrics", lambda session_id: {"hit": 1}
    )
    monkeypatch.setattr(
        telemetry_manager,
        "_finalize_session",
        lambda *args: finalize_calls.append(args),
    )
    monkeypatch.setattr(telemetry_manager, "_write_log", lambda *args: None)

    try:
        telemetry_manager.start_collection(
            drone_id="DRONE-RACE-001",
            session_type="simulation",
            sim_time=30,
            backend="replay",
            replay_path=str(replay_path),
        )
        worker1 = threading.Thread(target=call_stop, daemon=True)
        worker2 = threading.Thread(target=call_stop, daemon=True)
        worker1.start()
        assert entered_stop.wait(timeout=1)
        worker2.start()
        release_stop.set()
        worker1.join(timeout=1)
        worker2.join(timeout=1)
        assert not worker1.is_alive()
        assert not worker2.is_alive()
        assert errors == []
        assert len(results) == 2
        assert results[0]["session_id"] == results[1]["session_id"]
        assert len(finalize_calls) == 1
    finally:
        for name, value in state.items():
            setattr(telemetry_manager, name, value)


def test_telemetry_on_backend_frame_px4_marks_direct_live_fields_as_mapped(app):
    from collections import deque

    from uav_dashboard_api.services.telemetry_manager import telemetry_manager

    state = {
        name: getattr(telemetry_manager, name)
        for name in (
            "_backend_name",
            "_feature_coverage",
            "_realtime_buffer",
            "_frame_count",
        )
    }
    try:
        telemetry_manager._backend_name = "px4"
        telemetry_manager._feature_coverage = None
        telemetry_manager._realtime_buffer = deque(maxlen=8)
        telemetry_manager._frame_count = 0

        frame = {
            "time_s": 1.25,
            "seq": 7,
            "motor1": 1200,
            "motor2": 1190,
            "motor3": 1210,
            "motor4": 1180,
            "baro_temp": 24.5,
            "actuator_ctrl_1": 0.11,
            "actuator_ctrl_2": -0.22,
            "actuator_ctrl_3": 0.33,
            "actuator_ctrl_4": 0.44,
        }
        telemetry_manager._on_backend_frame(frame, deque([frame]), source_backend="px4")

        coverage = telemetry_manager.get_status()["feature_coverage"]
        assert "time_s" in coverage["mapped"]
        assert "seq" in coverage["mapped"]
        assert "baro_temp" in coverage["mapped"]
        assert "actuator_ctrl_1" in coverage["mapped"]
        assert "actuator_ctrl_4" in coverage["mapped"]
        assert "time_s" not in coverage["default"]
        assert "seq" not in coverage["default"]
    finally:
        for name, value in state.items():
            setattr(telemetry_manager, name, value)


def test_telemetry_on_backend_frame_requires_full_window_before_detector_update(app):
    from collections import deque

    from uav_dashboard_api.services.telemetry_manager import telemetry_manager

    class _FakeDetector:
        window_size = 3

        def __init__(self):
            self.calls = []

        def update(self, **kwargs):
            self.calls.append(kwargs)
            return {"fault_prob": 0.9}, []

    state = {
        name: getattr(telemetry_manager, name)
        for name in (
            "_backend_name",
            "_detecting",
            "_detector",
            "_realtime_buffer",
            "_frame_count",
        )
    }
    try:
        detector = _FakeDetector()
        telemetry_manager._backend_name = "replay"
        telemetry_manager._detecting = True
        telemetry_manager._detector = detector
        telemetry_manager._realtime_buffer = deque(maxlen=8)
        telemetry_manager._frame_count = 0

        frame = {
            "time_s": 0.2,
            "roll": 1.2,
            "pitch": 0.3,
            "yaw": -0.1,
            "pos_z": -12.5,
            "motor1": 1200,
            "motor2": 1190,
            "motor3": 1210,
            "motor4": 1180,
        }
        telemetry_manager._on_backend_frame(
            frame, deque([frame, frame]), source_backend="replay"
        )

        assert detector.calls == []
        payload = telemetry_manager.get_realtime_data()
        assert payload["times"] == [0.2]
        assert payload["roll"] == [1.2]
        assert payload["pos_z"] == [-12.5]
    finally:
        for name, value in state.items():
            setattr(telemetry_manager, name, value)


def test_telemetry_on_backend_frame_runs_detector_with_canonical_frame_after_full_window(
    app,
):
    from collections import deque

    from uav_dashboard_api.services.telemetry_manager import telemetry_manager

    class _FakeDetector:
        window_size = 2

        def __init__(self):
            self.calls = []

        def update(self, **kwargs):
            self.calls.append(kwargs)
            return {"fault_prob": 0.97}, []

    state = {
        name: getattr(telemetry_manager, name)
        for name in (
            "_backend_name",
            "_detecting",
            "_detector",
            "_realtime_buffer",
            "_frame_count",
        )
    }
    try:
        detector = _FakeDetector()
        telemetry_manager._backend_name = "px4"
        telemetry_manager._detecting = True
        telemetry_manager._detector = detector
        telemetry_manager._realtime_buffer = deque(maxlen=8)
        telemetry_manager._frame_count = 0

        frame = {
            "time_s": 1.25,
            "seq": 7,
            "motor1": 1200,
            "motor2": 1190,
            "motor3": 1210,
            "motor4": 1180,
            "roll": 1.2,
            "pitch": 0.3,
            "yaw": -0.1,
            "gyro_x": 0.11,
            "gyro_y": 0.22,
            "gyro_z": 0.33,
            "accel_x": 0.5,
            "accel_y": 0.6,
            "accel_z": 9.8,
            "baro_alt_meter": 123.4,
            "baro_temp": 24.5,
            "actuator_ctrl_1": 0.11,
            "actuator_ctrl_2": -0.22,
            "actuator_ctrl_3": 0.33,
            "actuator_ctrl_4": 0.44,
            "vx": 4.5,
            "vy": -0.5,
            "vz": -1.2,
            "pos_z": -12.5,
        }
        telemetry_manager._on_backend_frame(
            frame, deque([frame, frame]), source_backend="px4"
        )

        assert len(detector.calls) == 1
        call = detector.calls[0]
        assert call["frame_idx"] == 1
        assert len(call["window"]) == 2
        assert call["frame"]["time_s"] == 1.25
        assert call["frame"]["seq"] == 7.0
        assert call["frame"]["actuator_out_1"] == 1200.0
        assert call["frame"]["actuator_ctrl_1"] == 0.11
        assert call["frame"]["actuator_ctrl_4"] == 0.44
        assert call["frame"]["baro_alt"] == 123.4
        assert call["frame"]["baro_temp"] == 24.5
        assert call["frame"]["vel_z"] == -1.2
        payload = telemetry_manager.get_realtime_data()
        assert payload["motor1"] == [1200]
        assert payload["yaw"] == [-0.1]
    finally:
        for name, value in state.items():
            setattr(telemetry_manager, name, value)


def test_telemetry_on_backend_frame_replay_updates_feature_coverage(app):
    from collections import deque

    from uav_dashboard_api.services.telemetry_manager import telemetry_manager

    state = {
        name: getattr(telemetry_manager, name)
        for name in (
            "_backend_name",
            "_feature_coverage",
            "_realtime_buffer",
            "_frame_count",
        )
    }
    try:
        telemetry_manager._backend_name = "replay"
        telemetry_manager._feature_coverage = None
        telemetry_manager._realtime_buffer = deque(maxlen=8)
        telemetry_manager._frame_count = 0

        frame = {
            "time_s": 0.2,
            "seq": 3,
            "roll": 1.2,
            "pitch": 0.3,
            "yaw": -0.1,
            "actuator_out_1": 1200,
            "actuator_out_2": 1190,
            "actuator_out_3": 1210,
            "actuator_out_4": 1180,
            "accel_x": 0.5,
            "accel_y": 0.6,
            "accel_z": 9.8,
        }
        telemetry_manager._on_backend_frame(
            frame, deque([frame]), source_backend="replay"
        )

        coverage = telemetry_manager.get_status()["feature_coverage"]
        assert coverage is not None
        assert "time_s" in coverage["mapped"]
        assert "seq" in coverage["mapped"]
        assert "att_q_w" in coverage["derived"]
    finally:
        for name, value in state.items():
            setattr(telemetry_manager, name, value)


def test_telemetry_stop_detection_success(app, client, auth_header, monkeypatch):
    from uav_dashboard_api.services.telemetry_manager import telemetry_manager

    stop_calls = []

    def fake_stop_detection():
        stop_calls.append(True)
        return {
            "detector_status": "stopped",
            "fault_events_count": 2,
            "latest_diagnoses": [
                {"label": "fault", "severity": "heavy"},
                {"label": "fault", "severity": "medium"},
            ],
        }

    monkeypatch.setattr(telemetry_manager, "stop_detection", fake_stop_detection)

    response = client.post("/api/telemetry/stop-detection", headers=auth_header)

    assert response.status_code == 200
    payload = response.get_json()
    assert payload["success"] is True
    assert isinstance(payload["message"], str)
    assert payload["message"]
    assert payload["data"]["detector_status"] == "stopped"
    assert payload["data"]["fault_events_count"] == 2
    assert payload["data"]["latest_diagnoses"][0]["severity"] == "heavy"
    assert stop_calls == [True]


def test_telemetry_stop_detection_replay_exposes_metrics_before_finalize(
    app, monkeypatch, tmp_path: Path
):
    from uav_dashboard_api.services.telemetry_manager import telemetry_manager

    replay_path = tmp_path / "replay_stop_detection_metrics.csv"
    replay_path.write_text(
        "time_s,roll,is_fault_active\n0.0,0.1,0\n0.1,0.2,1\n0.2,0.3,1\n",
        encoding="utf-8",
    )

    class _LiveThread:
        def is_alive(self):
            return True

    class _FakeEvent:
        start_t = 0.1
        end_t = 0.2
        max_fault_prob = 0.97
        mean_fault_prob = 0.91
        n_points = 2

    class _FakeDetector:
        def flush(self):
            return [_FakeEvent()]

    state = {
        name: getattr(telemetry_manager, name)
        for name in (
            "_receiver_thread",
            "_backend",
            "_session_id",
            "_start_time",
            "_planned_duration_s",
            "_frame_count",
            "_fault_events",
            "_detecting",
            "_active_model",
            "_backend_name",
            "_backend_options",
            "_schema_version",
            "_feature_coverage",
            "_replay_metrics",
            "_detector",
            "_realtime_buffer",
        )
    }
    monkeypatch.setattr(
        "uav_dashboard_api.services.telemetry_manager.ReplayScenarioBackend.start",
        lambda self: setattr(self, "_thread", _LiveThread()),
    )

    try:
        result = telemetry_manager.start_collection(
            drone_id="DRONE-REPLAY-METRICS-001",
            session_type="simulation",
            sim_time=30,
            backend="replay",
            replay_path=str(replay_path),
        )
        Path(resolve_local_path(result["csv_path"], app.config)).write_text(
            replay_path.read_text(encoding="utf-8"), encoding="utf-8"
        )
        telemetry_manager._detector = _FakeDetector()
        telemetry_manager._detecting = True
        telemetry_manager._active_model = {"name": "online_detector_alpha"}

        stop_result = telemetry_manager.stop_detection()
        status = telemetry_manager.get_status()

        assert stop_result["fault_events_count"] == 1
        assert stop_result["replay_metrics"]["pred_event_count"] == 1
        assert stop_result["replay_metrics"]["hit"] == 1
        assert stop_result["replay_metrics"]["detection_latency_s"] == pytest.approx(
            0.0
        )
        assert status["is_collecting"] is True
        assert status["is_detecting"] is False
        assert status["replay_metrics"]["hit"] == 1
        assert status["latest_diagnoses"][0]["label"] == "fault"
    finally:
        for name, value in state.items():
            setattr(telemetry_manager, name, value)


def test_telemetry_stop_collection_flushes_detection_events_into_finalize(
    app, monkeypatch, tmp_path: Path
):
    from uav_dashboard_api.services.telemetry_manager import telemetry_manager

    replay_path = tmp_path / "replay_stop_detection.csv"
    replay_path.write_text(
        "time_s,roll,is_fault_active\n0.0,0.1,0\n0.1,0.2,1\n0.2,0.3,1\n",
        encoding="utf-8",
    )

    class _StoppedThread:
        def is_alive(self):
            return False

    class _FakeEvent:
        start_t = 0.1
        end_t = 0.2
        max_fault_prob = 0.97
        mean_fault_prob = 0.91
        n_points = 2

    class _FakeDetector:
        window_size = 150

        def flush(self):
            return [_FakeEvent()]

    original_receiver_thread = telemetry_manager._receiver_thread
    original_backend = telemetry_manager._backend
    original_session_id = telemetry_manager._session_id
    original_start_time = telemetry_manager._start_time
    original_planned_duration = telemetry_manager._planned_duration_s
    original_frame_count = telemetry_manager._frame_count
    original_fault_events = telemetry_manager._fault_events
    original_detecting = telemetry_manager._detecting
    original_active_model = telemetry_manager._active_model
    original_backend_name = telemetry_manager._backend_name
    original_backend_options = telemetry_manager._backend_options
    original_schema_version = telemetry_manager._schema_version
    original_feature_coverage = telemetry_manager._feature_coverage
    original_replay_metrics = telemetry_manager._replay_metrics
    original_detector = telemetry_manager._detector

    def fake_start(self):
        self._thread = _StoppedThread()

    monkeypatch.setattr(
        "uav_dashboard_api.services.telemetry_manager.ReplayScenarioBackend.start", fake_start
    )

    try:
        result = telemetry_manager.start_collection(
            drone_id="DRONE-REPLAY-FLUSH-001",
            session_type="simulation",
            sim_time=30,
            backend="replay",
            replay_path=str(replay_path),
        )
        Path(resolve_local_path(result["csv_path"], app.config)).write_text(
            replay_path.read_text(encoding="utf-8"), encoding="utf-8"
        )
        telemetry_manager._detector = _FakeDetector()
        telemetry_manager._detecting = True
        telemetry_manager._active_model = {"name": "online_detector_alpha"}

        stop_result = telemetry_manager.stop_collection()

        assert stop_result["session_id"] == result["session_id"]
        assert stop_result["fault_events_count"] == 1
        with app.app_context():
            session = db.session.get(TelemetrySession, result["session_id"])
            assert session is not None
            assert session.status == "completed"
            assert session.fault_events_count == 1
            events = TelemetryFaultEvent.query.filter_by(
                session_id=result["session_id"]
            ).all()
            assert len(events) == 1
            assert events[0].max_fault_prob == pytest.approx(0.97)
            assert events[0].get_diagnosis()["label"] == "fault"
            replay_metrics = session.get_meta()["replay_metrics"]
            assert replay_metrics["pred_event_count"] == 1
            assert replay_metrics["hit"] == 1
    finally:
        telemetry_manager._receiver_thread = original_receiver_thread
        telemetry_manager._backend = original_backend
        telemetry_manager._session_id = original_session_id
        telemetry_manager._start_time = original_start_time
        telemetry_manager._planned_duration_s = original_planned_duration
        telemetry_manager._frame_count = original_frame_count
        telemetry_manager._fault_events = original_fault_events
        telemetry_manager._detecting = original_detecting
        telemetry_manager._active_model = original_active_model
        telemetry_manager._backend_name = original_backend_name
        telemetry_manager._backend_options = original_backend_options
        telemetry_manager._schema_version = original_schema_version
        telemetry_manager._feature_coverage = original_feature_coverage
        telemetry_manager._replay_metrics = original_replay_metrics
        telemetry_manager._detector = original_detector


def test_telemetry_record_fault_event_delegates_persistence_writes(app, monkeypatch):
    from uav_dashboard_api.services.telemetry_manager import telemetry_manager

    class _FakeEvent:
        start_t = 0.1
        end_t = 0.2
        max_fault_prob = 0.97
        mean_fault_prob = 0.91
        n_points = 2

    original_session_id = telemetry_manager._session_id
    original_fault_events = telemetry_manager._fault_events

    captured = {}

    try:
        telemetry_manager._session_id = "TS-PERSIST-001"
        telemetry_manager._fault_events = []

        def fake_record_fault_event(*, session_id, event, diagnosis, legacy_severity):
            captured["session_id"] = session_id
            captured["event"] = event
            captured["diagnosis"] = diagnosis
            captured["legacy_severity"] = legacy_severity

        monkeypatch.setattr(
            telemetry_manager._persistence,
            "record_fault_event",
            fake_record_fault_event,
        )
        telemetry_manager._record_fault_event(_FakeEvent())

        assert len(telemetry_manager._fault_events) == 1
        assert captured["session_id"] == "TS-PERSIST-001"
        assert captured["event"].max_fault_prob == pytest.approx(0.97)
        assert captured["diagnosis"]["label"] == "fault"
        assert captured["diagnosis"]["severity"] == "heavy"
        assert captured["legacy_severity"] == "critical"
    finally:
        telemetry_manager._session_id = original_session_id
        telemetry_manager._fault_events = original_fault_events


def test_telemetry_finalize_session_delegates_persistence_writes(app, monkeypatch):
    from uav_dashboard_api.services.telemetry_manager import telemetry_manager

    captured = {}
    replay_metrics = {"pred_event_count": 1, "hit": 1}

    def fake_finalize_session(
        *, session_id, duration, frame_count, fault_count, replay_metrics
    ):
        captured["session_id"] = session_id
        captured["duration"] = duration
        captured["frame_count"] = frame_count
        captured["fault_count"] = fault_count
        captured["replay_metrics"] = replay_metrics

    monkeypatch.setattr(
        telemetry_manager._persistence, "finalize_session", fake_finalize_session
    )
    telemetry_manager._finalize_session("TS-PERSIST-002", 12.3, 45, 2, replay_metrics)

    assert captured == {
        "session_id": "TS-PERSIST-002",
        "duration": 12.3,
        "frame_count": 45,
        "fault_count": 2,
        "replay_metrics": replay_metrics,
    }


def test_telemetry_create_session_record_delegates_persistence_writes(app, monkeypatch):
    from uav_dashboard_api.services.telemetry_manager import telemetry_manager

    captured = {}
    meta = {"ground_truth": {"fault_type": "none"}}

    def fake_create_session_record(
        *,
        session_id,
        drone_id,
        session_type,
        csv_path,
        backend_name,
        schema_version,
        source_path,
        meta,
    ):
        captured["session_id"] = session_id
        captured["drone_id"] = drone_id
        captured["session_type"] = session_type
        captured["csv_path"] = csv_path
        captured["backend_name"] = backend_name
        captured["schema_version"] = schema_version
        captured["source_path"] = source_path
        captured["meta"] = meta

    original_schema_version = telemetry_manager._schema_version
    try:
        telemetry_manager._schema_version = "canonical-v-test"
        monkeypatch.setattr(
            telemetry_manager._persistence,
            "create_session_record",
            fake_create_session_record,
        )
        telemetry_manager._create_session_record(
            "TS-PERSIST-003",
            "DRONE-PERSIST-003",
            "simulation",
            "/tmp/session.csv",
            "replay",
            source_path="/tmp/replay.csv",
            meta=meta,
        )
    finally:
        telemetry_manager._schema_version = original_schema_version

    assert captured == {
        "session_id": "TS-PERSIST-003",
        "drone_id": "DRONE-PERSIST-003",
        "session_type": "simulation",
        "csv_path": "/tmp/session.csv",
        "backend_name": "replay",
        "schema_version": "canonical-v-test",
        "source_path": "/tmp/replay.csv",
        "meta": meta,
    }


def test_telemetry_delete_session_record_delegates_persistence_writes(app, monkeypatch):
    from uav_dashboard_api.services.telemetry_manager import telemetry_manager

    captured = {}

    def fake_delete_session_record(*, session_id):
        captured["session_id"] = session_id

    monkeypatch.setattr(
        telemetry_manager._persistence,
        "delete_session_record",
        fake_delete_session_record,
    )
    telemetry_manager._delete_session_record("TS-PERSIST-004")

    assert captured == {"session_id": "TS-PERSIST-004"}


def test_telemetry_update_session_detection_delegates_persistence_writes(
    app, monkeypatch
):
    from uav_dashboard_api.services.telemetry_manager import telemetry_manager

    captured = {}

    def fake_update_session_detection(*, session_id, has_detection):
        captured["session_id"] = session_id
        captured["has_detection"] = has_detection

    monkeypatch.setattr(
        telemetry_manager._persistence,
        "update_session_detection",
        fake_update_session_detection,
    )
    telemetry_manager._update_session_detection("TS-PERSIST-005", True)

    assert captured == {"session_id": "TS-PERSIST-005", "has_detection": True}


def test_telemetry_write_log_delegates_persistence_writes(app, monkeypatch):
    from uav_dashboard_api.services.telemetry_manager import telemetry_manager

    captured = {}

    def fake_write_log(*, drone_id, log_type, content):
        captured["drone_id"] = drone_id
        captured["log_type"] = log_type
        captured["content"] = content

    monkeypatch.setattr(telemetry_manager._persistence, "write_log", fake_write_log)
    telemetry_manager._write_log("DRONE-PERSIST-006", "telemetry", "hello")

    assert captured == {
        "drone_id": "DRONE-PERSIST-006",
        "log_type": "telemetry",
        "content": "hello",
    }


def test_telemetry_stop_collection_clears_runtime_status_after_finalize(
    app, monkeypatch, tmp_path: Path
):
    from uav_dashboard_api.services.telemetry_manager import telemetry_manager

    replay_path = tmp_path / "replay_stop_status_clear.csv"
    replay_path.write_text(
        "time_s,roll,is_fault_active\n0.0,0.1,0\n0.1,0.2,1\n0.2,0.3,1\n",
        encoding="utf-8",
    )

    class _StoppedThread:
        def is_alive(self):
            return False

    class _FakeEvent:
        start_t = 0.1
        end_t = 0.2
        max_fault_prob = 0.97
        mean_fault_prob = 0.91
        n_points = 2

    class _FakeDetector:
        window_size = 150

        def flush(self):
            return [_FakeEvent()]

    original_receiver_thread = telemetry_manager._receiver_thread
    original_backend = telemetry_manager._backend
    original_session_id = telemetry_manager._session_id
    original_start_time = telemetry_manager._start_time
    original_planned_duration = telemetry_manager._planned_duration_s
    original_frame_count = telemetry_manager._frame_count
    original_fault_events = telemetry_manager._fault_events
    original_detecting = telemetry_manager._detecting
    original_active_model = telemetry_manager._active_model
    original_backend_name = telemetry_manager._backend_name
    original_backend_options = telemetry_manager._backend_options
    original_schema_version = telemetry_manager._schema_version
    original_feature_coverage = telemetry_manager._feature_coverage
    original_replay_metrics = telemetry_manager._replay_metrics
    original_detector = telemetry_manager._detector

    def fake_start(self):
        self._thread = _StoppedThread()

    monkeypatch.setattr(
        "uav_dashboard_api.services.telemetry_manager.ReplayScenarioBackend.start", fake_start
    )

    try:
        result = telemetry_manager.start_collection(
            drone_id="DRONE-REPLAY-STATUS-CLEAR-001",
            session_type="simulation",
            sim_time=30,
            backend="replay",
            replay_path=str(replay_path),
        )
        Path(resolve_local_path(result["csv_path"], app.config)).write_text(
            replay_path.read_text(encoding="utf-8"), encoding="utf-8"
        )
        telemetry_manager._detector = _FakeDetector()
        telemetry_manager._detecting = True
        telemetry_manager._active_model = {"name": "online_detector_alpha"}

        telemetry_manager.stop_collection()

        status = telemetry_manager.get_status()
        assert status["is_collecting"] is False
        assert status["is_detecting"] is False
        assert status["current_session_id"] is None
        assert status["planned_duration_s"] is None
        assert status["frame_count"] == 0
        assert status["fault_events_count"] == 0
        assert status["fault_events"] == []
        assert status["latest_diagnoses"] == []
        assert status["scenario_backend"] is None
        assert status["schema_version"] is None
        assert status["feature_coverage"] is None
        assert status["replay_metrics"] is None
    finally:
        telemetry_manager._receiver_thread = original_receiver_thread
        telemetry_manager._backend = original_backend
        telemetry_manager._session_id = original_session_id
        telemetry_manager._start_time = original_start_time
        telemetry_manager._planned_duration_s = original_planned_duration
        telemetry_manager._frame_count = original_frame_count
        telemetry_manager._fault_events = original_fault_events
        telemetry_manager._detecting = original_detecting
        telemetry_manager._active_model = original_active_model
        telemetry_manager._backend_name = original_backend_name
        telemetry_manager._backend_options = original_backend_options
        telemetry_manager._schema_version = original_schema_version
        telemetry_manager._feature_coverage = original_feature_coverage
        telemetry_manager._replay_metrics = original_replay_metrics
        telemetry_manager._detector = original_detector


def test_telemetry_stop_collection_clears_realtime_buffer_after_finalize(
    app, monkeypatch, tmp_path: Path
):
    from uav_dashboard_api.services.telemetry_manager import telemetry_manager

    replay_path = tmp_path / "replay_stop_buffer_clear.csv"
    replay_path.write_text(
        "time_s,roll,is_fault_active\n0.0,0.1,0\n0.1,0.2,1\n",
        encoding="utf-8",
    )

    class _StoppedThread:
        def is_alive(self):
            return False

    original_receiver_thread = telemetry_manager._receiver_thread
    original_backend = telemetry_manager._backend
    original_session_id = telemetry_manager._session_id
    original_start_time = telemetry_manager._start_time
    original_planned_duration = telemetry_manager._planned_duration_s
    original_frame_count = telemetry_manager._frame_count
    original_fault_events = telemetry_manager._fault_events
    original_detecting = telemetry_manager._detecting
    original_active_model = telemetry_manager._active_model
    original_backend_name = telemetry_manager._backend_name
    original_backend_options = telemetry_manager._backend_options
    original_schema_version = telemetry_manager._schema_version
    original_feature_coverage = telemetry_manager._feature_coverage
    original_replay_metrics = telemetry_manager._replay_metrics
    original_realtime_buffer = telemetry_manager._realtime_buffer

    def fake_start(self):
        self._thread = _StoppedThread()

    monkeypatch.setattr(
        "uav_dashboard_api.services.telemetry_manager.ReplayScenarioBackend.start", fake_start
    )

    try:
        telemetry_manager._realtime_buffer.append(
            {
                "time_s": 9.9,
                "motor1": 1.0,
                "motor2": 2.0,
                "motor3": 3.0,
                "motor4": 4.0,
                "roll": 0.1,
                "pitch": 0.2,
                "yaw": 0.3,
                "pos_z": -1.0,
            }
        )
        result = telemetry_manager.start_collection(
            drone_id="DRONE-REPLAY-BUFFER-CLEAR-001",
            session_type="simulation",
            sim_time=30,
            backend="replay",
            replay_path=str(replay_path),
        )
        Path(resolve_local_path(result["csv_path"], app.config)).write_text(
            replay_path.read_text(encoding="utf-8"), encoding="utf-8"
        )
        telemetry_manager._realtime_buffer.append(
            {
                "time_s": 0.1,
                "motor1": 10.0,
                "motor2": 20.0,
                "motor3": 30.0,
                "motor4": 40.0,
                "roll": 0.01,
                "pitch": 0.02,
                "yaw": 0.03,
                "pos_z": -2.0,
            }
        )

        telemetry_manager.stop_collection()

        payload = telemetry_manager.get_realtime_data()
        assert payload == {
            "times": [],
            "motor1": [],
            "motor2": [],
            "motor3": [],
            "motor4": [],
            "roll": [],
            "pitch": [],
            "yaw": [],
            "pos_x": [],
            "pos_y": [],
            "pos_z": [],
            "fault_prob": [],
        }
    finally:
        telemetry_manager._receiver_thread = original_receiver_thread
        telemetry_manager._backend = original_backend
        telemetry_manager._session_id = original_session_id
        telemetry_manager._start_time = original_start_time
        telemetry_manager._planned_duration_s = original_planned_duration
        telemetry_manager._frame_count = original_frame_count
        telemetry_manager._fault_events = original_fault_events
        telemetry_manager._detecting = original_detecting
        telemetry_manager._active_model = original_active_model
        telemetry_manager._backend_name = original_backend_name
        telemetry_manager._backend_options = original_backend_options
        telemetry_manager._schema_version = original_schema_version
        telemetry_manager._feature_coverage = original_feature_coverage
        telemetry_manager._replay_metrics = original_replay_metrics
        telemetry_manager._realtime_buffer = original_realtime_buffer


def test_telemetry_blind_test_start_success(app, client, auth_header, monkeypatch):
    from uav_dashboard_api.services.telemetry_manager import telemetry_manager

    start_calls = []

    def fake_start_blind_test(**kwargs):
        start_calls.append(kwargs)
        return {
            "session_id": "TS-BLIND-001",
            "collector_status": "running",
            "blind_test": True,
        }

    monkeypatch.setattr(telemetry_manager, "start_blind_test", fake_start_blind_test)

    response = client.post(
        "/api/telemetry/blind-test/start",
        headers=auth_header,
        json={"drone_id": "DRONE-BLIND-001", "sim_time": 90},
    )

    assert response.status_code == 200
    payload = response.get_json()
    assert payload["success"] is True
    assert isinstance(payload["message"], str)
    assert payload["message"]
    assert payload["data"]["session_id"] == "TS-BLIND-001"
    assert payload["data"]["collector_status"] == "running"
    assert payload["data"]["blind_test"] is True
    assert start_calls == [{"drone_id": "DRONE-BLIND-001", "sim_time": 90.0}]


def test_telemetry_blind_test_start_validation_errors(
    app, client, auth_header, monkeypatch
):
    from uav_dashboard_api.services.telemetry_manager import telemetry_manager

    start_calls = []

    monkeypatch.setattr(
        telemetry_manager,
        "start_blind_test",
        lambda **kwargs: start_calls.append(kwargs) or {"session_id": "TS-BLIND-001"},
    )

    invalid_sim_time = client.post(
        "/api/telemetry/blind-test/start",
        headers=auth_header,
        json={"drone_id": "DRONE-BLIND-001", "sim_time": "abc"},
    )
    assert invalid_sim_time.status_code == 400
    invalid_payload = invalid_sim_time.get_json()
    assert invalid_payload["success"] is False
    assert invalid_payload["message"]

    non_positive_sim_time = client.post(
        "/api/telemetry/blind-test/start",
        headers=auth_header,
        json={"drone_id": "DRONE-BLIND-001", "sim_time": 0},
    )
    assert non_positive_sim_time.status_code == 400
    non_positive_payload = non_positive_sim_time.get_json()
    assert non_positive_payload["success"] is False
    assert non_positive_payload["message"]
    invalid_drone_id = client.post(
        "/api/telemetry/blind-test/start",
        headers=auth_header,
        json={"drone_id": {"id": "DRONE-BLIND-001"}, "sim_time": 90},
    )
    assert invalid_drone_id.status_code == 400
    payload = invalid_drone_id.get_json()
    assert payload["success"] is False
    assert payload["message"]
    assert payload["data"] is None
    assert start_calls == []


def test_telemetry_start_collection_validation_errors(
    app, client, auth_header, monkeypatch
):
    from uav_dashboard_api.services.telemetry_manager import telemetry_manager

    monkeypatch.setattr(
        telemetry_manager,
        "start_collection",
        lambda **kwargs: (_ for _ in ()).throw(
            AssertionError("start_collection should not be called")
        ),
    )

    invalid_sim_time = client.post(
        "/api/telemetry/start-collection",
        headers=auth_header,
        json={"sim_time": "abc", "backend": "px4"},
    )
    assert invalid_sim_time.status_code == 400
    invalid_payload = invalid_sim_time.get_json()
    assert invalid_payload["success"] is False
    assert invalid_payload["message"]

    non_positive_sim_time = client.post(
        "/api/telemetry/start-collection",
        headers=auth_header,
        json={"sim_time": 0, "backend": "px4"},
    )
    non_positive_payload = non_positive_sim_time.get_json()
    assert non_positive_payload["success"] is False
    assert non_positive_payload["message"]
    invalid_backend = client.post(
        "/api/telemetry/start-collection",
        headers=auth_header,
        json={"sim_time": 60, "backend": "invalid"},
    )
    invalid_backend_payload = invalid_backend.get_json()
    assert invalid_backend_payload["success"] is False
    assert invalid_backend_payload["message"]
    missing_replay_path = client.post(
        "/api/telemetry/start-collection",
        headers=auth_header,
        json={"sim_time": 60, "backend": "replay", "replay_path": "   "},
    )
    assert missing_replay_path.status_code == 400
    missing_replay_payload = missing_replay_path.get_json()
    assert missing_replay_payload["success"] is False
    assert missing_replay_payload["message"]


def test_telemetry_start_collection_replay_start_failure_rolls_back_runtime_and_session(
    app, monkeypatch, tmp_path: Path
):
    from uav_dashboard_api.services.telemetry_manager import telemetry_manager

    replay_path = tmp_path / "replay_start_fail.csv"
    replay_path.write_text("time_s,roll\n0.0,0.1\n", encoding="utf-8")
    original_backend = telemetry_manager._backend
    original_session_id = telemetry_manager._session_id
    original_start_time = telemetry_manager._start_time
    original_planned_duration = telemetry_manager._planned_duration_s
    original_frame_count = telemetry_manager._frame_count
    original_fault_events = telemetry_manager._fault_events
    original_detecting = telemetry_manager._detecting
    original_active_model = telemetry_manager._active_model
    original_backend_name = telemetry_manager._backend_name
    original_backend_options = telemetry_manager._backend_options
    original_schema_version = telemetry_manager._schema_version
    original_feature_coverage = telemetry_manager._feature_coverage
    original_replay_metrics = telemetry_manager._replay_metrics

    def fail_start(self):
        raise RuntimeError("replay start failed")

    monkeypatch.setattr(
        "uav_dashboard_api.services.telemetry_manager.ReplayScenarioBackend.start", fail_start
    )

    try:
        with pytest.raises(RuntimeError, match="replay start failed"):
            telemetry_manager.start_collection(
                drone_id="DRONE-REPLAY-START-FAIL",
                session_type="simulation",
                sim_time=30,
                backend="replay",
                replay_path=str(replay_path),
            )

        status = telemetry_manager.get_status()
        assert status["is_collecting"] is False
        assert status["current_session_id"] is None
        assert status["scenario_backend"] is None
        assert status["planned_duration_s"] is None
        assert status["schema_version"] is None
        assert status["feature_coverage"] is None
        with app.app_context():
            assert TelemetrySession.query.count() == 0
    finally:
        telemetry_manager._backend = original_backend
        telemetry_manager._session_id = original_session_id
        telemetry_manager._start_time = original_start_time
        telemetry_manager._planned_duration_s = original_planned_duration
        telemetry_manager._frame_count = original_frame_count
        telemetry_manager._fault_events = original_fault_events
        telemetry_manager._detecting = original_detecting
        telemetry_manager._active_model = original_active_model
        telemetry_manager._backend_name = original_backend_name
        telemetry_manager._backend_options = original_backend_options
        telemetry_manager._schema_version = original_schema_version
        telemetry_manager._feature_coverage = original_feature_coverage
        telemetry_manager._replay_metrics = original_replay_metrics


def test_telemetry_start_collection_replay_missing_source_does_not_leave_session_state(
    app, client, auth_header, tmp_path: Path
):
    from uav_dashboard_api.services.telemetry_manager import telemetry_manager

    missing_replay_path = tmp_path / "missing_replay.csv"
    original_backend = telemetry_manager._backend
    original_session_id = telemetry_manager._session_id
    original_start_time = telemetry_manager._start_time
    original_planned_duration = telemetry_manager._planned_duration_s
    original_frame_count = telemetry_manager._frame_count
    original_fault_events = telemetry_manager._fault_events
    original_detecting = telemetry_manager._detecting
    original_active_model = telemetry_manager._active_model
    original_backend_name = telemetry_manager._backend_name
    original_backend_options = telemetry_manager._backend_options
    original_schema_version = telemetry_manager._schema_version
    original_feature_coverage = telemetry_manager._feature_coverage
    original_replay_metrics = telemetry_manager._replay_metrics

    try:
        response = client.post(
            "/api/telemetry/start-collection",
            headers=auth_header,
            json={
                "drone_id": "DRONE-REPLAY-404",
                "sim_time": 30,
                "backend": "replay",
                "replay_path": str(missing_replay_path),
            },
        )

        assert response.status_code == 400
        assert response.get_json()["message"] == f"replay 源不存在: {missing_replay_path}"
        status = telemetry_manager.get_status()
        assert status["is_collecting"] is False
        assert status["current_session_id"] is None
        assert status["scenario_backend"] is None
        assert status["schema_version"] is None
        assert status["feature_coverage"] is None
        with app.app_context():
            assert TelemetrySession.query.count() == 0
    finally:
        telemetry_manager._backend = original_backend
        telemetry_manager._session_id = original_session_id
        telemetry_manager._start_time = original_start_time
        telemetry_manager._planned_duration_s = original_planned_duration
        telemetry_manager._frame_count = original_frame_count
        telemetry_manager._fault_events = original_fault_events
        telemetry_manager._detecting = original_detecting
        telemetry_manager._active_model = original_active_model
        telemetry_manager._backend_name = original_backend_name
        telemetry_manager._backend_options = original_backend_options
        telemetry_manager._schema_version = original_schema_version
        telemetry_manager._feature_coverage = original_feature_coverage
        telemetry_manager._replay_metrics = original_replay_metrics


def test_telemetry_start_collection_replay_runtime_prefixed_path_is_resolved(
    app, client, auth_header, monkeypatch
):
    from uav_dashboard_api.services.telemetry_manager import telemetry_manager

    runtime_csv_dir = Path(app.config["RUNTIME_DIR"]) / "uploads" / "csv"
    runtime_csv_dir.mkdir(parents=True, exist_ok=True)
    replay_file = runtime_csv_dir / "runtime_prefixed.csv"
    replay_file.write_text("time_s,roll\n0.0,0.1\n", encoding="utf-8")
    replay_path = f"runtime://uploads/csv/{replay_file.name}"

    original_backend = telemetry_manager._backend
    original_session_id = telemetry_manager._session_id
    original_start_time = telemetry_manager._start_time
    original_planned_duration = telemetry_manager._planned_duration_s
    original_frame_count = telemetry_manager._frame_count
    original_fault_events = telemetry_manager._fault_events
    original_detecting = telemetry_manager._detecting
    original_active_model = telemetry_manager._active_model
    original_backend_name = telemetry_manager._backend_name
    original_backend_options = telemetry_manager._backend_options
    original_schema_version = telemetry_manager._schema_version
    original_feature_coverage = telemetry_manager._feature_coverage
    original_replay_metrics = telemetry_manager._replay_metrics

    def fail_start(self):
        raise RuntimeError("replay start failed")

    monkeypatch.setattr(
        "uav_dashboard_api.services.telemetry_manager.ReplayScenarioBackend.start", fail_start
    )

    try:
        response = client.post(
            "/api/telemetry/start-collection",
            headers=auth_header,
            json={
                "drone_id": "DRONE-REPLAY-RUNTIME-PATH",
                "sim_time": 30,
                "backend": "replay",
                "replay_path": replay_path,
            },
        )

        assert response.status_code == 400
        assert response.get_json()["message"] == "replay start failed"
        status = telemetry_manager.get_status()
        assert status["is_collecting"] is False
        assert status["current_session_id"] is None
        assert status["scenario_backend"] is None
        with app.app_context():
            assert TelemetrySession.query.count() == 0
    finally:
        telemetry_manager._backend = original_backend
        telemetry_manager._session_id = original_session_id
        telemetry_manager._start_time = original_start_time
        telemetry_manager._planned_duration_s = original_planned_duration
        telemetry_manager._frame_count = original_frame_count
        telemetry_manager._fault_events = original_fault_events
        telemetry_manager._detecting = original_detecting
        telemetry_manager._active_model = original_active_model
        telemetry_manager._backend_name = original_backend_name
        telemetry_manager._backend_options = original_backend_options
        telemetry_manager._schema_version = original_schema_version
        telemetry_manager._feature_coverage = original_feature_coverage
        telemetry_manager._replay_metrics = original_replay_metrics


def test_telemetry_blind_test_start_collection_failure_rolls_back_blind_runtime_state(
    app, monkeypatch
):
    from uav_dashboard_api.services.telemetry_manager import telemetry_manager

    original_blind_test_active = telemetry_manager._blind_test_active
    original_blind_test_ground_truth = telemetry_manager._blind_test_ground_truth

    monkeypatch.setattr(
        telemetry_manager,
        "start_collection",
        lambda **kwargs: (_ for _ in ()).throw(RuntimeError("blind start failed")),
    )

    try:
        telemetry_manager._blind_test_active = False
        telemetry_manager._blind_test_ground_truth = None

        with pytest.raises(RuntimeError, match="blind start failed"):
            telemetry_manager.start_blind_test(drone_id="DRONE-BLIND-FAIL", sim_time=30)

        assert telemetry_manager._blind_test_active is False
        assert telemetry_manager._blind_test_ground_truth is None
    finally:
        telemetry_manager._blind_test_active = original_blind_test_active
        telemetry_manager._blind_test_ground_truth = original_blind_test_ground_truth


def test_telemetry_blind_test_reveal_success(app, client, auth_header, monkeypatch):
    from uav_dashboard_api.services.telemetry_manager import telemetry_manager

    reveal_calls = []

    def fake_get_blind_test_reveal():
        reveal_calls.append(True)
        return {
            "session_id": "TS-BLIND-001",
            "fault_type": "motor",
            "fault_injected": True,
            "reveal_status": "available",
        }

    monkeypatch.setattr(
        telemetry_manager, "get_blind_test_reveal", fake_get_blind_test_reveal
    )

    response = client.get("/api/telemetry/blind-test/reveal", headers=auth_header)

    assert response.status_code == 200
    payload = response.get_json()
    assert payload["success"] is True
    assert isinstance(payload["message"], str)
    assert payload["message"]
    assert payload["data"]["session_id"] == "TS-BLIND-001"
    assert payload["data"]["fault_type"] == "motor"
    assert payload["data"]["fault_injected"] is True
    assert payload["data"]["reveal_status"] == "available"
    assert reveal_calls == [True]


def test_telemetry_blind_test_reveal_runtime_error_returns_400(
    app, client, auth_header, monkeypatch
):
    from uav_dashboard_api.services.telemetry_manager import telemetry_manager

    monkeypatch.setattr(
        telemetry_manager,
        "get_blind_test_reveal",
        lambda: (_ for _ in ()).throw(RuntimeError("reveal failed")),
    )

    response = client.get("/api/telemetry/blind-test/reveal", headers=auth_header)

    assert response.status_code == 400
    payload = response.get_json()
    assert payload["success"] is False
    assert payload["message"] == "reveal failed"
    assert payload["data"] is None


def test_telemetry_start_collection_runtime_error_returns_error_contract(
    app, client, auth_header, monkeypatch
):
    from uav_dashboard_api.services.telemetry_manager import telemetry_manager

    monkeypatch.setattr(
        telemetry_manager,
        "start_collection",
        lambda **kwargs: (_ for _ in ()).throw(
            RuntimeError("start collection failed contract")
        ),
    )

    response = client.post(
        "/api/telemetry/start-collection",
        headers=auth_header,
        json={"sim_time": 60, "backend": "px4"},
    )

    assert response.status_code == 400
    payload = response.get_json()
    assert payload["success"] is False
    assert payload["message"] == "start collection failed contract"
    assert payload["data"] is None


def test_telemetry_start_detection_validation_errors(
    app, client, auth_header, monkeypatch
):
    from uav_dashboard_api.services.telemetry_manager import telemetry_manager

    start_calls = []

    monkeypatch.setattr(
        telemetry_manager,
        "start_detection",
        lambda requested_model=None: (
            start_calls.append(requested_model) or {"detector_status": "running"}
        ),
    )

    invalid_model = client.post(
        "/api/telemetry/start-detection",
        headers=auth_header,
        json={"model": {"name": "online_detector_alpha"}},
    )

    assert invalid_model.status_code == 400
    payload = invalid_model.get_json()
    assert payload["success"] is False
    assert payload["message"]
    assert payload["data"] is None
    assert start_calls == []


def test_telemetry_start_detection_runtime_error_returns_error_contract(
    app, client, auth_header, monkeypatch
):
    from uav_dashboard_api.services.telemetry_manager import telemetry_manager

    monkeypatch.setattr(
        telemetry_manager,
        "start_detection",
        lambda requested_model=None: (_ for _ in ()).throw(
            RuntimeError("start detection failed contract")
        ),
    )

    response = client.post(
        "/api/telemetry/start-detection",
        headers=auth_header,
        json={"model": "online_detector_alpha"},
    )

    assert response.status_code == 400
    payload = response.get_json()
    assert payload["success"] is False
    assert payload["message"] == "start detection failed contract"
    assert payload["data"] is None


def test_telemetry_blind_test_start_runtime_error_returns_error_contract(
    app, client, auth_header, monkeypatch
):
    from uav_dashboard_api.services.telemetry_manager import telemetry_manager

    monkeypatch.setattr(
        telemetry_manager,
        "start_blind_test",
        lambda **kwargs: (_ for _ in ()).throw(
            RuntimeError("start blind failed contract")
        ),
    )

    response = client.post(
        "/api/telemetry/blind-test/start",
        headers=auth_header,
        json={"drone_id": "DRONE-BLIND-001", "sim_time": 90},
    )

    assert response.status_code == 400
    payload = response.get_json()
    assert payload["success"] is False
    assert payload["message"] == "start blind failed contract"
    assert payload["data"] is None


def test_telemetry_stop_collection_runtime_error_returns_error_contract(
    app, client, auth_header, monkeypatch
):
    from uav_dashboard_api.services.telemetry_manager import telemetry_manager

    monkeypatch.setattr(
        telemetry_manager,
        "stop_collection",
        lambda: (_ for _ in ()).throw(RuntimeError("stop collection failed contract")),
    )

    response = client.post("/api/telemetry/stop-collection", headers=auth_header)

    assert response.status_code == 400
    payload = response.get_json()
    assert payload["success"] is False
    assert payload["message"] == "stop collection failed contract"
    assert payload["data"] is None


def test_telemetry_stop_detection_runtime_error_returns_error_contract(
    app, client, auth_header, monkeypatch
):
    from uav_dashboard_api.services.telemetry_manager import telemetry_manager

    monkeypatch.setattr(
        telemetry_manager,
        "stop_detection",
        lambda: (_ for _ in ()).throw(RuntimeError("stop detection failed contract")),
    )

    response = client.post("/api/telemetry/stop-detection", headers=auth_header)

    assert response.status_code == 400
    payload = response.get_json()
    assert payload["success"] is False
    assert payload["message"] == "stop detection failed contract"
    assert payload["data"] is None


def test_telemetry_realtime_data_success(app, client, auth_header, monkeypatch):
    from uav_dashboard_api.services.telemetry_manager import telemetry_manager

    realtime_calls = []

    def fake_get_realtime_data():
        realtime_calls.append(True)
        return {
            "times": [1.0, 2.0],
            "motor1": [1200.0, 1210.0],
            "motor2": [1195.0, 1205.0],
            "motor3": [1201.0, 1211.0],
            "motor4": [1198.0, 1208.0],
            "roll": [0.01, 0.02],
            "pitch": [0.03, 0.04],
            "yaw": [0.05, 0.06],
            "pos_x": [0.0, 0.0],
            "pos_y": [0.0, 0.0],
            "pos_z": [-10.0, -10.5],
            "fault_prob": [0.0, 0.0],
        }

    monkeypatch.setattr(telemetry_manager, "get_realtime_data", fake_get_realtime_data)

    response = client.get("/api/telemetry/realtime-data", headers=auth_header)

    assert response.status_code == 200
    payload = response.get_json()
    assert payload["success"] is True
    assert payload["message"] == "ok"
    assert set(payload["data"].keys()) == {
        "times",
        "motor1",
        "motor2",
        "motor3",
        "motor4",
        "roll",
        "pitch",
        "yaw",
        "pos_x",
        "pos_y",
        "pos_z",
        "fault_prob",
    }
    assert payload["data"]["times"] == [1.0, 2.0]
    assert payload["data"]["motor1"] == [1200.0, 1210.0]
    assert payload["data"]["roll"] == [0.01, 0.02]
    assert payload["data"]["pos_z"] == [-10.0, -10.5]
    assert "timestamps" not in payload["data"]
    assert "fault_probability" not in payload["data"]
    assert "series" not in payload["data"]
    assert realtime_calls == [True]


def test_telemetry_realtime_data_route_returns_chart_shape_from_manager_buffer(
    app, client, auth_header
):
    from uav_dashboard_api.services.telemetry_manager import telemetry_manager

    original_realtime_buffer = telemetry_manager._realtime_buffer
    try:
        telemetry_manager._realtime_buffer = [
            {
                "time_s": 1.0,
                "motor1": 1200.0,
                "motor2": 1195.0,
                "motor3": 1201.0,
                "motor4": 1198.0,
                "roll": 0.01,
                "pitch": 0.03,
                "yaw": 0.05,
                "pos_z": -10.0,
            },
            {
                "time_s": 2.0,
                "motor1": 1210.0,
                "motor2": 1205.0,
                "motor3": 1211.0,
                "motor4": 1208.0,
                "roll": 0.02,
                "pitch": 0.04,
                "yaw": 0.06,
                "pos_z": -10.5,
            },
        ]

        response = client.get("/api/telemetry/realtime-data", headers=auth_header)

        assert response.status_code == 200
        payload = response.get_json()["data"]
        assert set(payload.keys()) == {
            "times",
            "motor1",
            "motor2",
            "motor3",
            "motor4",
            "roll",
            "pitch",
            "yaw",
            "pos_x",
            "pos_y",
            "pos_z",
            "fault_prob",
        }
        assert payload["times"] == [1.0, 2.0]
        assert payload["motor4"] == [1198.0, 1208.0]
        assert payload["pitch"] == [0.03, 0.04]
        assert payload["pos_z"] == [-10.0, -10.5]
        assert "timestamps" not in payload
        assert "fault_probability" not in payload
        assert "series" not in payload
    finally:
        telemetry_manager._realtime_buffer = original_realtime_buffer

    with app.app_context():
        older = TelemetrySession(
            id="TS-LIST-001",
            session_type="simulation",
            scenario_backend="px4",
            schema_version="live_v1",
            status="running",
            frame_count=10,
            created_at=datetime(2026, 3, 21, 9, 0, 0),
        )
        latest = TelemetrySession(
            id="TS-LIST-002",
            session_type="simulation",
            scenario_backend="replay",
            schema_version="rfly_seq_v1",
            status="completed",
            frame_count=120,
            created_at=datetime(2026, 3, 21, 10, 0, 0),
        )
        db.session.add(older)
        db.session.add(latest)
        db.session.commit()

    response = client.get(
        "/api/telemetry/sessions?status=completed&page=1&per_page=5",
        headers=auth_header,
    )

    assert response.status_code == 200
    payload = response.get_json()
    assert payload["success"] is True
    assert payload["message"] == "ok"
    assert payload["data"]["total"] == 1
    assert payload["data"]["page"] == 1
    assert payload["data"]["per_page"] == 5
    assert len(payload["data"]["items"]) == 1
    assert payload["data"]["items"][0]["id"] == "TS-LIST-002"
    assert payload["data"]["items"][0]["scenario_backend"] == "replay"
    assert payload["data"]["items"][0]["status"] == "completed"
