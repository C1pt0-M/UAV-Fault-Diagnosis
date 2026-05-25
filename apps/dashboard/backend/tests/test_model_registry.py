import json
from pathlib import Path

import bcrypt
import pytest

from uav_dashboard_api import create_app, db
from uav_dashboard_api.models.asset import Asset
from uav_dashboard_api.models.diagnosis_report import DiagnosisReport
from uav_dashboard_api.models.run_record import RunRecord
from uav_dashboard_api.models.telemetry_session import TelemetryFaultEvent, TelemetrySession
from uav_dashboard_api.models.user import User
from uav_dashboard_api.services.model_registry import list_registered_models, resolve_registered_model
from uav_dashboard_api.utils.jwt_utils import JWTUtils
from uav_dashboard_api.config import TestingConfig


def _write_json(path: Path, payload: dict):
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')


def _create_model(
    model_root: Path,
    name: str,
    *,
    config: dict,
    metrics: dict | None = None,
    norm_stats: dict | None = None,
):
    model_dir = model_root / name
    model_dir.mkdir(parents=True, exist_ok=True)
    _write_json(model_dir / "config.json", config)
    (model_dir / "best.pt").write_bytes(b"checkpoint")
    if metrics is not None:
        _write_json(model_dir / "metrics_test.json", metrics)
    if norm_stats is not None:
        _write_json(model_dir / "norm_stats.json", norm_stats)
    return model_dir


@pytest.fixture
def model_root(tmp_path: Path) -> Path:
    model_root = tmp_path / "models"
    model_root.mkdir()

    _create_model(
        model_root,
        "lstm",
        config={
            "model": "lstm",
            "in_channels": 43,
            "num_classes": 5,
            "best_val_f1": 0.91,
        },
        metrics={"macro_f1": 0.88, "accuracy": 0.9, "weighted_f1": 0.89},
    )
    _create_model(
        model_root,
        "online_detector_alpha",
        config={
            "model": "cnn1d",
            "in_channels": 33,
            "num_classes": 2,
            "label_names": ["normal", "fault"],
            "best_val_f1": 0.7,
        },
        metrics={"macro_f1": 0.7, "accuracy": 0.8, "weighted_f1": 0.78},
        norm_stats={"columns": ["roll", "pitch"], "mean": [0.0, 0.0], "std": [1.0, 1.0]},
    )
    _create_model(
        model_root,
        "online_detector_beta",
        config={
            "model": "cnn1d",
            "in_channels": 33,
            "num_classes": 2,
            "label_names": ["normal", "fault"],
            "best_val_f1": 0.95,
        },
        metrics={"macro_f1": 0.94, "accuracy": 0.96, "weighted_f1": 0.95},
        norm_stats={"columns": ["roll", "pitch"], "mean": [0.0, 0.0], "std": [1.0, 1.0]},
    )
    _create_model(
        model_root,
        "cnn_transformer_4class",
        config={
            "model": "cnn_transformer",
            "in_channels": 48,
            "num_classes": 4,
            "label_names": ["normal", "motor", "prop", "sensor"],
        },
    )

    return model_root


@pytest.fixture
def registry_config(model_root: Path) -> dict:
    return {
        "MODEL_ROOT": str(model_root),
        "MODEL_WHITELIST": (),
        "DEFAULT_OFFLINE_MODEL": "",
        "DEFAULT_ONLINE_MODEL": "online_detector_alpha",
        "ONLINE_MODEL_PRIORITY": (),
    }


@pytest.fixture
def app(model_root: Path, tmp_path: Path):
    db_path = tmp_path / "test_dashboard.db"
    runtime_dir = tmp_path / "runtime"

    class LocalTestingConfig(TestingConfig):
        SQLALCHEMY_DATABASE_URI = f"sqlite:///{db_path}"
        MODEL_ROOT = str(model_root)
        MODEL_WHITELIST = ()
        DEFAULT_OFFLINE_MODEL = ""
        DEFAULT_ONLINE_MODEL = "online_detector_alpha"
        ONLINE_MODEL_PRIORITY = ()
        RUNTIME_DIR = runtime_dir
        UPLOAD_DIR = runtime_dir / "uploads"
        DERIVED_DIR = runtime_dir / "derived"
        RUNS_DIR = runtime_dir / "runs"
        REPORTS_DIR = runtime_dir / "reports"

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
        hashed = bcrypt.hashpw("test123".encode("utf-8"), bcrypt.gensalt()).decode("utf-8")
        user = User(
            username="registry_user",
            password=hashed,
            real_name="Registry User",
            email="registry@example.com",
            role="admin",
            status="active",
        )
        db.session.add(user)
        db.session.commit()
        token = JWTUtils.generate_token(user.id)
        return {"Authorization": f"Bearer {token}"}


def test_list_registered_models_filters_dashboard_capability(registry_config):
    offline_models = list_registered_models(registry_config, capability="offline_detection")
    assert [item["name"] for item in offline_models] == ["lstm"]

    online_models = list_registered_models(registry_config, capability="realtime_detection")
    assert [item["name"] for item in online_models] == ["online_detector_beta", "online_detector_alpha"]

    all_models = list_registered_models(registry_config)
    assert [item["name"] for item in all_models] == [
        "online_detector_beta",
        "lstm",
        "online_detector_alpha",
        "cnn_transformer_4class",
    ]


def test_resolve_registered_model_prefers_explicit_default_online_model(registry_config):
    selected = resolve_registered_model(
        registry_config,
        task="online_binary",
        capability="realtime_detection",
    )
    assert selected is not None
    assert selected["name"] == "online_detector_alpha"


def test_online_model_priority_controls_realtime_list_order(registry_config):
    registry_config["ONLINE_MODEL_PRIORITY"] = (
        "online_detector_alpha",
        "online_detector_beta",
    )

    online_models = list_registered_models(
        registry_config,
        capability="realtime_detection",
    )

    assert [item["name"] for item in online_models] == [
        "online_detector_alpha",
        "online_detector_beta",
    ]


def test_resolve_registered_online_model_exposes_runtime_summary_fields(registry_config):
    selected = resolve_registered_model(
        registry_config,
        task="online_binary",
        capability="realtime_detection",
    )
    assert selected is not None
    assert selected["window_size"] is None
    assert selected["feature_count"] == 2
    assert selected["metrics"]["best_val_f1"] == 0.7


def test_models_endpoint_filters_by_capability(client, auth_header):
    response = client.get("/api/models", headers=auth_header)
    assert response.status_code == 200
    payload = response.get_json()
    assert payload["success"] is True
    assert [item["name"] for item in payload["data"]] == ["lstm"]

    response = client.get("/api/models?capability=realtime_detection", headers=auth_header)
    assert response.status_code == 200
    payload = response.get_json()
    assert [item["name"] for item in payload["data"]] == ["online_detector_beta", "online_detector_alpha"]

    response = client.get("/api/models?include_unavailable=true", headers=auth_header)
    assert response.status_code == 200
    payload = response.get_json()
    assert "cnn_transformer_4class" in [item["name"] for item in payload["data"]]


def test_overview_endpoint_exposes_industrial_dashboard_aggregates(app, client, auth_header):
    with app.app_context():
        csv_asset = Asset(
            filename="flight.csv",
            kind="csv",
            stored_path="runtime/uploads/csv/flight.csv",
            size_bytes=100,
            sha256="a" * 64,
        )
        prepared_ulg = Asset(
            filename="prepared.ulg",
            kind="ulg",
            stored_path="runtime/uploads/ulg/prepared.ulg",
            size_bytes=200,
            sha256="b" * 64,
        )
        prepared_ulg.set_meta({"prepared_csv_path": "runtime/derived/prepared.csv"})
        raw_ulg = Asset(
            filename="raw.ulg",
            kind="ulg",
            stored_path="runtime/uploads/ulg/raw.ulg",
            size_bytes=300,
            sha256="c" * 64,
        )
        db.session.add_all([csv_asset, prepared_ulg, raw_ulg])
        db.session.flush()

        run = RunRecord(
            task="offline_multiclass",
            status="success",
            progress=1.0,
            asset_id=csv_asset.id,
        )
        run.set_result({
            "diagnosis": {
                "evidence_summary": {
                    "top_evidence": [
                        {"name": "motor_current_rms", "value": 0.82, "status": "observed"}
                    ]
                }
            }
        })
        db.session.add(run)
        db.session.flush()

        report = DiagnosisReport(
            run_id=run.id,
            asset_id=csv_asset.id,
            model_name="lstm",
            prediction_label="motor",
            confidence=0.86,
            pdf_path="runtime/reports/report.pdf",
        )
        report.set_summary({
            "severity": {"level": "high"},
            "recommendations": [{"title": "Check motor current"}],
        })
        db.session.add(report)

        session = TelemetrySession(
            id="TS-UNIT",
            drone_id="UAV-1023",
            session_type="simulation",
            scenario_backend="replay",
            schema_version="v1.4",
            csv_path="runtime/telemetry/session.csv",
            status="completed",
            frame_count=128,
            fault_events_count=1,
        )
        session.set_meta({
            "feature_coverage": {
                "schema_version": "v1.4",
                "mapped_count": 43,
                "default_count": 3,
                "feature_count": 48,
            }
        })
        db.session.add(session)
        db.session.flush()

        event = TelemetryFaultEvent(
            id="FE-UNIT",
            session_id=session.id,
            start_time=10.0,
            end_time=12.0,
            duration=2.0,
            max_fault_prob=0.91,
            mean_fault_prob=0.77,
            data_points=20,
            severity="high",
        )
        event.set_diagnosis({
            "label": "fault",
            "confidence": 0.91,
            "severity": "high",
            "evidence_summary": {
                "top_evidence": [
                    {"name": "fault_probability", "value": 0.91, "status": "observed"},
                    {"name": "warning_stage", "value": "critical", "status": "observed"},
                ]
            },
        })
        db.session.add(event)
        db.session.commit()
        report_id = report.id

    response = client.get("/api/overview", headers=auth_header)
    assert response.status_code == 200
    data = response.get_json()["data"]

    assert data["assets_total"] == 3
    assert data["asset_summary"]["diagnosable_total"] == 2
    assert data["task_summary"]["completed_total"] == 1
    assert data["warning_summary"]["confirmed_total"] == 1
    assert data["fault_summary"]["dominant_label"] == "motor"
    assert data["fault_summary"]["avg_confidence"] == 0.86
    assert data["model_summary"]["name"] == "online_detector_alpha"
    assert data["field_contract"]["status"] in {"passed", "degraded"}
    assert data["warning_summary"]["recent"][0]["message"] == "故障概率异常"
    assert data["risk_trend"]["points"]
    assert data["recent_evidence"][0]["name"] == "故障概率异常"
    assert data["recent_evidence"][1]["name"] == "告警阶段"
    assert data["similar_reports"][0]["id"] == report_id


def test_overview_field_contract_explains_waiting_coverage(app, client, auth_header):
    with app.app_context():
        session = TelemetrySession(
            id="TS-WAITING",
            drone_id="UAV-WAITING",
            session_type="simulation",
            scenario_backend="replay",
            schema_version="rfly_seq_v1",
            csv_path="runtime/telemetry/waiting.csv",
            status="completed",
        )
        db.session.add(session)
        db.session.commit()

    response = client.get("/api/overview", headers=auth_header)
    assert response.status_code == 200
    field_contract = response.get_json()["data"]["field_contract"]

    assert field_contract["status"] in {"waiting", "degraded"}
    assert field_contract["schema_version"] in {"rfly_seq_v1", "-"}
    assert field_contract["source"] in {"最近一次输入", "当前输入", "等待输入"}
    assert field_contract["message"] in {"等待特征覆盖数据", "存在默认填充字段"}
    assert field_contract["reason"] in {
        "latest_session_without_feature_coverage",
        "coverage_available",
    }


def test_telemetry_status_endpoint_exposes_online_model_summary(client, auth_header, monkeypatch):
    from uav_dashboard_api.services.telemetry_manager import telemetry_manager

    monkeypatch.setattr(telemetry_manager, "get_status", lambda: {
        "is_collecting": True,
        "is_detecting": True,
        "is_blind_test": False,
        "current_session_id": "TS-123456",
        "uptime_seconds": 12.3,
        "planned_duration_s": 120.0,
        "frame_count": 321,
        "fault_events_count": 2,
        "fault_events": [],
        "online_model": {
            "name": "online_detector_alpha",
            "architecture": "cnn1d",
            "window_size": 150,
            "feature_count": 33,
            "metrics": {"best_val_f1": 0.7, "macro_f1": 0.7, "f1_fault": 0.69, "accuracy": 0.8},
        },
        "scenario_backend": "replay",
        "schema_version": "rfly_seq_v1",
        "feature_coverage": {"mapped": ["roll", "pitch"], "filled": ["yaw"]},
        "replay_metrics": {"pred_event_count": 2, "hit": 1, "false_alarm": 1},
        "processes": {"px4_sitl": "running", "collector": "running", "receiver": "running", "detector": "running"},
    })

    response = client.get("/api/telemetry/status", headers=auth_header)
    assert response.status_code == 200
    payload = response.get_json()["data"]
    assert payload["online_model"]["window_size"] == 150
    assert payload["online_model"]["feature_count"] == 33
    assert payload["online_model"]["metrics"]["best_val_f1"] == 0.7
    assert payload["scenario_backend"] == "replay"
    assert payload["schema_version"] == "rfly_seq_v1"
    assert payload["feature_coverage"]["mapped"] == ["roll", "pitch"]
    assert payload["replay_metrics"]["false_alarm"] == 1


def test_start_detection_endpoint_returns_online_model_summary(client, auth_header, monkeypatch):
    from uav_dashboard_api.services.telemetry_manager import telemetry_manager

    captured = {}

    def fake_start_detection(*, requested_model=None):
        captured["requested_model"] = requested_model
        return {
            "detector_status": "running",
            "model": {
                "name": "online_detector_alpha",
                "task": "online_binary",
                "architecture": "cnn1d",
                "num_classes": 2,
                "label_names": ["normal", "fault"],
                "window_size": 150,
                "feature_count": 33,
                "metrics": {"best_val_f1": 0.7, "macro_f1": 0.7, "f1_fault": 0.69, "accuracy": 0.8},
            },
        }

    monkeypatch.setattr(telemetry_manager, "start_detection", fake_start_detection)

    response = client.post("/api/telemetry/start-detection", headers=auth_header)
    assert response.status_code == 200
    assert captured["requested_model"] is None
    payload = response.get_json()["data"]
    assert payload["detector_status"] == "running"
    assert payload["model"]["window_size"] == 150
    assert payload["model"]["feature_count"] == 33
    assert payload["model"]["metrics"]["macro_f1"] == 0.7


def test_model_registry_infers_multiclass_task_for_four_class_model(registry_config):
    selected = resolve_registered_model(
        registry_config,
        requested_name='cnn_transformer_4class',
    )
    assert selected is not None
    assert selected['task'] == 'offline_multiclass'
    assert selected['capabilities']['offline_detection'] is False

    models = list_registered_models(registry_config, capability='offline_detection', include_unavailable=True)
    by_name = {item['name']: item for item in models}
    assert by_name['lstm']['task'] == 'offline_5class'
    assert by_name['cnn_transformer_4class']['task'] == 'offline_multiclass'


def test_manifest_registry_ignores_unlisted_models_and_rejects_missing_entries(tmp_path: Path):
    model_root = tmp_path / "models"
    model_root.mkdir()
    _create_model(
        model_root,
        "listed_online",
        config={
            "model": "cnn1d",
            "in_channels": 33,
            "num_classes": 2,
            "label_names": ["normal", "fault"],
        },
        norm_stats={"columns": ["roll"], "mean": [0.0], "std": [1.0]},
    )
    _create_model(
        model_root,
        "unlisted_online",
        config={
            "model": "cnn1d",
            "in_channels": 33,
            "num_classes": 2,
            "label_names": ["normal", "fault"],
            "best_val_f1": 0.99,
        },
        norm_stats={"columns": ["roll"], "mean": [0.0], "std": [1.0]},
    )
    _write_json(
        model_root / "manifest.json",
        {
            "version": 1,
            "models": [
                {"name": "listed_online", "path": "listed_online"},
            ],
        },
    )

    config = {
        "MODEL_ROOT": str(model_root),
        "MODEL_WHITELIST": (),
        "DEFAULT_OFFLINE_MODEL": "",
        "DEFAULT_ONLINE_MODEL": "",
        "ONLINE_MODEL_PRIORITY": (),
    }

    models = list_registered_models(config, capability="realtime_detection")
    assert [item["name"] for item in models] == ["listed_online"]

    _write_json(
        model_root / "manifest.json",
        {
            "version": 1,
            "models": [
                {"name": "listed_online", "path": "listed_online"},
                {"name": "missing_online", "path": "missing_online"},
            ],
        },
    )

    with pytest.raises(FileNotFoundError, match="missing_online"):
        list_registered_models(config, capability="realtime_detection")
