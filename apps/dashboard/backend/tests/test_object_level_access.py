from __future__ import annotations

import pytest

from uav_dashboard_api import create_app, db
from uav_dashboard_api.config import TestingConfig
from uav_dashboard_api.models.asset import Asset
from uav_dashboard_api.models.diagnosis_report import DiagnosisReport
from uav_dashboard_api.models.run_record import RunRecord
from uav_dashboard_api.models.telemetry_session import TelemetrySession
from uav_dashboard_api.models.user import User
from uav_dashboard_api.routes.auth import hash_password
from uav_dashboard_api.utils.jwt_utils import JWTUtils


@pytest.fixture
def app(tmp_path):
    class LocalTestingConfig(TestingConfig):
        SQLALCHEMY_DATABASE_URI = f"sqlite:///{tmp_path / 'object_scope.db'}"
        RUNTIME_DIR = tmp_path / "runtime"
        UPLOAD_DIR = tmp_path / "runtime" / "uploads"
        DERIVED_DIR = tmp_path / "runtime" / "derived"
        RUNS_DIR = tmp_path / "runtime" / "runs"
        REPORTS_DIR = tmp_path / "runtime" / "reports"
        ENABLE_TELEMETRY = True
        ENABLE_AI = False

    app = create_app(LocalTestingConfig)
    with app.app_context():
        db.create_all()
        admin = User(
            username="admin_scope",
            password=hash_password("admin123"),
            role="admin",
            status="active",
        )
        user_a = User(
            username="user_a",
            password=hash_password("password123"),
            role="user",
            status="active",
        )
        user_b = User(
            username="user_b",
            password=hash_password("password123"),
            role="user",
            status="active",
        )
        db.session.add_all([admin, user_a, user_b])
        db.session.flush()
        app.config["ADMIN_ID"] = admin.id
        app.config["USER_A_ID"] = user_a.id
        app.config["USER_B_ID"] = user_b.id

        # user_a owns these resources
        asset_a = Asset(
            filename="a.csv",
            kind="csv",
            stored_path="runtime://uploads/csv/a.csv",
            size_bytes=10,
            sha256="a" * 64,
            created_by=user_a.id,
        )
        asset_a.set_meta({"prepared_csv_path": "runtime://uploads/csv/a.csv"})
        db.session.add(asset_a)
        db.session.flush()
        run_a = RunRecord(
            task="offline_multiclass",
            status="success",
            progress=1.0,
            created_by=user_a.id,
        )
        run_a.set_result({"prediction": {"label": "motor", "confidence": 0.9}})
        db.session.add(run_a)
        db.session.flush()
        report_a = DiagnosisReport(
            run_id=run_a.id,
            asset_id=asset_a.id,
            model_name="lstm",
            prediction_label="motor",
            confidence=0.9,
            created_by=user_a.id,
        )
        db.session.add(report_a)
        session_a = TelemetrySession(
            id="TS-SCOPE-A",
            drone_id="DRONE-001",
            session_type="simulation",
            scenario_backend="replay",
            csv_path="runtime://telemetry_sessions/TS-SCOPE-A.csv",
            status="completed",
            created_by=str(user_a.id),
        )
        db.session.add(session_a)
        db.session.flush()
        app.config["ASSET_A_ID"] = asset_a.id
        app.config["RUN_A_ID"] = run_a.id
        app.config["REPORT_A_ID"] = report_a.id
        db.session.commit()
        yield app
        db.session.remove()
        db.drop_all()


@pytest.fixture
def client(app):
    return app.test_client()


def _headers(app, user_id):
    with app.app_context():
        return {"Authorization": f"Bearer {JWTUtils.generate_token(user_id)}"}


def test_regular_user_cannot_see_others_assets(app, client):
    user_b = _headers(app, app.config["USER_B_ID"])

    listed = client.get("/api/assets", headers=user_b).get_json()["data"]
    assert listed == []

    for path in (
        f"/api/assets/{app.config['ASSET_A_ID']}",
        f"/api/assets/{app.config['ASSET_A_ID']}/download",
    ):
        assert client.get(path, headers=user_b).status_code == 404
    assert (
        client.delete(f"/api/assets/{app.config['ASSET_A_ID']}", headers=user_b).status_code
        == 404
    )


def test_admin_can_see_all_assets(app, client):
    admin = _headers(app, app.config["ADMIN_ID"])
    listed = client.get("/api/assets", headers=admin).get_json()["data"]
    assert any(item["id"] == app.config["ASSET_A_ID"] for item in listed)
    assert (
        client.get(f"/api/assets/{app.config['ASSET_A_ID']}", headers=admin).status_code
        == 200
    )


def test_regular_user_cannot_see_others_runs(app, client):
    user_b = _headers(app, app.config["USER_B_ID"])
    listed = client.get("/api/runs", headers=user_b).get_json()["data"]
    assert listed == []
    assert client.get(f"/api/runs/{app.config['RUN_A_ID']}", headers=user_b).status_code == 404
    assert (
        client.get(
            f"/api/runs/{app.config['RUN_A_ID']}/result", headers=user_b
        ).status_code
        == 404
    )


def test_regular_user_cannot_see_others_reports(app, client):
    user_b = _headers(app, app.config["USER_B_ID"])
    listed = client.get("/api/reports/diagnosis", headers=user_b).get_json()["data"]
    assert listed == []
    assert (
        client.get(
            f"/api/reports/diagnosis/{app.config['REPORT_A_ID']}", headers=user_b
        ).status_code
        == 404
    )
    assert (
        client.delete(
            f"/api/reports/diagnosis/{app.config['REPORT_A_ID']}", headers=user_b
        ).status_code
        == 404
    )


def test_regular_user_cannot_see_others_sessions(app, client):
    user_b = _headers(app, app.config["USER_B_ID"])
    listed = client.get("/api/telemetry/sessions", headers=user_b).get_json()["data"]
    assert listed["items"] == []
    assert (
        client.get("/api/telemetry/sessions/TS-SCOPE-A", headers=user_b).status_code == 404
    )


def test_regular_user_cannot_analyze_others_assets(app, client):
    user_b = _headers(app, app.config["USER_B_ID"])
    response = client.get(
        f"/api/analysis/fields?asset_id={app.config['ASSET_A_ID']}", headers=user_b
    )
    assert response.status_code == 404


def test_overview_is_scoped_to_current_user(app, client):
    user_b = _headers(app, app.config["USER_B_ID"])
    payload = client.get("/api/overview", headers=user_b).get_json()["data"]

    assert payload["assets_total"] == 0
    assert payload["runs_total"] == 0
    assert payload["recent_runs"] == []
    assert payload["fault_summary"]["report_count"] == 0
    assert payload["task_summary"]["completed_total"] == 0
    assert payload["warning_summary"]["confirmed_total"] == 0

    admin = _headers(app, app.config["ADMIN_ID"])
    admin_payload = client.get("/api/overview", headers=admin).get_json()["data"]

    assert admin_payload["assets_total"] == 1
    assert admin_payload["runs_total"] == 1
    assert admin_payload["fault_summary"]["report_count"] == 1
