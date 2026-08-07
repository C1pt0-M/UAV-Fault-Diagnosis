from __future__ import annotations

import pytest

from uav_dashboard_api import create_app, db
from uav_dashboard_api.config import TestingConfig
from uav_dashboard_api.models.asset import Asset
from uav_dashboard_api.models.user import User
from uav_dashboard_api.routes.auth import hash_password
from uav_dashboard_api.utils.jwt_utils import JWTUtils


@pytest.fixture
def app(tmp_path):
    class LocalTestingConfig(TestingConfig):
        SQLALCHEMY_DATABASE_URI = f"sqlite:///{tmp_path / 'asset_categories.db'}"
        RUNTIME_DIR = tmp_path / "runtime"
        UPLOAD_DIR = tmp_path / "runtime" / "uploads"
        DERIVED_DIR = tmp_path / "runtime" / "derived"
        RUNS_DIR = tmp_path / "runtime" / "runs"
        REPORTS_DIR = tmp_path / "runtime" / "reports"
        ENABLE_TELEMETRY = False
        ENABLE_AI = False

    app = create_app(LocalTestingConfig)
    with app.app_context():
        db.create_all()
        admin = User(
            username="admin_cat",
            password=hash_password("admin123"),
            role="admin",
            status="active",
        )
        db.session.add(admin)
        db.session.flush()

        upload_csv = Asset(
            filename="upload.csv",
            kind="csv",
            stored_path="runtime://uploads/csv/upload.csv",
            size_bytes=10,
            sha256="1" * 64,
            created_by=admin.id,
        )
        upload_csv.set_meta({"prepared_csv_path": "runtime://uploads/csv/upload.csv"})

        upload_ulg = Asset(
            filename="flight.ulg",
            kind="ulg",
            stored_path="runtime://uploads/ulg/flight.ulg",
            size_bytes=20,
            sha256="2" * 64,
            created_by=admin.id,
        )

        session_asset = Asset(
            filename="TS-123456.csv",
            kind="csv",
            stored_path="runtime://telemetry_sessions/TS-123456.csv",
            size_bytes=30,
            sha256="3" * 64,
            created_by=admin.id,
        )
        session_asset.set_meta(
            {
                "prepared_csv_path": "runtime://telemetry_sessions/TS-123456.csv",
                "source_type": "telemetry_session",
                "session_id": "TS-123456",
            }
        )
        db.session.add_all([upload_csv, upload_ulg, session_asset])
        db.session.commit()
        yield app
        db.session.remove()
        db.drop_all()


@pytest.fixture
def client(app):
    return app.test_client()


@pytest.fixture
def headers(app):
    with app.app_context():
        user = User.query.filter_by(username="admin_cat").first()
        return {"Authorization": f"Bearer {JWTUtils.generate_token(user.id)}"}


def _names(response):
    return [item["filename"] for item in response.get_json()["data"]]


def test_asset_list_filters_by_ts_csv_ulg(client, headers):
    assert set(_names(client.get("/api/assets", headers=headers))) == {
        "upload.csv",
        "flight.ulg",
        "TS-123456.csv",
    }
    assert _names(client.get("/api/assets?kind=ts", headers=headers)) == ["TS-123456.csv"]
    assert _names(client.get("/api/assets?kind=csv", headers=headers)) == ["upload.csv"]
    assert _names(client.get("/api/assets?kind=ulg", headers=headers)) == ["flight.ulg"]


def test_asset_payload_exposes_source_category(client, headers):
    items = client.get("/api/assets", headers=headers).get_json()["data"]
    by_name = {item["filename"]: item["source_category"] for item in items}
    assert by_name == {
        "upload.csv": "csv",
        "flight.ulg": "ulg",
        "TS-123456.csv": "ts",
    }
