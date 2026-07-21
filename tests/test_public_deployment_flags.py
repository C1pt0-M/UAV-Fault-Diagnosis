from __future__ import annotations

from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
from uav_dashboard_api import create_app, db
from uav_dashboard_api.routes.auth import hash_password
from uav_dashboard_api.models.user import User
from uav_dashboard_api.config import TestingConfig


class ReplayOnlyDeploymentConfig(TestingConfig):
    SECRET_KEY = "test-secret"
    JWT_SECRET_KEY = "test-jwt-secret"
    AI_ENABLE_KNOWLEDGE_BASE = False
    ENABLE_PUBLIC_REGISTER = False
    ENABLE_SETUP_API = False
    ENABLE_TELEMETRY = True
    ENABLE_BLIND_TEST = False
    TELEMETRY_ALLOWED_BACKENDS = ("replay",)
    DEFAULT_ONLINE_MODEL = "online_detector_cnn1d_rfly"
    MODEL_ROOT = str(PROJECT_ROOT / "assets" / "models")


@pytest.fixture()
def app():
    app = create_app(config_class=ReplayOnlyDeploymentConfig)
    with app.app_context():
        db.session.query(User).delete()
        admin = User(
            username="admin",
            password=hash_password("admin123"),
            real_name="Admin",
            email="admin@example.com",
            role="admin",
            status="active",
        )
        db.session.add(admin)
        db.session.commit()
    return app


@pytest.fixture()
def client(app):
    return app.test_client()


@pytest.fixture()
def auth_header(client):
    response = client.post(
        "/api/auth/login",
        json={"username": "admin", "password": "admin123"},
    )
    assert response.status_code == 200
    token = response.get_json()["data"]["token"]
    return {"Authorization": f"Bearer {token}"}


def test_setup_status_remains_available_for_first_use_initialization(client):
    response = client.get("/api/setup/status")
    assert response.status_code == 200
    payload = response.get_json()["data"]
    assert payload["is_installed"] is True
    assert payload["needs_setup"] is False


def test_register_route_is_disabled_when_public_register_off(client):
    response = client.post(
        "/api/auth/register",
        json={
            "username": "newuser",
            "password": "newuser123",
            "email": "newuser@example.com",
        },
    )
    assert response.status_code == 403
    assert response.get_json()["message"] == "public_registration_disabled"


def test_telemetry_status_exposes_replay_only_capabilities(client, auth_header):
    response = client.get("/api/telemetry/status", headers=auth_header)
    assert response.status_code == 200
    payload = response.get_json()["data"]

    assert payload["telemetry_capabilities"]["allowed_backends"] == ["replay"]
    assert payload["telemetry_capabilities"]["blind_test_enabled"] is False
    assert payload["telemetry_capabilities"]["px4_enabled"] is False


def test_telemetry_rejects_px4_collection_when_only_replay_is_allowed(
    client, auth_header
):
    response = client.post(
        "/api/telemetry/start-collection",
        headers=auth_header,
        json={
            "backend": "px4",
            "session_type": "simulation",
            "sim_time": 60,
        },
    )
    assert response.status_code == 400
    assert response.get_json()["message"] == "backend_not_allowed"


def test_blind_test_route_is_disabled_when_flag_is_off(client, auth_header):
    response = client.post(
        "/api/telemetry/blind-test/start",
        headers=auth_header,
        json={"sim_time": 60},
    )
    assert response.status_code == 403
    assert response.get_json()["message"] == "blind_test_disabled"
