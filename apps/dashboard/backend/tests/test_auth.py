from __future__ import annotations

import json

import bcrypt
import pytest

from uav_dashboard_api import create_app, db
from uav_dashboard_api.config import TestingConfig
from uav_dashboard_api.models.user import User


@pytest.fixture
def app():
    class LocalTestingConfig(TestingConfig):
        ENABLE_PUBLIC_REGISTER = True
        ENABLE_SETUP_API = True
        ALLOW_AUTO_BOOTSTRAP_ADMIN = False

    app = create_app(LocalTestingConfig)

    with app.app_context():
        db.create_all()
        yield app
        db.drop_all()


@pytest.fixture
def client(app):
    return app.test_client()


@pytest.fixture
def disabled_register_client():
    class LocalTestingConfig(TestingConfig):
        ENABLE_PUBLIC_REGISTER = False
        ENABLE_SETUP_API = True
        ALLOW_AUTO_BOOTSTRAP_ADMIN = False

    app = create_app(LocalTestingConfig)

    with app.app_context():
        db.create_all()
        yield app.test_client()
        db.drop_all()


@pytest.fixture
def test_user(app):
    with app.app_context():
        hashed = bcrypt.hashpw("test123".encode("utf-8"), bcrypt.gensalt())
        user = User(
            username="testuser",
            password=hashed.decode("utf-8"),
            real_name="Test User",
            email="test@example.com",
            role="user",
            status="active",
        )
        db.session.add(user)
        db.session.commit()
        return user


def _login_token(client) -> str:
    response = client.post(
        "/api/auth/login",
        json={"username": "testuser", "password": "test123"},
    )
    assert response.status_code == 200
    return json.loads(response.data)["data"]["token"]


class TestAuth:
    def test_register_success_by_default(self, client):
        response = client.post(
            "/api/auth/register",
            json={
                "username": "newuser",
                "password": "password123",
            },
        )

        assert response.status_code == 201
        data = json.loads(response.data)
        assert data["success"] is True
        assert data["data"]["username"] == "newuser"
        assert data["data"]["role"] == "user"

    def test_register_can_be_disabled_by_config(self, disabled_register_client):
        response = disabled_register_client.post(
            "/api/auth/register",
            json={
                "username": "newuser",
                "password": "password123",
            },
        )

        assert response.status_code == 403
        data = json.loads(response.data)
        assert data["success"] is False
        assert data["message"] == "public_registration_disabled"

    def test_register_missing_fields(self, client):
        response = client.post("/api/auth/register", json={"username": "newuser"})

        assert response.status_code == 400
        data = json.loads(response.data)
        assert data["success"] is False

    def test_register_duplicate_username(self, client):
        client.post(
            "/api/auth/register",
            json={
                "username": "testuser",
                "password": "password123",
                "real_name": "Test User",
                "email": "test@example.com",
            },
        )

        response = client.post(
            "/api/auth/register",
            json={
                "username": "testuser",
                "password": "password123",
                "real_name": "Other User",
                "email": "other@example.com",
            },
        )

        assert response.status_code == 400
        data = json.loads(response.data)
        assert data["success"] is False

    def test_login_rejects_empty_json_body(self, client):
        response = client.post("/api/auth/login", data="", content_type="application/json")

        assert response.status_code == 400
        data = json.loads(response.data)
        assert data["success"] is False

    def test_login_success(self, client, test_user):
        response = client.post(
            "/api/auth/login",
            json={"username": "testuser", "password": "test123"},
        )

        assert response.status_code == 200
        data = json.loads(response.data)
        assert data["success"] is True
        assert "token" in data["data"]
        assert "user" in data["data"]

    def test_login_wrong_password(self, client, test_user):
        response = client.post(
            "/api/auth/login",
            json={"username": "testuser", "password": "wrongpassword"},
        )

        assert response.status_code == 401
        data = json.loads(response.data)
        assert data["success"] is False

    def test_login_nonexistent_user(self, client):
        response = client.post(
            "/api/auth/login",
            json={"username": "nonexistent", "password": "password123"},
        )

        assert response.status_code == 401
        data = json.loads(response.data)
        assert data["success"] is False

    def test_login_disabled_user(self, client, app):
        with app.app_context():
            hashed = bcrypt.hashpw("test123".encode("utf-8"), bcrypt.gensalt())
            user = User(
                username="disabled_user",
                password=hashed.decode("utf-8"),
                real_name="Disabled User",
                email="disabled@example.com",
                role="admin",
                status="inactive",
            )
            db.session.add(user)
            db.session.commit()

        response = client.post(
            "/api/auth/login",
            json={"username": "disabled_user", "password": "test123"},
        )

        assert response.status_code == 403
        data = json.loads(response.data)
        assert data["success"] is False
        assert data["message"] == "账号已被禁用"

    def test_logout(self, client):
        response = client.post("/api/auth/logout")

        assert response.status_code == 200
        data = json.loads(response.data)
        assert data["success"] is True

    def test_get_profile_without_token(self, client):
        response = client.get("/api/auth/profile")

        assert response.status_code == 401

    def test_get_profile_with_invalid_token(self, client):
        response = client.get(
            "/api/auth/profile",
            headers={"Authorization": "Bearer invalid-token"},
        )

        assert response.status_code == 401
        data = json.loads(response.data)
        assert data["success"] is False

    def test_get_profile_with_token(self, client, test_user):
        token = _login_token(client)

        response = client.get(
            "/api/auth/profile",
            headers={"Authorization": f"Bearer {token}"},
        )

        assert response.status_code == 200
        data = json.loads(response.data)
        assert data["success"] is True
        assert data["data"]["username"] == "testuser"

    def test_update_profile(self, client, test_user):
        token = _login_token(client)

        response = client.put(
            "/api/auth/profile",
            headers={"Authorization": f"Bearer {token}"},
            json={"real_name": "Updated User", "email": "updated@example.com"},
        )

        assert response.status_code == 200
        data = json.loads(response.data)
        assert data["success"] is True
        assert data["data"]["real_name"] == "Updated User"


class TestSetup:
    def test_setup_status_before_initialization(self, client):
        response = client.get("/api/setup/status")

        assert response.status_code == 200
        data = json.loads(response.data)
        assert data["success"] is True
        assert data["data"]["is_installed"] is False
        assert data["data"]["needs_setup"] is True

    def test_initialize_system_creates_admin(self, client):
        response = client.post(
            "/api/setup/initialize",
            json={
                "username": "admin_user",
                "password": "admin123",
                "real_name": "Admin User",
                "email": "admin@example.com",
            },
        )

        assert response.status_code == 200
        data = json.loads(response.data)
        assert data["success"] is True
        assert data["data"]["username"] == "admin_user"
        assert data["data"]["role"] == "admin"
        assert "token" not in data["data"]

        login_response = client.post(
            "/api/auth/login",
            json={"username": "admin_user", "password": "admin123"},
        )
        assert login_response.status_code == 200

    def test_initialize_system_rejects_empty_json_body(self, client):
        response = client.post("/api/setup/initialize", data="", content_type="application/json")

        assert response.status_code == 400
        data = json.loads(response.data)
        assert data["success"] is False

    def test_initialize_system_rejects_repeat_initialization(self, client, test_user):
        response = client.post(
            "/api/setup/initialize",
            json={
                "username": "admin_user",
                "password": "admin123",
                "real_name": "Admin User",
                "email": "admin@example.com",
            },
        )

        assert response.status_code == 400
        data = json.loads(response.data)
        assert data["success"] is False
        assert data["message"] == "系统已安装，无法重复初始化"
