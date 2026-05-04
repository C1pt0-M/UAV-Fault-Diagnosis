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

    app = create_app(LocalTestingConfig)

    with app.app_context():
        db.create_all()
        yield app
        db.drop_all()


@pytest.fixture
def client(app):
    return app.test_client()


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
    def test_register_success(self, client):
        response = client.post(
            "/api/auth/register",
            json={
                "username": "newuser",
                "password": "password123",
                "real_name": "New User",
                "email": "new@example.com",
            },
        )

        assert response.status_code == 201
        data = json.loads(response.data)
        assert data["success"] is True
        assert data["data"]["username"] == "newuser"

    def test_register_missing_fields(self, client):
        response = client.post("/api/auth/register", json={"username": "newuser"})

        assert response.status_code == 400
        data = json.loads(response.data)
        assert data["success"] is False

    def test_register_duplicate_username(self, client, test_user):
        response = client.post(
            "/api/auth/register",
            json={"username": "testuser", "password": "password123"},
        )

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

    def test_logout(self, client):
        response = client.post("/api/auth/logout")

        assert response.status_code == 200
        data = json.loads(response.data)
        assert data["success"] is True

    def test_get_profile_without_token(self, client):
        response = client.get("/api/auth/profile")

        assert response.status_code == 401

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
