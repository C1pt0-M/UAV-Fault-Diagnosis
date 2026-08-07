from __future__ import annotations

import pytest

from uav_dashboard_api import create_app, db
from uav_dashboard_api.config import TestingConfig
from uav_dashboard_api.models.user import User
from uav_dashboard_api.routes.auth import hash_password
from uav_dashboard_api.utils.jwt_utils import JWTUtils


@pytest.fixture
def app():
    class LocalTestingConfig(TestingConfig):
        ENABLE_TELEMETRY = False
        ENABLE_AI = False
        EXPOSE_SERVER_PATHS = False

    app = create_app(LocalTestingConfig)
    with app.app_context():
        db.create_all()
        admin = User(
            username="admin_user",
            password=hash_password("admin123"),
            real_name="Admin",
            email="admin@example.com",
            role="admin",
            status="active",
        )
        user = User(
            username="normal_user",
            password=hash_password("normal123"),
            real_name="Normal",
            email="normal@example.com",
            role="user",
            status="active",
        )
        db.session.add_all([admin, user])
        db.session.commit()
        app.config["ADMIN_ID"] = admin.id
        app.config["USER_ID"] = user.id
        yield app
        db.session.remove()
        db.drop_all()


@pytest.fixture
def client(app):
    return app.test_client()


def _headers(app, user_id):
    with app.app_context():
        return {"Authorization": f"Bearer {JWTUtils.generate_token(user_id)}"}


@pytest.mark.parametrize("path", ["/api/users/", "/api/users/1", "/api/users/stats"])
def test_regular_user_cannot_read_user_directory(app, client, path):
    response = client.get(path, headers=_headers(app, app.config["USER_ID"]))
    assert response.status_code == 403


def test_admin_can_read_user_directory(app, client):
    response = client.get(
        "/api/users/",
        headers=_headers(app, app.config["ADMIN_ID"]),
    )
    assert response.status_code == 200
    assert response.get_json()["pagination"]["total"] == 2


def test_admin_can_create_multiple_users_without_email(app, client):
    headers = _headers(app, app.config["ADMIN_ID"])

    first = client.post(
        "/api/users/",
        headers=headers,
        json={"username": "user_one", "password": "password123"},
    )
    second = client.post(
        "/api/users/",
        headers=headers,
        json={"username": "user_two", "password": "password123"},
    )

    assert first.status_code == 201
    assert second.status_code == 201
    assert first.get_json()["data"]["email"] is None
    assert second.get_json()["data"]["email"] is None


def test_admin_create_user_validates_role_and_password(app, client):
    headers = _headers(app, app.config["ADMIN_ID"])

    invalid_role = client.post(
        "/api/users/",
        headers=headers,
        json={"username": "new_user", "password": "password123", "role": "owner"},
    )
    assert invalid_role.status_code == 400
    assert invalid_role.get_json()["message"] == "无效的用户角色"

    weak_password = client.post(
        "/api/users/",
        headers=headers,
        json={"username": "new_user", "password": "123"},
    )
    assert weak_password.status_code == 400
    assert "密码长度" in weak_password.get_json()["message"]


def test_admin_cannot_downgrade_or_disable_admin_account(app, client):
    headers = _headers(app, app.config["ADMIN_ID"])
    response = client.put(
        f"/api/users/{app.config['ADMIN_ID']}",
        headers=headers,
        json={"role": "user", "status": "inactive"},
    )
    assert response.status_code == 403


def test_auth_profile_validates_contact_fields_and_ignores_password(app, client):
    headers = _headers(app, app.config["USER_ID"])

    invalid_email = client.put(
        "/api/auth/profile",
        headers=headers,
        json={"email": "not-an-email"},
    )
    assert invalid_email.status_code == 400

    duplicate_email = client.put(
        "/api/auth/profile",
        headers=headers,
        json={"email": "admin@example.com"},
    )
    assert duplicate_email.status_code == 400

    response = client.put(
        "/api/auth/profile",
        headers=headers,
        json={"real_name": "Updated", "password": "replacement123"},
    )
    assert response.status_code == 200

    login_with_old_password = client.post(
        "/api/auth/login",
        json={"username": "normal_user", "password": "normal123"},
    )
    assert login_with_old_password.status_code == 200
