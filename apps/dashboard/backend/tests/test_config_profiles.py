from __future__ import annotations

import pytest

from uav_dashboard_api import _validate_runtime_config
from uav_dashboard_api.config import (
    DevelopmentConfig,
    ProductionConfig,
    TestingConfig,
    _is_pytest_process,
    resolve_config_class,
)


@pytest.mark.parametrize(
    ("profile_name", "expected"),
    [
        ("development", DevelopmentConfig),
        ("production", ProductionConfig),
        ("testing", TestingConfig),
    ],
)
def test_resolve_config_class(profile_name, expected):
    assert resolve_config_class(profile_name) is expected


def test_resolve_config_class_rejects_unknown_profile():
    with pytest.raises(ValueError, match="不支持的 APP_ENV"):
        resolve_config_class("staging")


def test_test_runner_skips_local_dotenv():
    """本地 .env 不得渗入测试进程，否则开关类配置会随开发者机器漂移。"""
    assert _is_pytest_process() is True


def test_production_profile_has_stable_identity_and_debug_policy():
    assert ProductionConfig.APP_ENV == "production"
    assert ProductionConfig.DEBUG is False


def test_production_runtime_rejects_default_or_short_secrets():
    app = type("App", (), {"config": {
        "APP_ENV": "production",
        "SECRET_KEY": "change-me",
        "JWT_SECRET_KEY": "short",
    }})()

    with pytest.raises(ValueError, match="SECRET_KEY"):
        _validate_runtime_config(app)


def test_production_runtime_requires_distinct_secrets():
    shared = "a" * 32
    app = type("App", (), {"config": {
        "APP_ENV": "production",
        "SECRET_KEY": shared,
        "JWT_SECRET_KEY": shared,
    }})()

    with pytest.raises(ValueError, match="必须使用不同值"):
        _validate_runtime_config(app)


def test_production_runtime_accepts_strong_distinct_secrets():
    app = type("App", (), {"config": {
        "APP_ENV": "production",
        "SECRET_KEY": "s" * 32,
        "JWT_SECRET_KEY": "j" * 32,
    }})()

    _validate_runtime_config(app)


@pytest.mark.parametrize('ai_enabled,telemetry_enabled', [(False, False), (False, True), (True, False), (True, True)])
def test_authenticated_capabilities_match_registered_optional_routes(tmp_path, monkeypatch, ai_enabled, telemetry_enabled):
    from uav_dashboard_api import create_app, db
    from uav_dashboard_api.models.user import User
    from uav_dashboard_api.services.ai_assistant import ai_assistant_service
    from uav_dashboard_api.services.telemetry_manager import shutdown_telemetry_manager
    from uav_dashboard_api.utils.jwt_utils import JWTUtils

    from uav_dashboard_api.routes import settings

    monkeypatch.setattr(ai_assistant_service, 'init_app', lambda app: None)
    monkeypatch.setattr(settings, 'OpenAICompatibleProvider', lambda config: pytest.fail('connection test must not run'))
    runtime = tmp_path / 'runtime'

    class LocalConfig(TestingConfig):
        ENABLE_AI = ai_enabled
        ENABLE_TELEMETRY = telemetry_enabled
        ENABLE_SETUP_API = False
        SQLALCHEMY_DATABASE_URI = 'sqlite:///:memory:'
        RUNTIME_DIR = runtime
        UPLOAD_DIR = runtime / 'uploads'
        DERIVED_DIR = runtime / 'derived'
        RUNS_DIR = runtime / 'runs'
        REPORTS_DIR = runtime / 'reports'
        AI_RUNTIME_DIR = runtime / 'ai'
        AI_SETTINGS_PATH = runtime / 'ai_settings.json'
        AI_API_KEY = 'must-not-appear-in-capabilities'

    app = create_app(LocalConfig)
    client = app.test_client()
    try:
        with app.app_context():
            admin = User(username='caps_admin', password='unused', role='admin', status='active')
            user = User(username='caps_user', password='unused', role='user', status='active')
            db.session.add_all([admin, user])
            db.session.commit()
            admin_headers = {'Authorization': f'Bearer {JWTUtils.generate_token(admin.id)}'}
            user_headers = {'Authorization': f'Bearer {JWTUtils.generate_token(user.id)}'}

        assert client.get('/api/auth/profile').status_code == 401
        for headers in (admin_headers, user_headers):
            response = client.get('/api/auth/profile', headers=headers)
            assert response.status_code == 200
            assert response.get_json()['capabilities'] == {'ai': ai_enabled, 'telemetry': telemetry_enabled}
            assert 'no-store' in response.headers['Cache-Control']
            assert 'must-not-appear' not in response.get_data(as_text=True)
        assert client.get('/api/ai/conversations').status_code == (401 if ai_enabled else 404)
        assert client.get('/api/telemetry/status').status_code == (401 if telemetry_enabled else 404)
        for method, path in [('get', '/ai'), ('put', '/ai'), ('post', '/ai/test'), ('post', '/ai/test-embedding')]:
            call = getattr(client, method)
            assert call(f'/api/settings{path}').status_code == (401 if ai_enabled else 404)
            assert call(f'/api/settings{path}', headers=user_headers).status_code == (403 if ai_enabled else 404)
            if not ai_enabled:
                assert call(f'/api/settings{path}', headers=admin_headers).status_code == 404
        assert not LocalConfig.AI_SETTINGS_PATH.exists()
    finally:
        shutdown_telemetry_manager(app)
        with app.app_context():
            db.session.remove()
            db.drop_all()
