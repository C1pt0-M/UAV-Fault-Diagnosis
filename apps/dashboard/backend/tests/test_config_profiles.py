from __future__ import annotations

import pytest

from uav_dashboard_api import _validate_runtime_config
from uav_dashboard_api.config import (
    DevelopmentConfig,
    ProductionConfig,
    TestingConfig,
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
