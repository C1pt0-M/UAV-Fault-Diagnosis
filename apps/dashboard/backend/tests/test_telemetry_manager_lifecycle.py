from __future__ import annotations

from types import SimpleNamespace

import pytest
from flask import Flask

from uav_dashboard_api.services.telemetry_manager import TelemetryProcessManager


def _app(name: str, runtime_dir) -> Flask:
    app = Flask(name)
    app.config["RUNTIME_DIR"] = runtime_dir
    return app


def test_init_app_rejects_switching_app_with_active_backend(tmp_path):
    manager = TelemetryProcessManager()
    original_app = manager.app
    original_backend = manager._backend
    original_session_id = manager._session_id
    try:
        first = _app("first", tmp_path / "first")
        second = _app("second", tmp_path / "second")
        manager._reset_runtime_state()
        manager.init_app(first)
        manager._session_id = "TS-ACTIVE"
        manager._backend = SimpleNamespace(is_collecting=lambda: True)

        with pytest.raises(RuntimeError, match="不能切换 Flask app"):
            manager.init_app(second)
    finally:
        manager._backend = original_backend
        manager._session_id = original_session_id
        manager.app = original_app
        manager._reset_runtime_state()
