from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pytest
from flask import Flask

from uav_dashboard_api.services.telemetry_manager import (
    TelemetryProcessManager,
    get_telemetry_manager,
    shutdown_telemetry_manager,
)


def _app(name: str, runtime_dir) -> Flask:
    app = Flask(name)
    app.config['RUNTIME_DIR'] = runtime_dir
    app.config['MODEL_ROOT'] = str(runtime_dir / 'models')
    return app


@pytest.mark.parametrize('active', [False, True])
def test_init_app_rejects_switching_owner_even_when_idle(tmp_path, active):
    first = _app('first', tmp_path / 'first')
    second = _app('second', tmp_path / 'second')
    manager = get_telemetry_manager(first)
    if active:
        manager._backend = SimpleNamespace(is_collecting=lambda: True)
    with pytest.raises(RuntimeError, match='不能切换 Flask app'):
        manager.init_app(second)
    assert manager.app is first
    assert get_telemetry_manager(first) is manager
    assert get_telemetry_manager(second) is not manager


def test_standalone_managers_are_independent():
    first = TelemetryProcessManager()
    second = TelemetryProcessManager()
    assert first is not second
    assert first._frame_queue is not second._frame_queue
    assert first._monitoring_session is not second._monitoring_session


def test_each_app_owns_its_paths_state_and_request_lookup(tmp_path):
    first = _app('first', tmp_path / 'first')
    second = _app('second', tmp_path / 'second')
    one = get_telemetry_manager(first)
    two = get_telemetry_manager(second)
    assert first.extensions['dashboard_telemetry_manager'] is one
    assert second.extensions['dashboard_telemetry_manager'] is two
    assert one._sessions_dir == tmp_path / 'first' / 'telemetry_sessions'
    assert two._sessions_dir == tmp_path / 'second' / 'telemetry_sessions'
    one._session_id = 'TS-FIRST'
    one._monitoring_session.start_detection(None)
    one._monitoring_session.ingest_external(
        result={'fault_prob': 0.8}, new_events=[], frame_idx=1, time_s=0.0,
    )
    with first.test_request_context():
        assert get_telemetry_manager() is one
        assert get_telemetry_manager().get_status()['current_fault_prob'] == 0.8
    with second.test_request_context():
        assert get_telemetry_manager() is two
        assert get_telemetry_manager().get_status()['current_fault_prob'] == 0.0
    # Popping a request/app context must not stop or reset a collection.
    assert one._session_id == 'TS-FIRST'
    assert one._monitoring_session.snapshot().state == 'running'
    shutdown_telemetry_manager(second)
    assert get_telemetry_manager(first) is one
    assert one._session_id == 'TS-FIRST'


def test_repeated_registration_does_not_reset_active_state(tmp_path):
    app = _app('first', tmp_path)
    manager = get_telemetry_manager(app)
    manager._session_id = 'TS-ACTIVE'
    manager.init_app(app)
    assert get_telemetry_manager(app) is manager
    assert manager._session_id == 'TS-ACTIVE'
    other = TelemetryProcessManager()
    with pytest.raises(RuntimeError, match='已有所属遥测管理器'):
        other.init_app(app)
    assert get_telemetry_manager(app) is manager


def test_concurrent_lookup_creates_only_one_manager(tmp_path):
    app = _app('concurrent', tmp_path)
    with ThreadPoolExecutor(max_workers=4) as executor:
        managers = list(executor.map(lambda _: get_telemetry_manager(app), range(16)))
    assert all(manager is managers[0] for manager in managers)
    assert managers[0].app is app


def test_disabled_app_does_not_create_manager(tmp_path):
    app = _app('disabled', tmp_path)
    app.config['ENABLE_TELEMETRY'] = False
    with pytest.raises(RuntimeError, match='未启用遥测'):
        get_telemetry_manager(app)
    assert 'dashboard_telemetry_manager' not in app.extensions
    shutdown_telemetry_manager(app)


def test_lookup_without_app_context_has_no_global_fallback():
    with pytest.raises(RuntimeError, match='application context'):
        get_telemetry_manager()


def test_explicit_shutdown_joins_owned_workers_and_is_idempotent(tmp_path):
    app = _app('shutdown', tmp_path)
    manager = get_telemetry_manager(app)
    manager.start_transport_processing()
    workers = [manager._frame_worker_thread, manager._inference_worker_thread]
    try:
        shutdown_telemetry_manager(app)
        assert all(not worker.is_alive() for worker in workers)
        assert 'dashboard_telemetry_manager' not in app.extensions
        shutdown_telemetry_manager(app)
    finally:
        manager.stop_transport_processing(drain=False)


def test_shutdown_uses_existing_stop_path_and_retains_owner_on_failure(tmp_path, monkeypatch):
    app = _app('shutdown', tmp_path)
    manager = get_telemetry_manager(app)
    manager._backend = SimpleNamespace(is_collecting=lambda: True)
    calls = []

    def fail_stop():
        calls.append('stop')
        raise RuntimeError('simulated stop failure')

    monkeypatch.setattr(manager, 'stop_collection', fail_stop)
    with pytest.raises(RuntimeError, match='simulated stop failure'):
        shutdown_telemetry_manager(app)
    assert calls == ['stop']
    assert get_telemetry_manager(app) is manager
    monkeypatch.setattr(manager, 'stop_collection', lambda: calls.append('retry'))
    shutdown_telemetry_manager(app)
    assert calls == ['stop', 'retry']
    assert 'dashboard_telemetry_manager' not in app.extensions


def test_shutdown_retains_owner_if_worker_did_not_exit(tmp_path, monkeypatch):
    app = _app('stuck-worker', tmp_path)
    manager = get_telemetry_manager(app)
    manager._frame_worker_thread = SimpleNamespace(is_alive=lambda: True)
    monkeypatch.setattr(manager, 'stop_transport_processing', lambda: None)
    with pytest.raises(RuntimeError, match='线程尚未退出'):
        shutdown_telemetry_manager(app)
    assert get_telemetry_manager(app) is manager


def test_local_cli_explicitly_shuts_down_on_run_exit(monkeypatch):
    from uav_dashboard_api import __main__ as cli
    from uav_dashboard_api.services import telemetry_manager as module

    def run(**kwargs):
        raise KeyboardInterrupt

    app = SimpleNamespace(config={}, run=run)
    closed = []
    monkeypatch.setattr(cli, 'create_app', lambda: app)
    monkeypatch.setattr(module, 'shutdown_telemetry_manager', closed.append)
    with pytest.raises(KeyboardInterrupt):
        cli.main()
    assert closed == [app]
