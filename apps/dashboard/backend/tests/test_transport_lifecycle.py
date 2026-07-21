from __future__ import annotations

import threading
from pathlib import Path
from types import SimpleNamespace

import pytest

from uav_dashboard_api.services.telemetry_backends import ScenarioBackend


class _Manager:
    def __init__(self) -> None:
        self.snapshots = []

    def on_transport_lifecycle(self, snapshot, session_id, backend) -> None:
        self.snapshots.append(snapshot)


class _Backend(ScenarioBackend):
    name = "test"

    def __init__(self, manager, *, fail_start: bool = False) -> None:
        super().__init__(
            manager,
            session_id="TS-LIFECYCLE",
            csv_path="session.csv",
            sim_time=1.0,
            options={},
        )
        self.fail_start = fail_start
        self.stop_calls = 0

    def _start(self) -> None:
        self._transition(phase="starting_test_transport")
        if self.fail_start:
            raise RuntimeError("transport unavailable")

    def _stop(self) -> None:
        self.stop_calls += 1

    def _status(self) -> dict:
        return {"receiver": "running" if self.lifecycle().state == "running" else None}


def test_transport_lifecycle_has_one_normal_transition_path() -> None:
    manager = _Manager()
    backend = _Backend(manager)

    backend.prepare()
    backend.start()
    running = backend.status()["transport_lifecycle"]
    backend.stop()
    backend.stop()

    assert [snapshot.state for snapshot in manager.snapshots] == [
        "preparing",
        "prepared",
        "starting",
        "starting",
        "running",
        "stopping",
        "stopped",
    ]
    assert running["state"] == "running"
    assert running["phase"] == "test_running"
    assert backend.lifecycle().state == "stopped"
    assert backend.stop_calls == 1


def test_transport_lifecycle_publishes_start_failure() -> None:
    manager = _Manager()
    backend = _Backend(manager, fail_start=True)
    backend.prepare()

    with pytest.raises(RuntimeError, match="transport unavailable"):
        backend.start()

    snapshot = backend.lifecycle()
    assert snapshot.state == "failed"
    assert snapshot.phase == "test_start_failed"
    assert snapshot.failure_reason == "test_start_failed"
    assert snapshot.detail == "transport unavailable"

    backend.stop()
    stopped = backend.lifecycle()
    assert stopped.state == "stopped"
    assert stopped.failure_reason == "test_start_failed"
    assert stopped.detail == "transport unavailable"


def test_transport_lifecycle_rejects_double_start() -> None:
    backend = _Backend(_Manager())
    backend.prepare()
    backend.start()

    with pytest.raises(RuntimeError, match="cannot start from running"):
        backend.start()


def test_transport_rejects_process_that_exits_during_startup() -> None:
    process = SimpleNamespace(poll=lambda: 2)

    with pytest.raises(RuntimeError, match="启动后提前退出"):
        ScenarioBackend._ensure_process_alive(process, "test process")


def test_px4_stop_cancels_delayed_pipeline_before_process_spawn(tmp_path) -> None:
    from uav_dashboard_api.services.telemetry_backends import PX4ScenarioBackend

    wait_entered = threading.Event()
    release_wait = threading.Event()
    spawn_calls = []

    def wait_for_px4(*, timeout, generation):
        wait_entered.set()
        release_wait.wait(2.0)
        return True

    manager = SimpleNamespace(
        app=None,
        _blind_test_active=False,
        resolve_telemetry_python_bin=lambda: "python-bin",
        on_transport_lifecycle=lambda snapshot, session_id, backend: None,
        stop_transport_processing=lambda drain=True: None,
    )
    backend = PX4ScenarioBackend(
        manager,
        session_id="TS-PX4-CANCEL",
        csv_path=str(tmp_path / "session.csv"),
        sim_time=30,
        options={},
    )
    backend.prepare()
    backend._wait_for_px4 = wait_for_px4
    backend._spawn_process = (
        lambda command, **kwargs: spawn_calls.append(command)
    )
    backend._transition(state="starting", phase="starting_px4")
    with backend._startup_lock:
        backend._startup_generation += 1
        generation = backend._startup_generation
    startup_thread = threading.Thread(
        target=backend._start_pipeline_delayed,
        args=(generation,),
        daemon=True,
    )
    backend._startup_thread = startup_thread
    startup_thread.start()
    assert wait_entered.wait(1.0)

    threading.Timer(0.05, release_wait.set).start()
    backend.stop()
    startup_thread.join(timeout=1.0)

    assert spawn_calls == []
    assert backend.lifecycle().state == "stopped"


def test_receiver_timeout_marks_adapter_failed_and_notifies_manager(tmp_path) -> None:
    finished = []
    manager = SimpleNamespace(
        on_transport_lifecycle=lambda snapshot, session_id, backend: None,
        on_transport_finished=lambda session_id, backend: finished.append(
            (session_id, backend)
        ),
    )
    backend = _Backend(manager)

    class _TimeoutReceiver:
        exit_reason = None

        def start(self):
            raise TimeoutError("no telemetry")

    backend._receiver = _TimeoutReceiver()
    backend._receiver_worker()

    assert backend.lifecycle().state == "failed"
    assert backend.lifecycle().failure_reason == "receiver_timeout"
    assert finished == [("TS-LIFECYCLE", backend)]


def test_stop_receiver_joins_owned_thread_and_stops_processing() -> None:
    calls = []
    manager = SimpleNamespace(
        stop_transport_processing=lambda drain=True: calls.append(("pipeline", drain))
    )
    backend = _Backend(manager)

    class _Receiver:
        def stop(self):
            calls.append(("receiver", None))

    class _Thread:
        alive = True

        def is_alive(self):
            return self.alive

        def join(self, timeout):
            calls.append(("join", timeout))
            self.alive = False

    backend._receiver = _Receiver()
    backend._receiver_thread = _Thread()
    backend._processing_started = True

    backend._stop_receiver()

    assert calls == [("receiver", None), ("join", 10), ("pipeline", True)]
    assert backend._receiver is None
    assert backend._receiver_thread is None


def test_px4_flight_spawn_failure_rolls_back_started_collector(tmp_path) -> None:
    from uav_dashboard_api.services.telemetry_backends import PX4ScenarioBackend

    stopped = []
    manager = SimpleNamespace(
        app=None,
        _blind_test_active=False,
        _blind_test_ground_truth=None,
        resolve_telemetry_python_bin=lambda: "python-bin",
        on_transport_lifecycle=lambda snapshot, session_id, backend: None,
    )
    backend = PX4ScenarioBackend(
        manager,
        session_id="TS-PX4-ROLLBACK",
        csv_path=str(tmp_path / "session.csv"),
        sim_time=30,
        options={},
    )

    class _Collector:
        pid = 101

        def poll(self):
            return None

    spawn_count = 0

    def spawn(command, **kwargs):
        nonlocal spawn_count
        spawn_count += 1
        if spawn_count == 1:
            return _Collector()
        raise RuntimeError("flight spawn failed")

    backend._transition(state="starting", phase="starting_px4")
    backend._startup_generation = 1
    backend._wait_for_px4 = lambda **kwargs: True
    backend._repo_root = lambda: tmp_path
    backend._build_collector_cmd = lambda python_bin: [python_bin, "collector"]
    backend._spawn_process = spawn
    backend._stop_process = (
        lambda process, name: stopped.append(name) if process is not None else None
    )

    backend._start_pipeline_delayed(1)

    assert backend.lifecycle().state == "failed"
    assert backend.lifecycle().failure_reason == "px4_pipeline_failed"
    assert backend._collector_proc is None
    assert backend._flight_proc is None
    assert stopped == ["collector"]


def test_px4_receiver_construction_failure_stops_sitl(
    tmp_path, monkeypatch
) -> None:
    from uav_dashboard_api.services import telemetry_backends
    from uav_dashboard_api.services.telemetry_backends import PX4ScenarioBackend

    stopped = []
    manager = SimpleNamespace(
        app=None,
        on_transport_lifecycle=lambda snapshot, session_id, backend: None,
    )
    backend = PX4ScenarioBackend(
        manager,
        session_id="TS-PX4-RECEIVER-FAIL",
        csv_path=str(tmp_path / "session.csv"),
        sim_time=30,
        options={},
    )
    px4_process = SimpleNamespace(pid=101)
    backend._start_px4 = lambda: setattr(backend, "_px4_proc", px4_process)
    backend._stop_process = (
        lambda process, name: stopped.append((process, name))
        if process is not None
        else None
    )
    monkeypatch.setattr(
        telemetry_backends,
        "TelemetryReceiver",
        lambda **kwargs: (_ for _ in ()).throw(OSError("port busy")),
    )

    with pytest.raises(OSError, match="port busy"):
        backend._start()

    assert backend._px4_proc is None
    assert stopped == [(px4_process, "px4")]


def test_transport_backend_source_does_not_write_manager_lifecycle_fields() -> None:
    source_path = (
        Path(__file__).resolve().parents[1]
        / "uav_dashboard_api"
        / "services"
        / "telemetry_backends.py"
    )
    source = source_path.read_text(encoding="utf-8")

    assert "manager._startup_phase" not in source
    assert "manager._timeout_reason" not in source


def test_wait_for_px4_returns_false_when_socket_never_becomes_ready(monkeypatch):
    from uav_dashboard_api.services.telemetry_manager import telemetry_manager
    from uav_dashboard_api.services.telemetry_backends import PX4ScenarioBackend

    class _FakeProc:
        def poll(self):
            return None

    timeline = {"now": 0.0}

    def _fake_time():
        return timeline["now"]

    def _fake_sleep(seconds):
        timeline["now"] += seconds

    def _fake_connect(*args, **kwargs):
        raise ConnectionRefusedError

    backend = PX4ScenarioBackend(
        telemetry_manager,
        session_id="TS-PX4-WAIT-TIMEOUT",
        csv_path="session.csv",
        sim_time=30,
        options={},
    )
    backend._px4_proc = _FakeProc()
    backend._startup_cancelled = lambda generation: False
    monkeypatch.setattr("uav_dashboard_api.services.telemetry_backends.time.time", _fake_time)
    monkeypatch.setattr("uav_dashboard_api.services.telemetry_backends.threading.Event.wait", lambda self, seconds: _fake_sleep(seconds))
    monkeypatch.setattr(
        "uav_dashboard_api.services.telemetry_backends.socket.create_connection", _fake_connect
    )

    assert backend._wait_for_px4(timeout=20, generation=1) is False
