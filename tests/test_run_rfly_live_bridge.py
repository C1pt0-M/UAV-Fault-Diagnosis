import sys
import threading
import types

from rfly_bridge import run_rfly_live_bridge

def test_runtime_start_starts_attach_session_and_optional_pymav():
    events = []

    attach_session = types.SimpleNamespace(
        start=lambda: events.append("attach_start") or object(),
        stop=lambda: events.append("attach_stop"),
    )
    pymav = types.SimpleNamespace(
        start=lambda: events.append("pymav_start"),
        stop=lambda: events.append("pymav_stop"),
    )
    emitter = types.SimpleNamespace(
        run=lambda stop_event: events.append(("emitter_run", stop_event.is_set()))
    )

    runtime = run_rfly_live_bridge.RflyBridgeRuntime(
        attach_session=attach_session,
        emitter=emitter,
        pymav_source=pymav,
    )
    runtime.start()

    assert events[:2] == ["attach_start", "pymav_start"]
    assert runtime._thread is not None


def test_runtime_stop_stops_pymav_before_attach():
    events = []
    done = threading.Event()

    attach_session = types.SimpleNamespace(
        start=lambda: object(),
        stop=lambda: events.append("attach_stop"),
    )
    pymav = types.SimpleNamespace(
        start=lambda: None,
        stop=lambda: events.append("pymav_stop"),
    )

    def _run(stop_event):
        done.set()
        stop_event.wait(0.01)

    emitter = types.SimpleNamespace(run=_run)

    runtime = run_rfly_live_bridge.RflyBridgeRuntime(
        attach_session=attach_session,
        emitter=emitter,
        pymav_source=pymav,
    )
    runtime.start()
    done.wait(0.5)
    runtime.stop()

    assert events == ["pymav_stop", "attach_stop"]


def test_parse_args_accepts_duration(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_rfly_live_bridge.py",
            "--px4mavctrl",
            "PX4MavCtrlV4.py",
            "--duration",
            "3.5",
        ],
    )

    args = run_rfly_live_bridge.parse_args()

    assert args.duration == 3.5
