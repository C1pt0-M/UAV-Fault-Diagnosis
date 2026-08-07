from __future__ import annotations

from pathlib import Path

import pandas as pd

from uav_dashboard_api.services.telemetry_backends import ReplayScenarioBackend


class _FakeManager:
    def __init__(self):
        self.processed_frames = 0
        self.finished = None

    def process_transport_frame(self, frame, window, source_backend=None):
        self.processed_frames += 1

    def on_transport_finished(self, session_id, backend):
        self.finished = (session_id, backend)


def _write_source(path: Path, rows: int) -> None:
    path.write_text(
        "time_s,seq,roll\n" + "".join(f"0,{i},0.1\n" for i in range(rows)),
        encoding="utf-8",
    )


def _make_backend(tmp_path: Path, manager, rows: int, *, stop_after: int | None = None):
    src = tmp_path / "source.csv"
    _write_source(src, rows)
    out = tmp_path / "session.csv"
    backend = ReplayScenarioBackend(
        manager,
        session_id="TS-TRUNC",
        csv_path=str(out),
        sim_time=0.0,
        options={"replay_path": str(src)},
    )

    if stop_after is not None:
        count = 0

        def process(frame, window, source_backend=None):
            nonlocal count
            count += 1
            if count >= stop_after:
                backend._stop_event.set()
            manager.processed_frames += 1

        manager.process_transport_frame = process
    return backend, out


def test_replay_stopped_early_writes_truncated_session_csv(tmp_path):
    manager = _FakeManager()
    backend, out = _make_backend(tmp_path, manager, rows=10, stop_after=3)

    backend._run()

    assert backend.replay_completed is False
    assert backend.session_meta() == {"replay_completed": False}
    df = pd.read_csv(out)
    assert len(df) == 3
    assert manager.finished == ("TS-TRUNC", backend)


def test_replay_completed_writes_full_session_csv(tmp_path):
    manager = _FakeManager()
    backend, out = _make_backend(tmp_path, manager, rows=10, stop_after=None)

    backend._run()

    assert backend.replay_completed is True
    assert backend.session_meta() == {"replay_completed": True}
    df = pd.read_csv(out)
    assert len(df) == 10
