from __future__ import annotations

from types import SimpleNamespace

from uav_fault.online.event_aggregator import AggregatedEvent
from uav_fault.online.monitoring_session import OnlineMonitoringSession


class _Runner:
    window_size = 1
    aggregator = SimpleNamespace(onset_th=0.6)

    def __init__(self, results=None, flushed=None):
        self.results = list(results or [])
        self.flushed = list(flushed or [])

    def update(self, *, frame, window, frame_idx):
        return self.results.pop(0)

    def flush(self):
        return list(self.flushed)


def _event(start_t=1.0, end_t=2.0):
    return AggregatedEvent(
        start_t=start_t,
        end_t=end_t,
        start_frame_idx=50,
        end_frame_idx=100,
        n_points=4,
        max_fault_prob=0.9,
        mean_fault_prob=0.8,
        meta={},
    )


def test_monitoring_session_tracks_preview_confirmation_and_lead_time():
    session = OnlineMonitoringSession()
    session.start_detection(_Runner(), model_summary={"name": "sim-model"})

    preview = session.ingest_external(
        result={"label": "fault", "confidence": 0.8, "fault_prob": 0.8},
        new_events=[],
        frame_idx=50,
        time_s=1.0,
    )
    event = _event()
    confirmed = session.ingest_external(
        result={"label": "fault", "confidence": 0.9, "fault_prob": 0.9},
        new_events=[event],
        frame_idx=100,
        time_s=2.0,
    )

    assert preview.snapshot.preview_active is True
    assert confirmed.snapshot.preview_active is False
    assert confirmed.snapshot.last_preview_lead_time_s == 1.0
    assert confirmed.snapshot.warning_stage == "critical"
    assert event.lead_time_s == 1.0


def test_monitoring_session_rejects_stale_async_generation():
    session = OnlineMonitoringSession()
    old_generation = session.start_detection(_Runner()).generation
    session.stop_detection()
    session.start_detection(_Runner())

    update = session.ingest_external(
        result={"label": "fault", "confidence": 0.9, "fault_prob": 0.9},
        new_events=[],
        frame_idx=1,
        time_s=1.0,
        generation=old_generation,
    )

    assert update.stale is True
    assert update.snapshot.current_fault_prob == 0.0


def test_monitoring_session_local_runner_and_stop_flush_share_state():
    final_event = _event(1.0, 1.5)
    runner = _Runner(
        results=[({"label": "normal", "confidence": 0.9, "fault_prob": 0.1}, [])],
        flushed=[final_event],
    )
    session = OnlineMonitoringSession()
    generation = session.start_detection(runner).generation

    update = session.ingest(
        frame={"time_s": 1.0},
        window=[{"time_s": 1.0}],
        frame_idx=1,
        generation=generation,
    )
    stopped = session.stop_detection()

    assert update.stale is False
    assert update.snapshot.warning_stage == "normal"
    assert stopped.new_events == (final_event,)
    assert stopped.snapshot.state == "stopped"


def test_new_collection_clears_previous_risk_state():
    session = OnlineMonitoringSession()
    session.start_detection(_Runner())
    session.ingest_external(
        result={"label": "fault", "confidence": 0.9, "fault_prob": 0.9},
        new_events=[],
        frame_idx=1,
        time_s=1.0,
    )

    snapshot = session.reset_collection()

    assert snapshot.state == "idle"
    assert snapshot.current_fault_prob == 0.0
    assert snapshot.risk_score == 0.0
    assert snapshot.preview_active is False
