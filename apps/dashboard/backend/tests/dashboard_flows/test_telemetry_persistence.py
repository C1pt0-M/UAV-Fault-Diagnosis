import csv
import io
import json
import sys
import time
import types
from datetime import datetime
from pathlib import Path

import pytest
from flask import Flask

from uav_dashboard_api import db
from uav_dashboard_api.models.asset import Asset
from uav_dashboard_api.models.run_record import RunRecord
from uav_dashboard_api.models.telemetry_session import TelemetryFaultEvent, TelemetrySession
from uav_dashboard_api.utils.path_store import resolve_local_path

def test_telemetry_replay_session_auto_finalize_persists_runtime_fields(
    app, auth_header, tmp_path: Path
):
    from uav_dashboard_api.services.telemetry_manager import get_telemetry_manager
    telemetry_manager = get_telemetry_manager()

    replay_path = tmp_path / "replay_auto_finalize.csv"
    replay_path.write_text(
        "time_s,roll,pitch,yaw,is_fault_active\n"
        "0.0,1.0,0.0,0.0,0\n"
        "0.1,1.5,0.1,0.0,1\n"
        "0.2,1.8,0.1,0.0,1\n",
        encoding="utf-8",
    )

    class _FakeThread:
        _deferred = []
        _active_replay = None

        def __init__(self, target=None, name=None, daemon=None, args=(), kwargs=None):
            self._target = target
            self._args = args
            self._kwargs = kwargs or {}
            self._name = name
            self._alive = False

        def _run_now(self):
            self._alive = True
            is_replay = self._name == "telemetry-replay"
            if is_replay:
                _FakeThread._active_replay = self
            try:
                if self._target:
                    self._target(*self._args, **self._kwargs)
            finally:
                self._alive = False
                if is_replay:
                    _FakeThread._active_replay = None
                    deferred = list(_FakeThread._deferred)
                    _FakeThread._deferred.clear()
                    for thread in deferred:
                        thread._run_now()

        def start(self):
            if (
                getattr(self._target, "__name__", None) == "_auto_finalize"
                and _FakeThread._active_replay is not None
            ):
                _FakeThread._deferred.append(self)
                return
            self._run_now()

        def is_alive(self):
            return self._alive

        def join(self, timeout=None):
            return None

    original_backend = telemetry_manager._backend
    original_session_id = telemetry_manager._session_id
    original_start_time = telemetry_manager._start_time
    original_planned_duration = telemetry_manager._planned_duration_s
    original_frame_count = telemetry_manager._frame_count
    original_fault_events = telemetry_manager._fault_events
    original_detecting = telemetry_manager._detecting
    original_active_model = telemetry_manager._active_model
    original_backend_name = telemetry_manager._backend_name
    original_backend_options = telemetry_manager._backend_options
    original_schema_version = telemetry_manager._schema_version
    original_feature_coverage = telemetry_manager._feature_coverage
    original_replay_metrics = telemetry_manager._replay_metrics

    try:
        telemetry_manager._backend = None
        telemetry_manager._session_id = None
        telemetry_manager._start_time = None
        telemetry_manager._planned_duration_s = None
        telemetry_manager._frame_count = 0
        telemetry_manager._fault_events = []
        telemetry_manager._detecting = False
        telemetry_manager._active_model = None
        telemetry_manager._backend_name = None
        telemetry_manager._backend_options = {}
        telemetry_manager._schema_version = "rfly_seq_v1"
        telemetry_manager._feature_coverage = None
        telemetry_manager._replay_metrics = None

        with pytest.MonkeyPatch.context() as mp:
            mp.setattr("uav_dashboard_api.services.telemetry_backends.threading.Thread", _FakeThread)
            mp.setattr("uav_dashboard_api.services.telemetry_manager.threading.Thread", _FakeThread)
            mp.setattr(
                "uav_dashboard_api.services.telemetry_backends.time.sleep",
                lambda *_args, **_kwargs: None,
            )
            mp.setattr(
                "uav_dashboard_api.services.telemetry_manager.time.sleep",
                lambda *_args, **_kwargs: None,
            )

            result = telemetry_manager.start_collection(
                drone_id="DRONE-REPLAY-INT-001",
                session_type="simulation",
                sim_time=3,
                backend="replay",
                replay_path=str(replay_path),
            )

        assert result["scenario_backend"] == "replay"
        assert result["replay_path"] == str(replay_path)

        deadline = time.time() + 1.0
        session = None
        while time.time() < deadline:
            with app.app_context():
                session = db.session.get(TelemetrySession, result["session_id"])
                if session is not None and session.status == "completed":
                    break
            time.sleep(0.01)

        assert session is not None
        assert session.drone_id == "DRONE-REPLAY-INT-001"
        assert session.session_type == "simulation"
        assert session.scenario_backend == "replay"
        assert session.schema_version == "rfly_seq_v1"
        assert session.source_path == str(replay_path)
        assert session.status == "completed"
        assert session.frame_count == 3
        assert session.csv_path == result["csv_path"]
        assert Path(resolve_local_path(session.csv_path, app.config)).exists()
        assert session.duration_seconds is not None
        assert session.duration_seconds >= 0
        assert session.fault_events_count == 0
        assert session.get_meta()["backend"] == "replay"
        assert session.get_meta()["replay_metrics"]["gt_has_labels"] == 1
        assert session.get_meta()["replay_metrics"]["gt_event_count"] == 1
        assert session.get_meta()["replay_metrics"]["pred_event_count"] == 0
    finally:
        telemetry_manager._backend = original_backend
        telemetry_manager._session_id = original_session_id
        telemetry_manager._start_time = original_start_time
        telemetry_manager._planned_duration_s = original_planned_duration
        telemetry_manager._frame_count = original_frame_count
        telemetry_manager._fault_events = original_fault_events
        telemetry_manager._detecting = original_detecting
        telemetry_manager._active_model = original_active_model
        telemetry_manager._backend_name = original_backend_name
        telemetry_manager._backend_options = original_backend_options
        telemetry_manager._schema_version = original_schema_version
        telemetry_manager._feature_coverage = original_feature_coverage
        telemetry_manager._replay_metrics = original_replay_metrics


def test_telemetry_stop_collection_flushes_detection_events_into_finalize(
    app, monkeypatch, tmp_path: Path
):
    from uav_dashboard_api.services.telemetry_manager import get_telemetry_manager
    telemetry_manager = get_telemetry_manager()

    replay_path = tmp_path / "replay_stop_detection.csv"
    replay_path.write_text(
        "time_s,roll,is_fault_active\n0.0,0.1,0\n0.1,0.2,1\n0.2,0.3,1\n",
        encoding="utf-8",
    )

    class _StoppedThread:
        def is_alive(self):
            return False

    class _FakeEvent:
        start_t = 0.1
        end_t = 0.2
        max_fault_prob = 0.97
        mean_fault_prob = 0.91
        n_points = 2
        meta = {'timing': {
            'confirmed_t': 0.15, 'emitted_t': 0.2,
            'confirmed_at_unix_s': 1000.0, 'emitted_at_unix_s': 1001.0,
            'emission_reason': 'flush',
        }}

    class _FakeDetector:
        window_size = 150

        def flush(self):
            return [_FakeEvent()]

    original_backend = telemetry_manager._backend
    original_session_id = telemetry_manager._session_id
    original_start_time = telemetry_manager._start_time
    original_planned_duration = telemetry_manager._planned_duration_s
    original_frame_count = telemetry_manager._frame_count
    original_fault_events = telemetry_manager._fault_events
    original_detecting = telemetry_manager._detecting
    original_active_model = telemetry_manager._active_model
    original_backend_name = telemetry_manager._backend_name
    original_backend_options = telemetry_manager._backend_options
    original_schema_version = telemetry_manager._schema_version
    original_feature_coverage = telemetry_manager._feature_coverage
    original_replay_metrics = telemetry_manager._replay_metrics
    original_detector = telemetry_manager._detector

    def fake_start(self):
        self._thread = _StoppedThread()

    monkeypatch.setattr(
        "uav_dashboard_api.services.telemetry_manager.ReplayScenarioBackend.start", fake_start
    )

    try:
        result = telemetry_manager.start_collection(
            drone_id="DRONE-REPLAY-FLUSH-001",
            session_type="simulation",
            sim_time=30,
            backend="replay",
            replay_path=str(replay_path),
        )
        Path(resolve_local_path(result["csv_path"], app.config)).write_text(
            replay_path.read_text(encoding="utf-8"), encoding="utf-8"
        )
        telemetry_manager._detector = _FakeDetector()
        telemetry_manager._detecting = True
        telemetry_manager._active_model = {"name": "online_detector_alpha"}

        stop_result = telemetry_manager.stop_collection()

        assert stop_result["session_id"] == result["session_id"]
        assert stop_result["fault_events_count"] == 1
        with app.app_context():
            session = db.session.get(TelemetrySession, result["session_id"])
            assert session is not None
            assert session.status == "completed"
            assert session.fault_events_count == 1
            events = TelemetryFaultEvent.query.filter_by(
                session_id=result["session_id"]
            ).all()
            assert len(events) == 1
            assert events[0].max_fault_prob == pytest.approx(0.97)
            assert events[0].get_diagnosis()["label"] == "fault"
            assert events[0].get_diagnosis()['event_timing'] == _FakeEvent.meta['timing']
            saved = events[0].to_dict()
            assert saved['event_timing'] == _FakeEvent.meta['timing']
            assert saved['start_time'] == 0.1
            assert saved['end_time'] == 0.2
            replay_metrics = session.get_meta()["replay_metrics"]
            assert replay_metrics["pred_event_count"] == 1
            assert replay_metrics["hit"] == 1
    finally:
        telemetry_manager._backend = original_backend
        telemetry_manager._session_id = original_session_id
        telemetry_manager._start_time = original_start_time
        telemetry_manager._planned_duration_s = original_planned_duration
        telemetry_manager._frame_count = original_frame_count
        telemetry_manager._fault_events = original_fault_events
        telemetry_manager._detecting = original_detecting
        telemetry_manager._active_model = original_active_model
        telemetry_manager._backend_name = original_backend_name
        telemetry_manager._backend_options = original_backend_options
        telemetry_manager._schema_version = original_schema_version
        telemetry_manager._feature_coverage = original_feature_coverage
        telemetry_manager._replay_metrics = original_replay_metrics
        telemetry_manager._detector = original_detector


def test_telemetry_record_fault_event_delegates_persistence_writes(app, monkeypatch):
    from uav_dashboard_api.services.telemetry_manager import get_telemetry_manager
    telemetry_manager = get_telemetry_manager()

    class _FakeEvent:
        start_t = 0.1
        end_t = 0.2
        max_fault_prob = 0.97
        mean_fault_prob = 0.91
        n_points = 2

    original_session_id = telemetry_manager._session_id
    original_fault_events = telemetry_manager._fault_events

    captured = {}

    try:
        telemetry_manager._session_id = "TS-PERSIST-001"
        telemetry_manager._fault_events = []

        def fake_record_fault_event(*, session_id, event, diagnosis, legacy_severity):
            captured["session_id"] = session_id
            captured["event"] = event
            captured["diagnosis"] = diagnosis
            captured["legacy_severity"] = legacy_severity

        monkeypatch.setattr(
            telemetry_manager._persistence,
            "record_fault_event",
            fake_record_fault_event,
        )
        telemetry_manager._record_fault_event(_FakeEvent())

        assert len(telemetry_manager._fault_events) == 1
        assert captured["session_id"] == "TS-PERSIST-001"
        assert captured["event"].max_fault_prob == pytest.approx(0.97)
        assert captured["diagnosis"]["label"] == "fault"
        assert captured["diagnosis"]["severity"] == "heavy"
        assert captured["legacy_severity"] == "critical"
    finally:
        telemetry_manager._session_id = original_session_id
        telemetry_manager._fault_events = original_fault_events


def test_telemetry_finalize_session_delegates_persistence_writes(app, monkeypatch):
    from uav_dashboard_api.services.telemetry_manager import get_telemetry_manager
    telemetry_manager = get_telemetry_manager()

    captured = {}
    replay_metrics = {"pred_event_count": 1, "hit": 1}
    original_fault_prob_curve = telemetry_manager._fault_prob_curve
    telemetry_manager._fault_prob_curve = [
        {"time_s": 0.1, "fault_prob": 0.8},
    ]

    def fake_finalize_session(
        *,
        session_id,
        duration,
        frame_count,
        fault_count,
        replay_metrics,
        fault_prob_curve,
        mec=None,
        inference=None,
        session_meta=None,
    ):
        captured["session_id"] = session_id
        captured["duration"] = duration
        captured["frame_count"] = frame_count
        captured["fault_count"] = fault_count
        captured["replay_metrics"] = replay_metrics
        captured["fault_prob_curve"] = fault_prob_curve
        captured["mec"] = mec
        captured["inference"] = inference
        captured["session_meta"] = session_meta

    monkeypatch.setattr(
        telemetry_manager._persistence, "finalize_session", fake_finalize_session
    )
    try:
        telemetry_manager._finalize_session("TS-PERSIST-002", 12.3, 45, 2, replay_metrics)
    finally:
        telemetry_manager._fault_prob_curve = original_fault_prob_curve

    assert captured["session_id"] == "TS-PERSIST-002"
    assert captured["duration"] == 12.3
    assert captured["frame_count"] == 45
    assert captured["fault_count"] == 2
    assert captured["replay_metrics"] == replay_metrics
    assert captured["fault_prob_curve"] == [{"time_s": 0.1, "fault_prob": 0.8}]
    assert captured["mec"] == telemetry_manager._mec_status
    assert captured["inference"] == telemetry_manager._inference_status
    assert captured["session_meta"] is None


def test_telemetry_finalize_session_skips_replay_asset_registration(app, tmp_path: Path):
    from uav_dashboard_api.services.telemetry_manager import get_telemetry_manager
    telemetry_manager = get_telemetry_manager()

    csv_path = tmp_path / "session_asset.csv"
    csv_path.write_text("time_s,motor1,fault_prob\n0.0,1000,0.2\n", encoding="utf-8")

    with app.app_context():
        session = TelemetrySession(
            id="TS-ASSET-001",
            session_type="simulation",
            scenario_backend="replay",
            csv_path=str(csv_path),
            has_detection=True,
            status="running",
        )
        db.session.add(session)
        db.session.commit()

    telemetry_manager._persistence.finalize_session(
        session_id="TS-ASSET-001",
        duration=1.2,
        frame_count=1,
        fault_count=0,
        fault_prob_curve=[{"time_s": 0.0, "fault_prob": 0.2}],
        inference={
            "location": "center",
            "avg_inference_ms": 7.2,
            "p95_inference_ms": 9.8,
        },
    )

    with app.app_context():
        asset = Asset.query.filter_by(sha256="telemetry-session-TS-ASSET-001").first()
        assert asset is None
        session = db.session.get(TelemetrySession, "TS-ASSET-001")
        assert session.get_meta()["fault_prob_curve_points"] == 1
        assert session.get_meta()["inference_location"] == "center"
        assert session.get_meta()["inference"]["avg_inference_ms"] == 7.2
        assert session.get_meta()["inference"]["p95_inference_ms"] == 9.8


def test_telemetry_finalize_session_registers_non_replay_session_asset(
    app, tmp_path: Path
):
    from uav_dashboard_api.services.telemetry_manager import get_telemetry_manager
    telemetry_manager = get_telemetry_manager()

    csv_path = tmp_path / "px4_session_asset.csv"
    csv_path.write_text("time_s,motor1,fault_prob\n0.0,1000,0.2\n", encoding="utf-8")

    with app.app_context():
        session = TelemetrySession(
            id="TS-ASSET-PX4-001",
            session_type="simulation",
            scenario_backend="px4",
            csv_path=str(csv_path),
            has_detection=True,
            status="running",
        )
        db.session.add(session)
        db.session.commit()

    telemetry_manager._persistence.finalize_session(
        session_id="TS-ASSET-PX4-001",
        duration=1.2,
        frame_count=1,
        fault_count=0,
        fault_prob_curve=[{"time_s": 0.0, "fault_prob": 0.2}],
    )

    with app.app_context():
        asset = Asset.query.filter_by(sha256="telemetry-session-TS-ASSET-PX4-001").first()
        assert asset is not None
        assert asset.filename == "px4_session_asset.csv"
        assert asset.get_meta()["source_type"] == "telemetry_session"
        assert asset.get_meta()["session_id"] == "TS-ASSET-PX4-001"
        session = db.session.get(TelemetrySession, "TS-ASSET-PX4-001")
        assert session.get_meta()["fault_prob_curve_points"] == 1


def test_telemetry_create_session_record_delegates_persistence_writes(app, monkeypatch):
    from uav_dashboard_api.services.telemetry_manager import get_telemetry_manager
    telemetry_manager = get_telemetry_manager()

    captured = {}
    meta = {"ground_truth": {"fault_type": "none"}}

    def fake_create_session_record(
        *,
        session_id,
        drone_id,
        session_type,
        csv_path,
        backend_name,
        schema_version,
        source_path,
        meta,
        created_by=None,
    ):
        captured["session_id"] = session_id
        captured["drone_id"] = drone_id
        captured["session_type"] = session_type
        captured["csv_path"] = csv_path
        captured["backend_name"] = backend_name
        captured["schema_version"] = schema_version
        captured["source_path"] = source_path
        captured["meta"] = meta
        captured["created_by"] = created_by

    original_schema_version = telemetry_manager._schema_version
    try:
        telemetry_manager._schema_version = "canonical-v-test"
        monkeypatch.setattr(
            telemetry_manager._persistence,
            "create_session_record",
            fake_create_session_record,
        )
        telemetry_manager._create_session_record(
            "TS-PERSIST-003",
            "DRONE-PERSIST-003",
            "simulation",
            "/tmp/session.csv",
            "replay",
            source_path="/tmp/replay.csv",
            meta=meta,
        )
    finally:
        telemetry_manager._schema_version = original_schema_version

    assert captured == {
        "session_id": "TS-PERSIST-003",
        "drone_id": "DRONE-PERSIST-003",
        "session_type": "simulation",
        "csv_path": "/tmp/session.csv",
        "backend_name": "replay",
        "schema_version": "canonical-v-test",
        "source_path": "/tmp/replay.csv",
        "meta": meta,
        "created_by": None,
    }


def test_telemetry_delete_session_record_delegates_persistence_writes(app, monkeypatch):
    from uav_dashboard_api.services.telemetry_manager import get_telemetry_manager
    telemetry_manager = get_telemetry_manager()

    captured = {}

    def fake_delete_session_record(*, session_id):
        captured["session_id"] = session_id

    monkeypatch.setattr(
        telemetry_manager._persistence,
        "delete_session_record",
        fake_delete_session_record,
    )
    telemetry_manager._delete_session_record("TS-PERSIST-004")

    assert captured == {"session_id": "TS-PERSIST-004"}


def test_telemetry_update_session_detection_delegates_persistence_writes(
    app, monkeypatch
):
    from uav_dashboard_api.services.telemetry_manager import get_telemetry_manager
    telemetry_manager = get_telemetry_manager()

    captured = {}

    def fake_update_session_detection(*, session_id, has_detection):
        captured["session_id"] = session_id
        captured["has_detection"] = has_detection

    monkeypatch.setattr(
        telemetry_manager._persistence,
        "update_session_detection",
        fake_update_session_detection,
    )
    telemetry_manager._update_session_detection("TS-PERSIST-005", True)

    assert captured == {"session_id": "TS-PERSIST-005", "has_detection": True}


def test_telemetry_write_log_delegates_persistence_writes(app, monkeypatch):
    from uav_dashboard_api.services.telemetry_manager import get_telemetry_manager
    telemetry_manager = get_telemetry_manager()

    captured = {}

    def fake_write_log(*, drone_id, log_type, content):
        captured["drone_id"] = drone_id
        captured["log_type"] = log_type
        captured["content"] = content

    monkeypatch.setattr(telemetry_manager._persistence, "write_log", fake_write_log)
    telemetry_manager._write_log("DRONE-PERSIST-006", "telemetry", "hello")

    assert captured == {
        "drone_id": "DRONE-PERSIST-006",
        "log_type": "telemetry",
        "content": "hello",
    }
