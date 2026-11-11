"""Exercise the real diagnosis/task orchestration with only inference replaced."""
from __future__ import annotations

import json
from pathlib import Path

import pytest
from flask import has_request_context

from uav_dashboard_api import db
from uav_dashboard_api.models.asset import Asset
from uav_dashboard_api.models.diagnosis_report import DiagnosisReport
from uav_dashboard_api.models.run_record import RunRecord
from uav_dashboard_api.models.telemetry_session import TelemetrySession
from uav_dashboard_api.services import task_manager
from uav_dashboard_api.services.offline_diagnosis import (
    OfflineDiagnosisRejected,
    submit_offline_diagnosis,
)
from uav_dashboard_api.utils.path_store import serialize_local_path


class DeferredExecutor:
    """Delay execution without replacing the task manager or its worker."""

    def __init__(self):
        self.jobs = []
        self.unavailable = False

    def submit(self, function, *args, **kwargs):
        if self.unavailable:
            raise RuntimeError("executor unavailable")
        self.jobs.append((function, args, kwargs))

    def drain(self):
        while self.jobs:
            function, args, kwargs = self.jobs.pop(0)
            function(*args, **kwargs)


@pytest.fixture
def deferred(monkeypatch):
    executor = DeferredExecutor()
    monkeypatch.setattr(task_manager, "get_executor", lambda app: executor)
    return executor


@pytest.fixture
def inference(app, monkeypatch):
    from uav_dashboard_api.services import offline_diagnosis

    state = {"calls": [], "error": None}

    def infer(*, csv_path, model_dir, output_json):
        assert not has_request_context()
        state["calls"].append((csv_path, model_dir, output_json))
        if state["error"]:
            raise state["error"]
        # Keep a missing CSV an execution-time failure, as with real inference.
        Path(csv_path).read_text(encoding="utf-8")
        result = {
            "prediction": {"label": "motor", "confidence": 0.91},
            "probs": {"normal": 0.05, "motor": 0.91, "sensor": 0.04},
            "labels": ["normal", "motor", "sensor"],
            "topk": [{"label": "motor", "prob": 0.91}],
            "debug": {"model_dir": model_dir, "model_name": "lstm"},
            "stdout_tail": "synthetic inference output",
            "stderr_tail": "",
        }
        Path(output_json).write_text(json.dumps(result), encoding="utf-8")
        return result

    monkeypatch.setattr(offline_diagnosis, "run_offline_multiclass", infer)
    return state


def make_source(app, kind="csv"):
    csv_path = Path(app.config["RUNTIME_DIR"]) / f"diagnosis_{kind}.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    csv_path.write_text(
        "time_s,motor1,motor2,motor3,motor4,fault_severity\n"
        "0,1,1,1,0,0.8\n1,1,1,1,0,0.8\n",
        encoding="utf-8",
    )
    if kind == "session":
        session = TelemetrySession(
            id="TS-DIAGNOSIS", session_type="simulation",
            csv_path=serialize_local_path(csv_path), status="completed",
        )
        db.session.add(session)
        db.session.commit()
        return {"session_id": session.id}, csv_path

    stored_path = csv_path if kind == "csv" else csv_path.with_suffix(".ulg")
    asset = Asset(
        filename=stored_path.name, kind=kind,
        stored_path=serialize_local_path(stored_path), sha256="a" * 64,
    )
    if kind == "ulg":
        asset.set_meta({"prepared_csv_path": str(csv_path)})
    db.session.add(asset)
    db.session.commit()
    return {"asset_id": asset.id}, csv_path


def submit(client, auth_header, source, **changes):
    return client.post(
        "/api/runs", headers=auth_header,
        json={"model": "lstm", **source, **changes},
    )


@pytest.mark.parametrize("kind", ["csv", "ulg", "session"])
def test_real_diagnosis_result_and_report_contract(
    app, client, auth_header, deferred, inference, kind,
):
    source, csv_path = make_source(app, kind)
    response = submit(client, auth_header, source)
    assert response.status_code == 202
    queued = response.get_json()["data"]
    assert queued["status"] == "queued"
    assert queued["progress"] == 0.0
    assert inference["calls"] == []

    deferred.drain()
    result_response = client.get(
        f"/api/runs/{queued['id']}/result", headers=auth_header,
    )
    assert result_response.status_code == 200
    result = result_response.get_json()["data"]
    assert set(result) == {
        "asset_id", "asset_kind", "input_csv", "model", "prediction", "probs",
        "labels", "topk", "severity", "recommendations", "recommendation_meta",
        "diagnosis", "unified_decision", "debug", "stdout_tail", "stderr_tail",
    }
    assert result["prediction"] == {"label": "motor", "confidence": 0.91}
    assert result["input_csv"] == serialize_local_path(csv_path)
    assert result["asset_kind"] == ("csv" if kind == "session" else kind)
    assert result["labels"] == ["normal", "motor", "sensor"]
    assert result["probs"] == {"normal": 0.05, "motor": 0.91, "sensor": 0.04}
    assert result["topk"] == [{"label": "motor", "prob": 0.91}]
    assert result["severity"]["level"] == "heavy"
    assert result["severity"]["method"] == "rule_v1_primary"
    assert result["severity"]["evidence"][0]["metric"] == "fault_severity_p95"
    assert result["recommendation_meta"]["fallback_used"] is False
    assert result["recommendations"][0]["priority"] == "P1"
    assert result["diagnosis"]["model"] == {
        "name": "lstm", "task": "offline_multiclass",
        "architecture": "cnn_lstm_v2_normalproto",
        "label_names": ["normal", "motor", "sensor"],
    }
    decision = result["unified_decision"]
    assert decision == result["diagnosis"]["decision"]
    assert decision["state"] == "DECIDED"
    assert decision["fault_label"] == "motor"
    assert decision["severity"] == "heavy"
    assert decision["episode_id"].startswith("ep_")
    assert decision["evidence_summary"]["source"] == "offline"
    assert result["stdout_tail"] == "synthetic inference output"
    assert result["stderr_tail"] == ""
    assert result["debug"]["model_dir"] == serialize_local_path(app.config["MODEL_ROOT"] + "/lstm")
    assert inference["calls"][0][0] == str(csv_path.resolve())
    assert Path(inference["calls"][0][2]).is_file()

    run = db.session.get(RunRecord, queued["id"])
    assert run.status == "success" and run.progress == 1.0
    assert run.started_at is not None and run.finished_at is not None
    assert DiagnosisReport.query.count() == 0
    report_response = client.post(
        "/api/reports/diagnosis", headers=auth_header, json={"run_id": run.id},
    )
    assert report_response.status_code == 201
    report = report_response.get_json()["data"]
    assert report["diagnosis"]["label"] == "motor"
    assert report["summary"]["decision"]["episode_id"] == decision["episode_id"]


@pytest.mark.parametrize("change", ["prepared_csv", "model", "asset"])
def test_queued_diagnosis_rechecks_inputs(
    app, client, auth_header, deferred, inference, change,
):
    source, csv_path = make_source(app, "ulg")
    response = submit(client, auth_header, source)
    assert response.status_code == 202
    run_id = response.get_json()["data"]["id"]
    if change == "prepared_csv":
        csv_path.unlink()
        expected_error = "asset_not_prepared"
    elif change == "model":
        config = Path(app.config["MODEL_ROOT"]) / "lstm" / "config.json"
        config.write_text("{}", encoding="utf-8")
        expected_error = "模型不存在: lstm"
    else:
        db.session.delete(db.session.get(Asset, source["asset_id"]))
        db.session.commit()
        expected_error = "资产不存在"
    deferred.drain()
    run = db.session.get(RunRecord, run_id)
    assert run.status == "error"
    assert run.error == expected_error
    assert run.finished_at is not None
    assert run.get_result() is None
    assert inference["calls"] == []


def test_inference_failure_is_recorded_by_real_task_manager(
    app, client, auth_header, deferred, inference,
):
    source, _ = make_source(app)
    inference["error"] = RuntimeError("synthetic inference failed")
    response = submit(client, auth_header, source)
    run_id = response.get_json()["data"]["id"]
    deferred.drain()
    run = db.session.get(RunRecord, run_id)
    assert run.status == "error" and run.get_result() is None
    assert run.error == "synthetic inference failed"
    assert "RuntimeError: synthetic inference failed" in run.error_detail
    assert DiagnosisReport.query.count() == 0


def test_executor_rejection_keeps_existing_error_record(
    app, client, auth_header, deferred, inference,
):
    source, _ = make_source(app)
    deferred.unavailable = True
    response = submit(client, auth_header, source)
    assert response.status_code == 500
    db.session.expire_all()
    run = RunRecord.query.one()
    assert run.status == "error"
    assert run.error == "任务提交失败: executor unavailable"
    assert inference["calls"] == []


def test_missing_plain_csv_remains_an_execution_time_failure(
    app, client, auth_header, deferred, inference,
):
    source, csv_path = make_source(app)
    csv_path.unlink()
    response = submit(client, auth_header, source)
    assert response.status_code == 202
    deferred.drain()
    run = db.session.get(RunRecord, response.get_json()["data"]["id"])
    assert run.status == "error" and run.get_result() is None
    assert "FileNotFoundError" in run.error_detail


def test_session_source_has_precedence_and_reuses_its_asset(
    app, client, auth_header, deferred, inference,
):
    asset_source, _ = make_source(app)
    session_source, session_csv = make_source(app, "session")
    first = submit(client, auth_header, {**asset_source, **session_source})
    second = submit(client, auth_header, session_source)
    assert first.status_code == second.status_code == 202
    first_run, second_run = first.get_json()["data"], second.get_json()["data"]
    assert first_run["asset_id"] != asset_source["asset_id"]
    assert first_run["asset_id"] == second_run["asset_id"]
    assert first_run["params"] == {
        "asset_id": first_run["asset_id"], "model": "lstm",
        "session_id": "TS-DIAGNOSIS", "source_type": "telemetry_session",
    }
    deferred.drain()
    assert all(call[0] == str(session_csv.resolve()) for call in inference["calls"])


def test_real_executor_runs_diagnosis_without_request_context(
    app, client, auth_header, inference,
):
    source, _ = make_source(app)
    try:
        response = submit(client, auth_header, source)
        assert response.status_code == 202
        task_manager.shutdown_executor(app, wait=True)
        # The request and background worker have different SQLAlchemy sessions.
        db.session.expire_all()
        run = db.session.get(RunRecord, response.get_json()["data"]["id"])
        assert run.status == "success"
        assert run.get_result()["diagnosis"]["label"] == "motor"
        assert len(inference["calls"]) == 1
    finally:
        task_manager.shutdown_executor(app, wait=True)


@pytest.mark.parametrize("case,status,message", [
    ("source_required", 400, "asset_id/session_id 和 model 为必填"),
    ("model_required", 400, "asset_id/session_id 和 model 为必填"),
    ("asset_missing", 404, "资产不存在"),
    ("unprepared", 409, "asset_not_prepared"),
    ("model_missing", 400, "模型不存在: unknown"),
])
def test_admission_rejection_preserves_http_contract(
    app, client, auth_header, deferred, case, status, message,
):
    source, _ = make_source(app, "ulg" if case == "unprepared" else "csv")
    payload = {"model": "lstm", **source}
    if case == "source_required":
        payload.pop("asset_id")
    elif case == "model_required":
        payload["model"] = ""
    elif case == "asset_missing":
        payload["asset_id"] = 9999
    elif case == "unprepared":
        asset = db.session.get(Asset, source["asset_id"])
        asset.set_meta({})
        db.session.commit()
    else:
        payload["model"] = "unknown"
    response = client.post("/api/runs", headers=auth_header, json=payload)
    assert response.status_code == status
    expected = {"success": False, "message": message}
    if case == "unprepared":
        expected["prepare_endpoint"] = f"/api/assets/{source['asset_id']}/prepare"
    assert response.get_json() == expected
    assert RunRecord.query.count() == 0
    assert deferred.jobs == []


def test_rejected_session_handoff_rolls_back_its_uncommitted_asset(app, deferred):
    source, _ = make_source(app, "session")
    with pytest.raises(OfflineDiagnosisRejected, match="模型不存在"):
        submit_offline_diagnosis(model_name="unknown", created_by=None, **source)
    # A later operation must not accidentally commit the rejected handoff.
    db.session.commit()
    assert Asset.query.count() == RunRecord.query.count() == 0
    assert TelemetrySession.query.count() == 1
    assert deferred.jobs == []


def test_public_submission_needs_only_application_context(app, deferred, inference):
    assert not has_request_context()
    source, _ = make_source(app)
    queued = submit_offline_diagnosis(model_name="lstm", created_by=None, **source)
    run_id = queued.id
    assert queued.status == "queued"
    deferred.drain()
    db.session.expire_all()
    run = db.session.get(RunRecord, run_id)
    assert run.status == "success"
    assert run.get_result()["unified_decision"]["fault_label"] == "motor"


def test_session_asset_and_run_rollback_together_when_commit_fails(
    app, deferred, monkeypatch,
):
    source, _ = make_source(app, "session")

    def fail_commit():
        raise RuntimeError("synthetic database failure")

    monkeypatch.setattr(db.session, "commit", fail_commit)
    with pytest.raises(RuntimeError, match="synthetic database failure"):
        submit_offline_diagnosis(model_name="lstm", created_by=None, **source)
    assert Asset.query.count() == RunRecord.query.count() == 0
    assert deferred.jobs == []
