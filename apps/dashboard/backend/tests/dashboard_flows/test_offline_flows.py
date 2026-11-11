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

def test_asset_upload_uses_safe_storage_name_and_preserves_display_name(
    app,
    client,
    auth_header,
):
    upload = client.post(
        "/api/assets/upload",
        headers=auth_header,
        data={"file": (io.BytesIO(b"time_s,roll\n0,0\n"), "飞行 记录.csv")},
    )

    assert upload.status_code == 201
    payload = upload.get_json()["data"]
    assert payload["filename"] == "飞行 记录.csv"
    with app.app_context():
        asset = db.session.get(Asset, payload["id"])
        stored_path = Path(resolve_local_path(asset.stored_path, app.config))
        assert stored_path.parent == Path(app.config["UPLOAD_DIR"]) / "csv"
        assert stored_path.suffix == ".csv"
        assert "/" not in stored_path.name
        assert "\\" not in stored_path.name


def test_asset_upload_cleans_file_when_database_commit_fails(
    app,
    client,
    auth_header,
    monkeypatch,
):
    from uav_dashboard_api.routes import dashboard_assets

    def fail_commit():
        raise RuntimeError("database unavailable")

    monkeypatch.setattr(dashboard_assets.db.session, "commit", fail_commit)
    response = client.post(
        "/api/assets/upload",
        headers=auth_header,
        data={"file": (io.BytesIO(b"time_s,roll\n0,0\n"), "failed.csv")},
    )

    assert response.status_code == 500
    upload_dir = Path(app.config["UPLOAD_DIR"]) / "csv"
    assert list(upload_dir.glob("*failed.csv")) == []


def test_offline_prepare_run_report_main_flow(app, client, auth_header, monkeypatch):
    from uav_dashboard_api.routes import dashboard_assets
    from uav_dashboard_api.services import offline_diagnosis, task_manager

    class ImmediateExecutor:
        def submit(self, function, *args, **kwargs):
            function(*args, **kwargs)

    def prepare(ulg_path, output_csv):
        Path(output_csv).write_text(
            "timestamp,motor1,battery_voltage,fault_severity\n0,0.1,11.1,0.88\n",
            encoding="utf-8",
        )
        return {"output_csv": output_csv}

    def infer(*, csv_path, model_dir, output_json):
        return {
            "prediction": {"label": "motor", "confidence": 0.91},
            "probs": {"motor": 0.91, "normal": 0.09},
        }

    monkeypatch.setattr(task_manager, "get_executor", lambda app: ImmediateExecutor())
    monkeypatch.setattr(dashboard_assets, "run_ulg_to_csv", prepare)
    monkeypatch.setattr(offline_diagnosis, "run_offline_multiclass", infer)
    upload = client.post(
        "/api/assets/upload",
        headers=auth_header,
        data={"file": (io.BytesIO(b"ULGDATA"), "mission.ulg")},
    )
    asset_id = upload.get_json()["data"]["id"]
    assert upload.status_code == 201
    prepared = client.post(f"/api/assets/{asset_id}/prepare", headers=auth_header)
    assert prepared.status_code == 202
    db.session.expire_all()
    prepare_run = db.session.get(RunRecord, prepared.get_json()["data"]["id"])
    assert prepare_run.status == "success", prepare_run.error
    asset_detail = client.get(
        f"/api/assets/{asset_id}", headers=auth_header
    ).get_json()["data"]
    assert asset_detail["meta"]["prepared_csv_path"]

    run_resp = client.post(
        "/api/runs",
        headers=auth_header,
        json={"task": "offline_multiclass", "asset_id": asset_id, "model": "lstm"},
    )
    run_id = run_resp.get_json()["data"]["id"]
    assert run_resp.status_code == 202
    assert run_resp.get_json()["data"]["task"] == "offline_multiclass"
    result = client.get(f"/api/runs/{run_id}/result", headers=auth_header).get_json()[
        "data"
    ]
    assert result["diagnosis"]["label"] == "motor"
    assert result["diagnosis"]["severity"] == "heavy"
    assert result["unified_decision"]["state"] == "DECIDED"
    assert result["diagnosis"]["decision"]["fault_label"] == "motor"
    assert result["unified_decision"]["episode_id"].startswith("ep_")
    assert result["unified_decision"]["evidence_summary"]["source"] == "offline"

    report = client.post(
        "/api/reports/diagnosis", headers=auth_header, json={"run_id": run_id}
    )
    report_id = report.get_json()["data"]["id"]
    assert report.status_code == 201
    report_detail = client.get(
        f"/api/reports/diagnosis/{report_id}", headers=auth_header
    ).get_json()["data"]
    assert report_detail["diagnosis"]["label"] == "motor"
    assert report_detail["diagnosis"]["severity"] == "heavy"
    assert report_detail["diagnosis"]["model"]["task"] == "offline_multiclass"
    assert report_detail["summary"]["decision"]["state"] == "DECIDED"
    assert report_detail["summary"]["decision"]["episode_id"] == result["unified_decision"]["episode_id"]
    assert report_detail["summary"]["data_quality"]["level"] in {
        "good",
        "warning",
        "poor",
        "unknown",
    }
    assert report_detail["summary"]["traceability"]["run_id"] == run_id
    assert report_detail["summary"]["report_confidence"]["level"] in {
        "high",
        "medium",
        "low",
        "unknown",
    }
    assert report_detail["summary"]["review"]["status"] == "pending"

    run_detail = client.get(f"/api/runs/{run_id}", headers=auth_header).get_json()[
        "data"
    ]
    assert run_detail["report_id"] == report_id

    listed_runs = client.get(
        "/api/runs?task=offline_multiclass&limit=5", headers=auth_header
    ).get_json()["data"]
    assert listed_runs[0]["report_id"] == report_id


def test_offline_task_requires_canonical_name_and_filters_legacy_records(
    app, client, auth_header, monkeypatch
):
    from uav_dashboard_api.services import offline_diagnosis

    def fake_submit(app_obj, run_id, worker_fn, *args, **kwargs):
        with app_obj.app_context():
            run = db.session.get(RunRecord, run_id)
            run.status = "success"
            run.progress = 1.0
            run.finished_at = datetime.now()
            run.set_result(
                {
                    "prediction": {"label": "sensor", "confidence": 0.87},
                    "probs": {"sensor": 0.87, "normal": 0.13},
                }
            )
            db.session.commit()

    monkeypatch.setattr(offline_diagnosis, "submit_background_run", fake_submit)
    upload = client.post(
        "/api/assets/upload",
        headers=auth_header,
        data={"file": (io.BytesIO(b"timestamp,a\n0,1\n"), "mission.csv")},
    )
    asset_id = upload.get_json()["data"]["id"]
    with app.app_context():
        asset = db.session.get(Asset, asset_id)
        meta = asset.get_meta()
        meta["prepared_csv_path"] = asset.stored_path
        asset.set_meta(meta)
        db.session.commit()

    legacy_submit = client.post(
        "/api/runs",
        headers=auth_header,
        json={"task": "offline_5class", "asset_id": asset_id, "model": "lstm"},
    )
    assert legacy_submit.status_code == 400
    assert "不支持的任务类型" in legacy_submit.get_json()["message"]

    run_resp = client.post(
        "/api/runs",
        headers=auth_header,
        json={"task": "offline_multiclass", "asset_id": asset_id, "model": "lstm"},
    )
    assert run_resp.status_code == 202
    run_id = run_resp.get_json()["data"]["id"]
    assert run_resp.get_json()["data"]["task"] == "offline_multiclass"

    with app.app_context():
        legacy_run = RunRecord(
            task="offline_5class", status="success", progress=1.0, asset_id=asset_id
        )
        legacy_run.set_result({"prediction": {"label": "motor", "confidence": 0.8}})
        db.session.add(legacy_run)
        db.session.commit()
        legacy_run_id = legacy_run.id

    legacy_report = client.post(
        "/api/reports/diagnosis", headers=auth_header, json={"run_id": legacy_run_id}
    )
    assert legacy_report.status_code == 400
    assert legacy_report.get_json()["message"] == "仅支持离线分类诊断任务"

    legacy_list = client.get("/api/runs?task=offline_5class", headers=auth_header)
    canonical_list = client.get(
        "/api/runs?task=offline_multiclass", headers=auth_header
    )
    assert legacy_list.status_code == 200
    assert canonical_list.status_code == 200
    assert {item["id"] for item in legacy_list.get_json()["data"]} == {legacy_run_id}
    assert {item["id"] for item in canonical_list.get_json()["data"]} == {run_id}
    assert run_id in {item["id"] for item in canonical_list.get_json()["data"]}

    overview = client.get("/api/overview", headers=auth_header)
    assert overview.status_code == 200
    fault_distribution = overview.get_json()["data"]["fault_distribution"]
    assert fault_distribution["sensor"] == 1
    assert fault_distribution["motor"] == 0


def test_offline_run_accepts_telemetry_session_bridge(
    app, client, auth_header, monkeypatch, tmp_path: Path
):
    from uav_dashboard_api.services import offline_diagnosis

    submit_calls = []

    def fake_submit(app_obj, run_id, worker_fn, *args, **kwargs):
        submit_calls.append(run_id)

    monkeypatch.setattr(offline_diagnosis, "submit_background_run", fake_submit)
    csv_path = tmp_path / "telemetry_bridge.csv"
    csv_path.write_text("timestamp,motor1\n0,1\n", encoding="utf-8")

    with app.app_context():
        db.session.add(
            TelemetrySession(
                id="TS-BRIDGE-001",
                session_type="simulation",
                csv_path=str(csv_path),
                fault_events_count=1,
                status="completed",
            )
        )
        db.session.commit()

    response = client.post(
        "/api/runs",
        headers=auth_header,
        json={"task": "offline_multiclass", "session_id": "TS-BRIDGE-001", "model": "lstm"},
    )

    assert response.status_code == 202
    payload = response.get_json()["data"]
    assert payload["task"] == "offline_multiclass"
    assert payload["asset_id"] is not None
    assert payload["params"]["session_id"] == "TS-BRIDGE-001"
    assert payload["params"]["source_type"] == "telemetry_session"
    assert payload["params"]["model"] == "lstm"
    assert submit_calls == [payload["id"]]

    with app.app_context():
        run = db.session.get(RunRecord, payload["id"])
        assert run is not None
        assert run.asset_id == payload["asset_id"]
        assert run.get_params()["session_id"] == "TS-BRIDGE-001"
        assert run.get_params()["source_type"] == "telemetry_session"


def test_offline_run_rejects_missing_telemetry_session(app, client, auth_header):
    response = client.post(
        "/api/runs",
        headers=auth_header,
        json={"task": "offline_multiclass", "session_id": "TS-MISSING-001", "model": "lstm"},
    )

    assert response.status_code == 404
    assert response.get_json()["message"] == "遥测会话不存在"


def test_offline_run_rejects_telemetry_session_without_csv(app, client, auth_header):
    with app.app_context():
        db.session.add(
            TelemetrySession(
                id="TS-NO-CSV-001",
                session_type="simulation",
                status="completed",
            )
        )
        db.session.commit()

    response = client.post(
        "/api/runs",
        headers=auth_header,
        json={"task": "offline_multiclass", "session_id": "TS-NO-CSV-001", "model": "lstm"},
    )

    assert response.status_code == 409
    assert response.get_json()["message"] == "session_csv_unavailable"
def test_offline_run_rejects_telemetry_session_with_missing_csv_file(
    app, client, auth_header, tmp_path: Path
):
    missing_csv = tmp_path / "missing_telemetry_bridge.csv"

    with app.app_context():
        db.session.add(
            TelemetrySession(
                id="TS-MISSING-CSV-001",
                session_type="simulation",
                csv_path=str(missing_csv),
                status="completed",
            )
        )
        db.session.commit()

    response = client.post(
        "/api/runs",
        headers=auth_header,
        json={
            "task": "offline_multiclass",
            "session_id": "TS-MISSING-CSV-001",
            "model": "lstm",
        },
    )

    assert response.status_code == 409
    assert response.get_json()["message"] == "session_csv_unavailable"
