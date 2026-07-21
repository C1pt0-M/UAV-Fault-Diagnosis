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
from uav_dashboard_api.models.user import User
from uav_dashboard_api.utils.path_store import resolve_local_path

def test_analysis_fault_prob_curve_ignores_plain_csv_asset(
    app, client, auth_header, tmp_path: Path
):
    csv_path = tmp_path / "fault_prob_curve_case.csv"
    csv_path.write_text(
        "time_s,roll,pitch,yaw,fault_prob\n"
        "0.0,0.1,0.0,0.0,0.2\n"
        "0.1,0.2,0.0,0.0,0.8\n",
        encoding="utf-8",
    )

    with app.app_context():
        user = User.query.filter_by(username="flow_user").first()
        asset = Asset(
            filename=csv_path.name,
            kind="csv",
            stored_path=str(csv_path),
            size_bytes=csv_path.stat().st_size,
            sha256="test-sha256-fault-prob-curve",
            created_by=user.id,
        )
        db.session.add(asset)
        db.session.commit()
        asset_id = asset.id

    resp = client.get(
        f"/api/analysis/fault_prob_curve?asset_id={asset_id}",
        headers=auth_header,
    )

    assert resp.status_code == 200
    payload = resp.get_json()
    assert payload["success"] is True
    assert payload["data"] == []
    assert payload["meta"]["available"] is False
    assert payload["meta"]["reason"] == "asset_has_no_monitoring_session"


def test_analysis_fault_prob_curve_reads_online_monitoring_session_asset(
    app, client, auth_header, tmp_path: Path
):
    csv_path = tmp_path / "telemetry_with_fault_prob.csv"
    csv_path.write_text(
        "time_s,roll,pitch,fault_prob\n"
        "0.0,0.1,0.0,0.25\n"
        "0.1,0.2,0.0,0.8\n",
        encoding="utf-8",
    )

    with app.app_context():
        user = User.query.filter_by(username="flow_user").first()
        session = TelemetrySession(
            id="TS-CURVE-001",
            session_type="simulation",
            scenario_backend="replay",
            csv_path=str(csv_path),
            has_detection=True,
            status="completed",
        )
        asset = Asset(
            filename=csv_path.name,
            kind="csv",
            stored_path=str(csv_path),
            size_bytes=csv_path.stat().st_size,
            sha256="test-sha256-telemetry-fault-prob-curve",
            created_by=user.id,
        )
        asset.set_meta(
            {
                "prepared_csv_path": str(csv_path),
                "source_type": "telemetry_session",
                "session_id": session.id,
            }
        )
        db.session.add(session)
        db.session.add(asset)
        db.session.commit()
        asset_id = asset.id

    resp = client.get(
        f"/api/analysis/fault_prob_curve?asset_id={asset_id}",
        headers=auth_header,
    )

    assert resp.status_code == 200
    payload = resp.get_json()
    assert payload["success"] is True
    assert payload["data"] == [
        {"time_s": 0.0, "fault_prob": 0.25},
        {"time_s": 0.1, "fault_prob": 0.8},
    ]
    assert payload["meta"]["available"] is True
    assert payload["meta"]["session_id"] == "TS-CURVE-001"


def test_analysis_fault_prob_curve_reads_source_asset_monitoring_session(
    app, client, auth_header, tmp_path: Path
):
    source_csv = tmp_path / "source_asset.csv"
    source_csv.write_text(
        "time_s,roll,pitch\n"
        "0.0,0.1,0.0\n"
        "0.1,0.2,0.0\n",
        encoding="utf-8",
    )
    session_csv = tmp_path / "telemetry_session.csv"
    session_csv.write_text(
        "time_s,roll,pitch\n"
        "0.0,0.1,0.0\n"
        "0.1,0.2,0.0\n",
        encoding="utf-8",
    )
    curve_path = tmp_path / "curve.json"
    curve_path.write_text(
        json.dumps(
            [
                {"time_s": 0.0, "fault_prob": 0.31},
                {"time_s": 0.1, "fault_prob": 0.86},
            ],
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    with app.app_context():
        user = User.query.filter_by(username="flow_user").first()
        asset = Asset(
            filename=source_csv.name,
            kind="csv",
            stored_path=str(source_csv),
            size_bytes=source_csv.stat().st_size,
            sha256="test-sha256-source-asset-curve",
            created_by=user.id,
        )
        asset.set_meta({"prepared_csv_path": str(source_csv)})
        db.session.add(asset)
        db.session.flush()

        session = TelemetrySession(
            id="TS-SOURCE-ASSET-001",
            session_type="simulation",
            scenario_backend="replay",
            source_path=str(source_csv),
            csv_path=str(session_csv),
            has_detection=True,
            status="completed",
        )
        session.set_meta(
            {
                "backend": "replay",
                "source_asset_id": asset.id,
                "fault_prob_curve_path": str(curve_path),
                "fault_prob_curve_points": 2,
            }
        )
        db.session.add(session)
        db.session.commit()
        asset_id = asset.id

    resp = client.get(
        f"/api/analysis/fault_prob_curve?asset_id={asset_id}",
        headers=auth_header,
    )

    assert resp.status_code == 200
    payload = resp.get_json()
    assert payload["success"] is True
    assert payload["data"] == [
        {"time_s": 0.0, "fault_prob": 0.31},
        {"time_s": 0.1, "fault_prob": 0.86},
    ]
    assert payload["meta"]["available"] is True
    assert payload["meta"]["session_id"] == "TS-SOURCE-ASSET-001"
    assert payload["meta"]["source"] == "fault_prob_curve_file"


def test_analysis_fields_accepts_legacy_linux_runtime_csv_path(
    app, client, auth_header
):
    csv_dir = Path(app.config["UPLOAD_DIR"]) / "csv"
    csv_dir.mkdir(parents=True, exist_ok=True)
    csv_name = "20260326143545_2a2be49b_Case_2508000005.csv"
    csv_path = csv_dir / csv_name
    csv_path.write_text(
        "time_s,motor1,motor2\n"
        "0.0,1200,1190\n"
        "0.1,1210,1200\n",
        encoding="utf-8",
    )

    with app.app_context():
        user = User.query.filter_by(username="flow_user").first()
        asset = Asset(
            filename=csv_name,
            kind="csv",
            stored_path=(
                "/home/jack/UAV-Fault-Diagnosis/apps/dashboard/backend/"
                f"runtime/uploads/csv/{csv_name}"
            ),
            size_bytes=csv_path.stat().st_size,
            sha256="legacy-linux-runtime-csv-sha256",
            created_by=user.id,
        )
        db.session.add(asset)
        db.session.commit()
        asset_id = asset.id

    response = client.get(
        f"/api/analysis/fields?asset_id={asset_id}",
        headers=auth_header,
    )

    assert response.status_code == 200
    payload = response.get_json()
    assert payload["success"] is True
    fields = {
        item["field"]
        for group in payload["data"]["field_groups"]
        for item in group["fields"]
    }
    assert "motor1" in fields
    assert payload["data"]["time"]["key"] == "time_s"
