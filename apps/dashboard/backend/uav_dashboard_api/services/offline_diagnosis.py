"""Offline diagnosis admission, task submission and execution.

The public interface requires a Flask application context and owns the
submission transaction. A successful return means the task was submitted,
not that diagnosis has finished. Only stable IDs and the model name cross
into the background task; task_manager owns final status and result storage.
"""
from __future__ import annotations

from pathlib import Path

from flask import current_app

from uav_dashboard_api import db
from uav_dashboard_api.models.asset import Asset
from uav_dashboard_api.models.run_record import RunRecord
from uav_dashboard_api.models.telemetry_session import TelemetrySession
from uav_dashboard_api.services.bridge import run_offline_multiclass
from uav_dashboard_api.services.diagnosis_contract import build_diagnosis_payload
from uav_dashboard_api.services.flight_log_asset_input import (
    AssetNotPreparedError,
    SessionCsvUnavailableError,
    get_or_create_session_asset,
    resolve_csv_path,
)
from uav_dashboard_api.services.model_registry import (
    OFFLINE_MULTICLASS_TASK,
    list_registered_models,
    resolve_registered_model,
)
from uav_dashboard_api.services.severity_recommendation import evaluate_severity_and_recommendations
from uav_dashboard_api.services.task_manager import mark_run_progress, submit_background_run


class OfflineDiagnosisRejected(ValueError):
    """Expected admission failure; the HTTP caller chooses its response."""

    def __init__(self, code: str, message: str, *, asset_id: int | None = None):
        super().__init__(message)
        self.code = code
        self.asset_id = asset_id


def submit_offline_diagnosis(
    *,
    model_name: str,
    created_by: int | None,
    asset_id: int | None = None,
    session_id: str | None = None,
) -> RunRecord:
    """Validate a source, persist its queued run and submit real diagnosis.

    A supplied session takes precedence over an asset, matching the existing
    handoff. Admission errors leave no newly committed run or session asset.
    Executor submission errors propagate after task_manager records failure.
    """
    session_id = str(session_id or "").strip()
    if (not asset_id and not session_id) or not model_name:
        raise OfflineDiagnosisRejected(
            "input_required", "asset_id/session_id 和 model 为必填"
        )

    try:
        if session_id:
            session = db.session.get(TelemetrySession, session_id)
            if session is None:
                raise OfflineDiagnosisRejected("session_missing", "遥测会话不存在")
            try:
                asset = get_or_create_session_asset(session, created_by)
            except SessionCsvUnavailableError as exc:
                raise OfflineDiagnosisRejected(
                    "session_csv_unavailable", str(exc)
                ) from exc
        else:
            asset = db.session.get(Asset, int(asset_id))
            if asset is None:
                raise OfflineDiagnosisRejected("asset_missing", "资产不存在")

        try:
            resolve_csv_path(asset, strict_prepare=True)
        except AssetNotPreparedError as exc:
            raise OfflineDiagnosisRejected(
                "asset_not_prepared", "asset_not_prepared", asset_id=asset.id
            ) from exc

        model_names = {
            item["name"]
            for item in list_registered_models(
                current_app.config, capability="offline_detection"
            )
        }
        if model_name not in model_names:
            raise OfflineDiagnosisRejected("model_missing", f"模型不存在: {model_name}")

        run = RunRecord(
            task=OFFLINE_MULTICLASS_TASK,
            status="queued",
            progress=0.0,
            asset_id=asset.id,
            created_by=created_by,
        )
        params = {"asset_id": asset.id, "model": model_name}
        if session_id:
            params.update(session_id=session_id, source_type="telemetry_session")
        run.set_params(params)
        db.session.add(run)
        db.session.commit()
    except Exception:
        # Session handoff only flushes its asset; admission and queued run
        # creation belong to the same transaction.
        db.session.rollback()
        raise

    # Keep this outside the transaction above: submission failure is recorded
    # by task_manager and must not erase the committed error record.
    submit_background_run(
        current_app._get_current_object(),
        run.id,
        _execute_offline_diagnosis,
        asset.id,
        model_name,
    )
    return run


def _execute_offline_diagnosis(run_id: int, asset_id: int, model_name: str):
    asset = db.session.get(Asset, asset_id)
    if asset is None:
        raise RuntimeError("资产不存在")
    csv_path = resolve_csv_path(asset, strict_prepare=True)
    model_meta = resolve_registered_model(
        current_app.config,
        requested_name=model_name,
        capability="offline_detection",
    )
    if model_meta is None:
        raise RuntimeError(f"模型不存在: {model_name}")
    model_dir = Path(model_meta["path"])

    mark_run_progress(run_id, 0.2)
    run_dir = Path(current_app.config["RUNS_DIR"]) / f"run_{run_id}"
    run_dir.mkdir(parents=True, exist_ok=True)
    output_json = run_dir / "result.json"

    prediction = run_offline_multiclass(
        csv_path=csv_path,
        model_dir=str(model_dir),
        output_json=str(output_json),
    )
    prediction_info = (
        prediction.get("prediction")
        if isinstance(prediction.get("prediction"), dict)
        else {}
    )
    severity_bundle = evaluate_severity_and_recommendations(
        csv_path=csv_path,
        fault_label=str(prediction_info.get("label", "")).strip(),
        confidence=prediction_info.get("confidence"),
    )
    diagnosis = build_diagnosis_payload(
        source="offline",
        label=str(prediction_info.get("label", "")).strip() or "unknown",
        confidence=prediction_info.get("confidence"),
        severity=(severity_bundle.get("severity") or {}).get("level"),
        recommendations=severity_bundle.get("recommendations", []),
        model={
            "name": model_meta["name"],
            "task": model_meta.get("task"),
            "architecture": model_meta.get("architecture"),
            "label_names": model_meta.get("label_names") or [],
        },
        time_range=None,
    )
    from uav_fault.online.decision_engine import TemporalDecisionEngine

    decision_engine = TemporalDecisionEngine()
    offline_decision = decision_engine.attach_offline_diagnosis(
        {
            "fault_label": diagnosis.get("label"),
            "severity": severity_bundle.get("severity"),
            "recommendations": severity_bundle.get("recommendations", []),
            "evidence": (severity_bundle.get("severity") or {}).get("evidence", []),
        }
    )
    diagnosis["decision"] = {
        "episode_id": offline_decision.episode_id,
        "state": offline_decision.state,
        "warning_stage": offline_decision.warning_stage,
        "risk_score": offline_decision.risk_score,
        "risk_trend": offline_decision.risk_trend,
        "warning_reason": offline_decision.warning_reason,
        "lead_time_s": offline_decision.lead_time_s,
        "fault_label": offline_decision.fault_label,
        "severity": offline_decision.severity,
        "evidence_summary": offline_decision.evidence_summary,
        "recommendations": offline_decision.recommendations,
    }
    mark_run_progress(run_id, 0.95)

    return {
        "asset_id": asset.id,
        "asset_kind": asset.kind,
        "input_csv": csv_path,
        "model": model_meta["name"],
        "prediction": prediction.get("prediction"),
        "probs": prediction.get("probs"),
        "labels": prediction.get("labels"),
        "topk": prediction.get("topk"),
        "severity": severity_bundle.get("severity"),
        "recommendations": severity_bundle.get("recommendations", []),
        "recommendation_meta": severity_bundle.get("recommendation_meta", {}),
        "diagnosis": diagnosis,
        "unified_decision": diagnosis["decision"],
        "debug": prediction.get("debug", {}),
        "stdout_tail": prediction.get("stdout_tail", ""),
        "stderr_tail": prediction.get("stderr_tail", ""),
    }
