from __future__ import annotations

from pathlib import Path

from flask import current_app, jsonify, request

from uav_dashboard_api import db
from uav_dashboard_api.models.asset import Asset
from uav_dashboard_api.models.run_record import RunRecord
from uav_dashboard_api.models.telemetry_session import TelemetrySession
from uav_dashboard_api.routes.dashboard_asset_input import (
    AssetNotPreparedError,
    SessionCsvUnavailableError,
    _get_or_create_session_asset,
    _resolved_csv_path,
)
from uav_dashboard_api.services.bridge import run_offline_5class
from uav_dashboard_api.services.diagnosis_contract import build_diagnosis_payload
from uav_dashboard_api.services.model_registry import list_registered_models, resolve_registered_model
from uav_dashboard_api.services.severity_recommendation import evaluate_severity_and_recommendations
from uav_dashboard_api.services.task_manager import mark_run_progress, submit_background_run
from uav_dashboard_api.utils.jwt_utils import token_required


OFFLINE_MULTICLASS_TASK = "offline_multiclass"

def _is_offline_multiclass_task(task: str | None) -> bool:
    return str(task).strip() == OFFLINE_MULTICLASS_TASK

def _offline_5class_worker(run_id: int, asset_id: int, model_name: str):
    asset = db.session.get(Asset, asset_id)
    if asset is None:
        raise RuntimeError("资产不存在")
    csv_path = _resolved_csv_path(asset, strict_prepare=True)
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

    prediction = run_offline_5class(
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

def create_run(current_user):
    payload = request.get_json() or {}
    task = str(payload.get("task") or OFFLINE_MULTICLASS_TASK).strip()
    if not _is_offline_multiclass_task(task):
        return jsonify({"success": False, "message": f"不支持的任务类型: {task}"}), 400

    asset_id = payload.get("asset_id")
    session_id = str(payload.get("session_id") or "").strip()
    model_name = payload.get("model") or payload.get("model_name")
    if (not asset_id and not session_id) or not model_name:
        return jsonify({"success": False, "message": "asset_id/session_id 和 model 为必填"}), 400

    source_type = "asset"
    if session_id:
        from uav_dashboard_api.models.telemetry_session import TelemetrySession

        session = db.session.get(TelemetrySession, session_id)
        if session is None:
            return jsonify({"success": False, "message": "遥测会话不存在"}), 404
        try:
            asset = _get_or_create_session_asset(
                session,
                getattr(current_user, "id", None),
            )
        except SessionCsvUnavailableError as exc:
            return jsonify({"success": False, "message": str(exc)}), 409
        source_type = "telemetry_session"
    else:
        asset = db.session.get(Asset, int(asset_id))
        if asset is None:
            return jsonify({"success": False, "message": "资产不存在"}), 404

    try:
        _resolved_csv_path(asset, strict_prepare=True)
    except AssetNotPreparedError:
        return jsonify(
            {
                "success": False,
                "message": "asset_not_prepared",
                "prepare_endpoint": f"/api/assets/{asset.id}/prepare",
            }
        ), 409

    model_names = {
        item["name"]
        for item in list_registered_models(
            current_app.config,
            capability="offline_detection",
        )
    }
    if model_name not in model_names:
        return jsonify({"success": False, "message": f"模型不存在: {model_name}"}), 400

    run = RunRecord(
        task=OFFLINE_MULTICLASS_TASK,
        status="queued",
        progress=0.0,
        asset_id=asset.id,
        created_by=getattr(current_user, "id", None),
    )
    params = {"asset_id": asset.id, "model": model_name}
    if session_id:
        params["session_id"] = session_id
        params["source_type"] = source_type
    run.set_params(params)
    db.session.add(run)
    db.session.commit()

    submit_background_run(
        current_app._get_current_object(),
        run.id,
        _offline_5class_worker,
        asset.id,
        model_name,
    )
    return jsonify(
        {
            "success": True,
            "message": "检测任务已提交",
            "data": run.to_dict(),
        }
    ), 202

def list_runs(current_user):
    query = RunRecord.query
    task = request.args.get("task")
    status = request.args.get("status")
    limit = request.args.get("limit", default=50, type=int)

    if task:
        task = str(task).strip()
        if _is_offline_multiclass_task(task):
            query = query.filter_by(task=OFFLINE_MULTICLASS_TASK)
        else:
            query = query.filter_by(task=task)
    if status:
        query = query.filter_by(status=status)
    items = (
        query.order_by(RunRecord.created_at.desc()).limit(max(1, min(limit, 200))).all()
    )
    return jsonify({"success": True, "data": [item.to_dict() for item in items]})

def get_run(current_user, run_id: int):
    run = db.session.get(RunRecord, run_id)
    if run is None:
        return jsonify({"success": False, "message": "任务不存在"}), 404
    return jsonify({"success": True, "data": run.to_dict()})

def get_run_result(current_user, run_id: int):
    run = db.session.get(RunRecord, run_id)
    if run is None:
        return jsonify({"success": False, "message": "任务不存在"}), 404
    if run.status != "success":
        return jsonify(
            {
                "success": False,
                "message": "任务尚未成功完成",
                "data": run.to_dict(),
            }
        ), 409
    return jsonify({"success": True, "data": run.get_result()})


def register_routes(bp):
    bp.add_url_rule(
        "/runs", endpoint="create_run", view_func=token_required(create_run), methods=["POST"]
    )
    bp.add_url_rule(
        "/runs", endpoint="list_runs", view_func=token_required(list_runs), methods=["GET"]
    )
    bp.add_url_rule(
        "/runs/<int:run_id>", endpoint="get_run", view_func=token_required(get_run), methods=["GET"]
    )
    bp.add_url_rule(
        "/runs/<int:run_id>/result", endpoint="get_run_result", view_func=token_required(get_run_result), methods=["GET"]
    )
