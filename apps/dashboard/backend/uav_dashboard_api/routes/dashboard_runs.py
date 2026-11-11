from __future__ import annotations

from flask import jsonify, request

from uav_dashboard_api import db
from uav_dashboard_api.models.run_record import RunRecord
from uav_dashboard_api.services.model_registry import OFFLINE_MULTICLASS_TASK
from uav_dashboard_api.services.offline_diagnosis import (
    OfflineDiagnosisRejected,
    submit_offline_diagnosis,
)
from uav_dashboard_api.utils.jwt_utils import token_required, is_admin


_DIAGNOSIS_REJECTION_STATUS = {
    "input_required": 400,
    "asset_missing": 404,
    "session_missing": 404,
    "session_csv_unavailable": 409,
    "asset_not_prepared": 409,
    "model_missing": 400,
}


def _is_offline_multiclass_task(task: str | None) -> bool:
    return str(task).strip() == OFFLINE_MULTICLASS_TASK


def create_run(current_user):
    payload = request.get_json() or {}
    task = str(payload.get("task") or OFFLINE_MULTICLASS_TASK).strip()
    if not _is_offline_multiclass_task(task):
        return jsonify({"success": False, "message": f"不支持的任务类型: {task}"}), 400

    try:
        run = submit_offline_diagnosis(
            asset_id=payload.get("asset_id"),
            session_id=payload.get("session_id"),
            model_name=payload.get("model") or payload.get("model_name"),
            created_by=getattr(current_user, "id", None),
        )
    except OfflineDiagnosisRejected as exc:
        response = {"success": False, "message": str(exc)}
        if exc.code == "asset_not_prepared":
            response["prepare_endpoint"] = f"/api/assets/{exc.asset_id}/prepare"
        return jsonify(response), _DIAGNOSIS_REJECTION_STATUS[exc.code]

    return jsonify(
        {
            "success": True,
            "message": "检测任务已提交",
            "data": run.to_dict(),
        }
    ), 202


def list_runs(current_user):
    query = RunRecord.query
    if not is_admin(current_user):
        query = query.filter(RunRecord.created_by == current_user.id)
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
    if not is_admin(current_user) and run.created_by != getattr(current_user, 'id', None):
        return jsonify({"success": False, "message": "任务不存在"}), 404
    return jsonify({"success": True, "data": run.to_dict()})

def get_run_result(current_user, run_id: int):
    run = db.session.get(RunRecord, run_id)
    if run is None:
        return jsonify({"success": False, "message": "任务不存在"}), 404
    if not is_admin(current_user) and run.created_by != getattr(current_user, 'id', None):
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
