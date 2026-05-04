from datetime import datetime
import hashlib
import uuid
from pathlib import Path

from flask import Blueprint, current_app, jsonify, request, send_file
from werkzeug.utils import secure_filename

from uav_dashboard_api import db
from uav_dashboard_api.models.asset import Asset
from uav_dashboard_api.models.run_record import RunRecord
from uav_dashboard_api.models.telemetry_session import TelemetrySession
from uav_dashboard_api.services.analysis import build_fields_response, build_timeseries_response
from uav_dashboard_api.services.bridge import run_offline_5class, run_ulg_to_csv
from uav_dashboard_api.services.diagnosis_contract import build_diagnosis_payload
from uav_dashboard_api.services.model_registry import list_registered_models, resolve_registered_model
from uav_dashboard_api.services.severity_recommendation import evaluate_severity_and_recommendations
from uav_dashboard_api.services.task_manager import mark_run_progress, submit_background_run
from uav_dashboard_api.utils.jwt_utils import token_required
from uav_dashboard_api.utils.path_store import resolve_local_path, serialize_local_path

bp = Blueprint("dashboard", __name__)

ALLOWED_EXTS = {
    ".ulg": "ulg",
    ".csv": "csv",
}

OFFLINE_MULTICLASS_TASK = "offline_multiclass"
FAULT_PROB_CURVE_CACHE_VERSION = "v2"


def _is_offline_multiclass_task(task: str | None) -> bool:
    return str(task).strip() == OFFLINE_MULTICLASS_TASK


class AssetNotPreparedError(RuntimeError):
    """ULG 资产尚未prepare。"""


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fp:
        while True:
            chunk = fp.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _get_asset_or_404(asset_id: int):
    asset = db.session.get(Asset, asset_id)
    if asset is None:
        return None, (jsonify({"success": False, "message": "资产不存在"}), 404)
    return asset, None


def _resolved_csv_path(asset: Asset, *, strict_prepare: bool = True) -> str:
    if asset.kind == "csv":
        return resolve_local_path(asset.stored_path, current_app.config)
    meta = asset.get_meta()
    csv_path = resolve_local_path(meta.get("prepared_csv_path"), current_app.config)
    if csv_path and Path(csv_path).exists():
        return csv_path
    if strict_prepare:
        raise AssetNotPreparedError("asset_not_prepared")
    return ""


class SessionCsvUnavailableError(RuntimeError):
    """遥测会话未提供可分析CSV。"""


def _get_or_create_session_asset(session, created_by: int | None):
    csv_path = resolve_local_path(session.csv_path, current_app.config)
    if not csv_path:
        raise SessionCsvUnavailableError("session_csv_unavailable")
    csv_file = Path(csv_path)
    if not csv_file.exists():
        raise SessionCsvUnavailableError("session_csv_unavailable")

    serialized_csv_path = serialize_local_path(csv_file, current_app.config)
    asset = Asset.query.filter_by(stored_path=serialized_csv_path).first()
    if asset is not None:
        return asset

    asset = Asset(
        filename=csv_file.name,
        kind="csv",
        stored_path=serialized_csv_path,
        size_bytes=csv_file.stat().st_size,
        sha256=_sha256_file(csv_file),
        created_by=created_by,
    )
    asset.set_meta({
        "prepared_csv_path": str(csv_file),
        "source_type": "telemetry_session",
        "session_id": session.id,
    })
    db.session.add(asset)
    db.session.flush()
    return asset


def _prepare_ulg_worker(run_id: int, asset_id: int):
    asset = db.session.get(Asset, asset_id)
    if asset is None:
        raise RuntimeError("资产不存在")

    mark_run_progress(run_id, 0.1)
    output_dir = Path(current_app.config["DERIVED_DIR"]) / f"asset_{asset.id}"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_csv = output_dir / f"{Path(asset.filename).stem}_prepared.csv"

    bridge_info = run_ulg_to_csv(
        resolve_local_path(asset.stored_path, current_app.config),
        str(output_csv),
    )
    if not output_csv.exists():
        raise RuntimeError("转换完成但未生成CSV")
    mark_run_progress(run_id, 0.85)

    meta = asset.get_meta()
    meta["prepared_csv_path"] = str(output_csv)
    meta["prepared_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    asset.set_meta(meta)
    db.session.commit()

    try:
        import pandas as pd

        sample_df = pd.read_csv(output_csv, nrows=1000)
        columns = list(sample_df.columns)
        n_rows = sum(1 for _ in output_csv.open("r", encoding="utf-8")) - 1
    except Exception:
        columns = []
        n_rows = None

    return {
        "asset_id": asset.id,
        "output_csv": serialize_local_path(output_csv, current_app.config),
        "n_rows": n_rows,
        "columns": columns,
        "bridge": bridge_info,
    }


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


@bp.route("/overview", methods=["GET"])
@token_required
def get_overview(current_user):
    kind_counts = dict(
        db.session.query(Asset.kind, db.func.count(Asset.id)).group_by(Asset.kind).all()
    )
    run_counts = dict(
        db.session.query(RunRecord.status, db.func.count(RunRecord.id))
        .group_by(RunRecord.status)
        .all()
    )
    recent_runs = RunRecord.query.order_by(RunRecord.created_at.desc()).limit(10).all()

    # 统计成功任务的预测故障类型
    fault_counts = {
        "normal": 0,
        "motor": 0,
        "sensor": 0,
    }

    successful_runs = RunRecord.query.filter(
        RunRecord.status == "success",
        RunRecord.task == OFFLINE_MULTICLASS_TASK,
    ).all()
    for run in successful_runs:
        result = run.get_result()
        if isinstance(result, dict) and "prediction" in result:
            label = result["prediction"].get("label", "")
            if label in fault_counts:
                fault_counts[label] += 1

    return jsonify(
        {
            "success": True,
            "data": {
                "assets_total": int(Asset.query.count()),
                "assets_by_kind": {
                    "ulg": int(kind_counts.get("ulg", 0)),
                    "csv": int(kind_counts.get("csv", 0)),
                },
                "runs_total": int(RunRecord.query.count()),
                "runs_by_status": {
                    "queued": int(run_counts.get("queued", 0)),
                    "running": int(run_counts.get("running", 0)),
                    "success": int(run_counts.get("success", 0)),
                    "error": int(run_counts.get("error", 0)),
                },
                "fault_distribution": fault_counts,
                "recent_runs": [item.to_dict() for item in recent_runs],
            },
        }
    )


@bp.route("/models", methods=["GET"])
@token_required
def get_models(current_user):
    task = request.args.get("task")
    capability = request.args.get("capability", "offline_detection")
    include_unavailable = request.args.get("include_unavailable", "").lower() in {
        "1",
        "true",
        "yes",
    }
    models = list_registered_models(
        current_app.config,
        task=task or None,
        capability=capability or None,
        include_unavailable=include_unavailable,
    )
    return jsonify({"success": True, "data": models})


@bp.route("/assets/upload", methods=["POST"])
@token_required
def upload_asset(current_user):
    if "file" not in request.files:
        return jsonify({"success": False, "message": "缺少file字段"}), 400
    file = request.files["file"]
    if not file or not file.filename:
        return jsonify({"success": False, "message": "未选择文件"}), 400

    filename = secure_filename(file.filename) or file.filename
    ext = Path(filename).suffix.lower()
    kind = ALLOWED_EXTS.get(ext)
    if kind is None:
        return jsonify({"success": False, "message": "仅支持 .ulg 或 .csv 文件"}), 400

    save_dir = Path(current_app.config["UPLOAD_DIR"]) / kind
    save_dir.mkdir(parents=True, exist_ok=True)

    stored_name = (
        f"{datetime.now().strftime('%Y%m%d%H%M%S')}_{uuid.uuid4().hex[:8]}_{filename}"
    )
    stored_path = save_dir / stored_name
    file.save(stored_path)

    sha256 = _sha256_file(stored_path)

    # Check for duplicates using SHA256
    existing_asset = Asset.query.filter_by(sha256=sha256).first()
    if existing_asset:
        stored_path.unlink()
        return jsonify(
            {
                "success": False,
                "message": "文件已存在(文件指纹雷同)，跳过上传",
                "data": existing_asset.to_dict(),
            }
        ), 409

    asset = Asset(
        filename=file.filename,
        kind=kind,
        stored_path=serialize_local_path(stored_path, current_app.config),
        size_bytes=stored_path.stat().st_size,
        sha256=sha256,
        created_by=getattr(current_user, "id", None),
    )
    meta = {}
    if kind == "csv":
        meta["prepared_csv_path"] = str(stored_path)
    asset.set_meta(meta)
    db.session.add(asset)
    db.session.commit()

    return jsonify(
        {
            "success": True,
            "message": "上传成功",
            "data": asset.to_dict(),
        }
    ), 201


@bp.route("/assets", methods=["GET"])
@token_required
def list_assets(current_user):
    kind = request.args.get("kind", "all")
    query = Asset.query
    if kind in ("ulg", "csv"):
        query = query.filter_by(kind=kind)
    assets = query.order_by(Asset.created_at.desc()).all()
    return jsonify({"success": True, "data": [item.to_dict() for item in assets]})


@bp.route("/assets/<int:asset_id>", methods=["GET"])
@token_required
def get_asset(current_user, asset_id: int):
    asset, error = _get_asset_or_404(asset_id)
    if error:
        return error
    return jsonify({"success": True, "data": asset.to_dict()})


@bp.route("/assets/<int:asset_id>/download", methods=["GET"])
@token_required
def download_asset(current_user, asset_id: int):
    asset, error = _get_asset_or_404(asset_id)
    if error:
        return error
    file_path = Path(resolve_local_path(asset.stored_path, current_app.config))
    if not file_path.exists():
        return jsonify({"success": False, "message": "文件不存在"}), 404
    return send_file(file_path, as_attachment=True, download_name=asset.filename)


@bp.route("/assets/<int:asset_id>", methods=["DELETE"])
@token_required
def delete_asset(current_user, asset_id: int):
    asset, error = _get_asset_or_404(asset_id)
    if error:
        return error

    active_run = (
        RunRecord.query.filter(
            RunRecord.asset_id == asset.id, RunRecord.status.in_(["queued", "running"])
        )
        .order_by(RunRecord.created_at.desc())
        .first()
    )
    if active_run is not None:
        return jsonify(
            {
                "success": False,
                "message": "asset_has_active_runs",
                "run_id": active_run.id,
            }
        ), 409

    file_paths: list[Path] = []
    if asset.stored_path:
        file_paths.append(Path(resolve_local_path(asset.stored_path, current_app.config)))
    prepared_csv_path = asset.get_meta().get("prepared_csv_path")
    if prepared_csv_path:
        prepared_path = Path(resolve_local_path(prepared_csv_path, current_app.config))
        if prepared_path not in file_paths:
            file_paths.append(prepared_path)

    # 保留历史任务记录，但解除与资产的外键关联。
    RunRecord.query.filter_by(asset_id=asset.id).update(
        {"asset_id": None},
        synchronize_session=False,
    )
    db.session.delete(asset)
    db.session.commit()

    deleted_files = []
    missing_files = []
    failed_files = []
    for path in file_paths:
        try:
            if path.exists():
                path.unlink()
                deleted_files.append(str(path))
            else:
                missing_files.append(str(path))
        except Exception as exc:
            failed_files.append({"path": str(path), "error": str(exc)})

    return jsonify(
        {
            "success": True,
            "message": "资产已删除",
            "data": {
                "asset_id": asset_id,
                "deleted_files": deleted_files,
                "missing_files": missing_files,
                "failed_files": failed_files,
            },
        }
    )


@bp.route("/assets/<int:asset_id>/prepare", methods=["POST"])
@token_required
def prepare_asset(current_user, asset_id: int):
    asset, error = _get_asset_or_404(asset_id)
    if error:
        return error
    if asset.kind != "ulg":
        return jsonify({"success": False, "message": "仅ULG文件需要prepare"}), 400

    csv_path = resolve_local_path(
        asset.get_meta().get("prepared_csv_path"),
        current_app.config,
    )
    if csv_path and Path(csv_path).exists():
        return jsonify(
            {
                "success": True,
                "message": "资产已prepare",
                "data": {"asset": asset.to_dict()},
            }
        )

    run = RunRecord(
        task="convert_ulg",
        status="queued",
        progress=0.0,
        asset_id=asset.id,
        created_by=getattr(current_user, "id", None),
    )
    run.set_params({"asset_id": asset.id, "filename": asset.filename})
    db.session.add(run)
    db.session.commit()

    submit_background_run(
        current_app._get_current_object(),
        run.id,
        _prepare_ulg_worker,
        asset.id,
    )
    return jsonify(
        {
            "success": True,
            "message": "prepare任务已提交",
            "data": run.to_dict(),
        }
    ), 202


@bp.route("/runs", methods=["POST"])
@token_required
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


@bp.route("/runs", methods=["GET"])
@token_required
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


@bp.route("/runs/<int:run_id>", methods=["GET"])
@token_required
def get_run(current_user, run_id: int):
    run = db.session.get(RunRecord, run_id)
    if run is None:
        return jsonify({"success": False, "message": "任务不存在"}), 404
    return jsonify({"success": True, "data": run.to_dict()})


@bp.route("/runs/<int:run_id>/result", methods=["GET"])
@token_required
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


@bp.route("/analysis/fields", methods=["GET"])
@token_required
def get_analysis_fields(current_user):
    asset_id = request.args.get("asset_id", type=int)
    if not asset_id:
        return jsonify({"success": False, "message": "asset_id 为必填"}), 400
    asset = db.session.get(Asset, asset_id)
    if asset is None:
        return jsonify({"success": False, "message": "资产不存在"}), 404

    try:
        csv_path = _resolved_csv_path(asset, strict_prepare=True)
    except AssetNotPreparedError:
        return jsonify(
            {
                "success": False,
                "message": "asset_not_prepared",
                "prepare_endpoint": f"/api/assets/{asset.id}/prepare",
            }
        ), 409

    data = build_fields_response(csv_path)
    return jsonify({"success": True, "data": data})


@bp.route("/analysis/timeseries", methods=["GET"])
@token_required
def get_analysis_timeseries(current_user):
    asset_id = request.args.get("asset_id", type=int)
    fields_raw = request.args.get("fields", "").strip()
    start_t = request.args.get("start_t", type=float)
    end_t = request.args.get("end_t", type=float)
    max_points = request.args.get("max_points", default=2000, type=int)

    if not asset_id:
        return jsonify({"success": False, "message": "asset_id 为必填"}), 400
    fields = [item.strip() for item in fields_raw.split(",") if item.strip()]
    if not fields:
        return jsonify({"success": False, "message": "fields 不能为空"}), 400

    asset = db.session.get(Asset, asset_id)
    if asset is None:
        return jsonify({"success": False, "message": "资产不存在"}), 404

    try:
        csv_path = _resolved_csv_path(asset, strict_prepare=True)
    except AssetNotPreparedError:
        return jsonify(
            {
                "success": False,
                "message": "asset_not_prepared",
                "prepare_endpoint": f"/api/assets/{asset.id}/prepare",
            }
        ), 409

    data = build_timeseries_response(
        csv_path=csv_path,
        fields=fields,
        start_t=start_t,
        end_t=end_t,
        max_points=max_points,
    )
    return jsonify({"success": True, "data": data})


@bp.route("/analysis/fault_prob_curve", methods=["GET"])
@token_required
def get_fault_prob_curve(current_user):
    import json
    import subprocess

    asset_id = request.args.get("asset_id", type=int)
    requested_model = request.args.get("model", default="", type=str).strip()
    if not asset_id:
        return jsonify({"success": False, "message": "asset_id 为必填"}), 400
    asset = db.session.get(Asset, asset_id)
    if asset is None:
        return jsonify({"success": False, "message": "资产不存在"}), 404

    try:
        csv_path = _resolved_csv_path(asset, strict_prepare=True)
    except AssetNotPreparedError:
        return jsonify(
            {
                "success": False,
                "message": "asset_not_prepared",
                "prepare_endpoint": f"/api/assets/{asset.id}/prepare",
            }
        ), 409

    model_meta = resolve_registered_model(
        current_app.config,
        requested_name=requested_model or None,
        task="online_binary",
        capability="fault_prob_curve",
    )
    if model_meta is None:
        return jsonify(
            {
                "success": False,
                "message": "未找到可用在线模型，请先配置 DEFAULT_ONLINE_MODEL 或补齐模型目录",
            }
        ), 404

    cache_name = secure_filename(model_meta["name"]) or "online_model"
    output_json = (
        Path(current_app.config["DERIVED_DIR"])
        / f"asset_{asset.id}"
        / f"fault_prob_curve_{cache_name}_{FAULT_PROB_CURVE_CACHE_VERSION}.json"
    )
    if output_json.exists():
        with open(output_json, "r", encoding="utf-8") as f:
            data = json.load(f)
        return jsonify({"success": True, "data": data, "meta": {"model": model_meta}})

    cfg = current_app.config
    python_bin = cfg.get("ML_PYTHON_BIN", "python")
    script = Path(cfg["REPO_ROOT"]) / "src" / "online" / "run_detector.py"
    model_dir = Path(model_meta["path"])

    cmd = [
        python_bin,
        str(script),
        "--mode",
        "csv",
        "--csv",
        str(csv_path),
        "--model-dir",
        str(model_dir),
        "--every",
        "10",
    ]

    completed = subprocess.run(
        cmd, capture_output=True, text=True, cwd=cfg["REPO_ROOT"]
    )
    if completed.returncode != 0:
        current_app.logger.error(f"Online detector failed: {completed.stderr}")
        return jsonify({"success": False, "message": "在线检测推理失败"}), 500

    curve = []
    for line in completed.stderr.splitlines() + completed.stdout.splitlines():
        line = line.strip()
        if line.startswith("{"):
            try:
                row = json.loads(line)
                if "time_s" in row and "fault_prob" in row:
                    curve.append(
                        {
                            "time_s": float(row["time_s"]),
                            "fault_prob": float(row["fault_prob"]),
                        }
                    )
            except Exception:
                pass

    output_json.parent.mkdir(parents=True, exist_ok=True)
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(curve, f)

    return jsonify({"success": True, "data": curve, "meta": {"model": model_meta}})
