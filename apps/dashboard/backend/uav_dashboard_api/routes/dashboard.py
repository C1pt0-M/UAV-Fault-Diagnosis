from datetime import datetime, timedelta
import hashlib
import uuid
from pathlib import Path

from flask import Blueprint, current_app, jsonify, request, send_file
from werkzeug.utils import secure_filename

from uav_dashboard_api import db
from uav_dashboard_api.models.asset import Asset
from uav_dashboard_api.models.diagnosis_report import DiagnosisReport
from uav_dashboard_api.models.run_record import RunRecord
from uav_dashboard_api.models.telemetry_session import TelemetryFaultEvent, TelemetrySession
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


def _round_or_none(value, digits: int = 2):
    try:
        if value is None:
            return None
        return round(float(value), digits)
    except (TypeError, ValueError):
        return None


def _label_text(label: str | None) -> str:
    mapping = {
        "normal": "正常",
        "motor": "电机故障",
        "motor_failure": "电机故障",
        "sensor": "传感器故障",
        "battery_failure": "电池故障",
        "prop_imbalance": "旋翼异常",
        "thrust_loss": "推力损失",
        "fault": "故障",
    }
    key = str(label or "").strip()
    return mapping.get(key, key or "暂无数据")


def _evidence_label_text(name: str | None) -> str:
    mapping = {
        "fault_probability": "故障概率异常",
        "fault_prob": "故障概率异常",
        "max_fault_prob": "最大故障概率异常",
        "motor_current_rms": "电机电流 RMS 升高",
        "phase_current_imbalance": "相电流不平衡",
        "motor_current_imbalance": "电机电流不平衡",
        "line_temperature": "线组温度升高",
        "temperature": "温度异常",
        "pwm_duty_cycle": "PWM 占空比波动",
        "voltage_sag": "电压纹波异常",
        "battery_voltage": "电池电压异常",
        "imu_drift": "IMU 数据漂移",
        "gyro_bias": "陀螺仪偏置异常",
        "barometer": "气压计信号异常",
        "warning_stage": "告警阶段",
        "lead_time_s": "预测提前量",
        "risk_score": "风险评分",
    }
    raw = str(name or "").strip()
    key = raw.lower()
    return mapping.get(key, raw or "故障概率异常")


def _severity_bucket(value: str | dict | None) -> str:
    if isinstance(value, dict):
        value = value.get("level")
    raw = str(value or "").strip().lower()
    if raw in {"critical", "high", "heavy", "danger"}:
        return "high"
    if raw in {"medium", "warning", "warn", "moderate"}:
        return "medium"
    if raw in {"low", "minor", "normal"}:
        return "low"
    return "unknown"


def _asset_summary(kind_counts: dict) -> dict:
    csv_total = int(kind_counts.get("csv", 0))
    ulg_total = int(kind_counts.get("ulg", 0))
    prepared_ulg = 0
    for asset in Asset.query.filter_by(kind="ulg").all():
        if asset.is_prepared:
            prepared_ulg += 1
    return {
        "total": int(Asset.query.count()),
        "csv_total": csv_total,
        "ulg_total": ulg_total,
        "prepared_ulg_total": prepared_ulg,
        "diagnosable_total": csv_total + prepared_ulg,
    }


def _task_summary(run_counts: dict, recent_runs: list[RunRecord]) -> dict:
    today = datetime.now().date()
    today_diagnosis_total = 0
    for run in RunRecord.query.filter_by(task=OFFLINE_MULTICLASS_TASK).all():
        if run.created_at and run.created_at.date() == today:
            today_diagnosis_total += 1
    return {
        "completed_total": int(run_counts.get("success", 0)),
        "today_diagnosis_total": today_diagnosis_total,
        "success_total": int(run_counts.get("success", 0)),
        "failed_total": int(run_counts.get("error", 0)) + int(run_counts.get("failed", 0)),
        "running_total": int(run_counts.get("running", 0)),
        "queued_total": int(run_counts.get("queued", 0)),
        "recent": [item.to_dict() for item in recent_runs],
    }


def _fault_summary() -> dict:
    reports = DiagnosisReport.query.order_by(DiagnosisReport.created_at.desc()).all()
    counts: dict[str, int] = {}
    confidence_sum = 0.0
    confidence_count = 0
    for report in reports:
        label = report.prediction_label or "unknown"
        counts[label] = counts.get(label, 0) + 1
        confidence_sum += float(report.confidence or 0.0)
        confidence_count += 1

    dominant_label = ""
    dominant_count = 0
    if counts:
        dominant_label, dominant_count = sorted(
            counts.items(), key=lambda item: item[1], reverse=True
        )[0]
    total = sum(counts.values())
    avg_confidence = (
        round(confidence_sum / confidence_count, 3)
        if confidence_count
        else None
    )
    return {
        "dominant_label": dominant_label,
        "dominant_label_text": _label_text(dominant_label),
        "dominant_count": dominant_count,
        "dominant_ratio": round((dominant_count / total) * 100) if total else 0,
        "avg_confidence": avg_confidence,
        "report_count": len(reports),
        "label_counts": counts,
    }


def _event_message(event: TelemetryFaultEvent) -> str:
    diagnosis = event.get_diagnosis()
    evidence = (diagnosis.get("evidence_summary") or {}).get("top_evidence") or []
    if evidence:
        first = evidence[0]
        return _evidence_label_text(first.get("name") or first.get("metric"))
    label = diagnosis.get("label") or "fault"
    return f"{_label_text(label)}告警"


def _warning_summary() -> dict:
    events = TelemetryFaultEvent.query.order_by(TelemetryFaultEvent.created_at.desc()).all()
    severity_counts = {"high": 0, "medium": 0, "low": 0, "unknown": 0}
    recent = []
    for event in events:
        severity = _severity_bucket(event.get_diagnosis().get("severity") or event.severity)
        severity_counts[severity] = severity_counts.get(severity, 0) + 1
        if len(recent) < 5:
            session = db.session.get(TelemetrySession, event.session_id)
            recent.append(
                {
                    "id": event.id,
                    "time": event.created_at.strftime("%H:%M:%S") if event.created_at else "-",
                    "asset": session.drone_id if session and session.drone_id else event.session_id,
                    "message": _event_message(event),
                    "severity": severity,
                    "max_fault_prob": _round_or_none(event.max_fault_prob, 3),
                }
            )
    return {
        "confirmed_total": len(events),
        "severity_counts": severity_counts,
        "recent": recent,
    }


def _risk_value_from_report(report: DiagnosisReport) -> float:
    label = str(report.prediction_label or "").lower()
    confidence = float(report.confidence or 0.0)
    if label in {"normal", "none"}:
        return round(max(0.0, 1.0 - confidence), 3)
    return round(confidence, 3)


def _risk_trend() -> dict:
    since = datetime.now() - timedelta(hours=24)
    points = []
    for report in DiagnosisReport.query.filter(DiagnosisReport.created_at >= since).all():
        points.append(
            {
                "time": report.created_at.strftime("%m-%d %H:%M") if report.created_at else "-",
                "risk": _risk_value_from_report(report),
                "source": "report",
                "label": _label_text(report.prediction_label),
            }
        )
    for event in TelemetryFaultEvent.query.filter(TelemetryFaultEvent.created_at >= since).all():
        points.append(
            {
                "time": event.created_at.strftime("%m-%d %H:%M") if event.created_at else "-",
                "risk": _round_or_none(event.max_fault_prob, 3) or 0.0,
                "source": "warning",
                "marker": "warning",
                "label": _event_message(event),
            }
        )
    points.sort(key=lambda item: item["time"])
    return {
        "threshold": 0.7,
        "points": points[-24:],
        "description": "基于最近诊断任务、告警事件与故障概率记录聚合",
    }


def _model_summary() -> dict:
    model = resolve_registered_model(
        current_app.config,
        task="online_binary",
        capability="realtime_detection",
    )
    if model is None:
        models = list_registered_models(
            current_app.config,
            capability="realtime_detection",
            include_unavailable=True,
        )
        model = models[0] if models else None
    if model is None:
        return {
            "status": "missing",
            "name": "暂无注册模型",
            "display_name": "暂无注册模型",
            "metrics": {},
            "capabilities": {},
        }
    payload = dict(model)
    payload["status"] = "registered"
    payload["display_name"] = _model_display_name(payload)
    updated_at = payload.get("updated_at")
    if updated_at:
        try:
            payload["updated_at_text"] = datetime.fromtimestamp(int(updated_at)).strftime("%Y-%m-%d")
        except (TypeError, ValueError, OSError):
            payload["updated_at_text"] = "-"
    return payload


def _model_display_name(model: dict) -> str:
    architecture = str(model.get("architecture") or model.get("model") or "").strip()
    name = str(model.get("name") or "").strip()
    lookup = f"{architecture} {name}".lower()
    if "kg" in lookup and "mstca" in lookup:
        return "KG-MSTCA 在线检测模型"
    if "mstca" in lookup:
        return "MSTCA 在线检测模型"
    if "cnn1d" in lookup or "cnn_1d" in lookup:
        return "CNN1D 在线检测模型"
    if "lstm" in lookup:
        return "LSTM 离线诊断模型"
    if name:
        return name
    return "暂无注册模型"


def _runtime_telemetry_status() -> dict:
    try:
        from uav_dashboard_api.services.telemetry_manager import telemetry_manager

        status = telemetry_manager.get_status()
        return status if isinstance(status, dict) else {}
    except Exception:
        return {}


def _field_contract(runtime_status: dict) -> dict:
    coverage = runtime_status.get("feature_coverage")
    source = "当前输入"
    schema_version = runtime_status.get("schema_version")

    if not coverage:
        latest_session = TelemetrySession.query.order_by(TelemetrySession.created_at.desc()).first()
        if latest_session is not None:
            meta = latest_session.get_meta()
            coverage = meta.get("feature_coverage") or meta.get("coverage")
            schema_version = schema_version or latest_session.schema_version
            source = "最近一次输入"

    if not isinstance(coverage, dict):
        has_schema_context = bool(schema_version and schema_version != "-")
        return {
            "status": "waiting",
            "source": source if has_schema_context else "等待输入",
            "schema_version": schema_version or "-",
            "input_integrity": None,
            "missing_fields": None,
            "feature_coverage": None,
            "message": "等待特征覆盖数据" if has_schema_context else "等待输入",
            "reason": "latest_session_without_feature_coverage"
            if has_schema_context
            else "no_telemetry_input",
        }

    mapped = int(coverage.get("mapped_count") or len(coverage.get("mapped") or []))
    derived = int(coverage.get("derived_count") or len(coverage.get("derived") or []))
    defaults = int(coverage.get("default_count") or len(coverage.get("default") or []))
    proxies = int(coverage.get("proxy_count") or len(coverage.get("proxy") or []))
    feature_count = int(coverage.get("feature_count") or (mapped + derived + defaults + proxies))
    observed = mapped + derived + proxies
    denominator = max(feature_count, observed + defaults, 1)
    input_integrity = round((observed / denominator) * 100)
    feature_coverage = round(((mapped + derived) / denominator) * 100)
    missing_fields = defaults
    status = "passed" if missing_fields == 0 else "degraded"
    return {
        "status": status,
        "source": source,
        "schema_version": coverage.get("schema_version") or schema_version or "-",
        "input_integrity": input_integrity,
        "missing_fields": missing_fields,
        "feature_coverage": feature_coverage,
        "message": "字段覆盖达标" if status == "passed" else "存在默认填充字段",
        "reason": "coverage_available",
    }


def _telemetry_summary(runtime_status: dict) -> dict:
    latest_session = TelemetrySession.query.order_by(TelemetrySession.created_at.desc()).first()
    current_session_id = runtime_status.get("current_session_id")
    return {
        "current_session_id": current_session_id,
        "latest_session_id": current_session_id or (latest_session.id if latest_session else None),
        "business_status": runtime_status.get("business_status") or "idle",
        "backend_status": "正常",
        "last_sync": datetime.now().strftime("%H:%M:%S"),
        "is_detecting": bool(runtime_status.get("is_detecting")),
        "frame_count": runtime_status.get("frame_count"),
        "risk_score": runtime_status.get("risk_score"),
        "warning_stage": runtime_status.get("warning_stage"),
    }


def _recent_evidence() -> list[dict]:
    evidence_items = []
    events = TelemetryFaultEvent.query.order_by(TelemetryFaultEvent.created_at.desc()).limit(5).all()
    for event in events:
        diagnosis = event.get_diagnosis()
        for item in (diagnosis.get("evidence_summary") or {}).get("top_evidence") or []:
            evidence_items.append(
                {
                    "name": _evidence_label_text(item.get("name") or item.get("metric")),
                    "raw_name": item.get("name") or item.get("metric") or "fault_probability",
                    "value": item.get("value"),
                    "status": item.get("status") or "observed",
                }
            )
            if len(evidence_items) >= 5:
                return evidence_items

    runs = RunRecord.query.order_by(RunRecord.created_at.desc()).limit(10).all()
    for run in runs:
        result = run.get_result() or {}
        diagnosis = result.get("diagnosis") or {}
        for item in (diagnosis.get("evidence_summary") or {}).get("top_evidence") or []:
            evidence_items.append(
                {
                    "name": _evidence_label_text(item.get("name") or item.get("metric")),
                    "raw_name": item.get("name") or item.get("metric") or "diagnosis_evidence",
                    "value": item.get("value"),
                    "status": item.get("status") or "observed",
                }
            )
            if len(evidence_items) >= 5:
                return evidence_items
    return evidence_items


def _similar_reports(dominant_label: str | None) -> list[dict]:
    query = DiagnosisReport.query
    if dominant_label:
        query = query.filter(DiagnosisReport.prediction_label == dominant_label)
    reports = query.order_by(DiagnosisReport.created_at.desc()).limit(3).all()
    if not reports and dominant_label:
        reports = DiagnosisReport.query.order_by(DiagnosisReport.created_at.desc()).limit(3).all()
    items = []
    for report in reports:
        items.append(
            {
                "id": report.id,
                "asset": f"Asset #{report.asset_id}" if report.asset_id else "-",
                "fault_label": _label_text(report.prediction_label),
                "confidence": _round_or_none(report.confidence, 3),
                "created_at": report.created_at.strftime("%Y-%m-%d %H:%M:%S") if report.created_at else None,
            }
        )
    return items


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

    runtime_status = _runtime_telemetry_status()
    asset_summary = _asset_summary(kind_counts)
    task_summary = _task_summary(run_counts, recent_runs)
    warning_summary = _warning_summary()
    fault_summary = _fault_summary()

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
                "asset_summary": asset_summary,
                "task_summary": task_summary,
                "warning_summary": warning_summary,
                "fault_summary": fault_summary,
                "risk_trend": _risk_trend(),
                "model_summary": _model_summary(),
                "field_contract": _field_contract(runtime_status),
                "telemetry_summary": _telemetry_summary(runtime_status),
                "recent_evidence": _recent_evidence(),
                "similar_reports": _similar_reports(fault_summary.get("dominant_label")),
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
