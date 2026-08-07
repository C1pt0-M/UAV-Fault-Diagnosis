from __future__ import annotations

from datetime import datetime, timedelta

from flask import current_app, jsonify, request

from uav_dashboard_api import db
from uav_dashboard_api.models.asset import Asset
from uav_dashboard_api.models.diagnosis_report import DiagnosisReport
from uav_dashboard_api.models.run_record import RunRecord
from uav_dashboard_api.models.telemetry_session import TelemetryFaultEvent, TelemetrySession
from uav_dashboard_api.services.model_registry import (
    list_registered_models,
    resolve_registered_model,
)
from uav_dashboard_api.utils.jwt_utils import token_required, is_admin


OFFLINE_MULTICLASS_TASK = "offline_multiclass"

def _owned_filters(user, column):
    return [] if is_admin(user) else [column == getattr(user, 'id', None)]


def _session_owned_filters(user):
    return [] if is_admin(user) else [TelemetrySession.created_by == str(getattr(user, 'id', None))]


def _fault_events_scoped(user):
    query = TelemetryFaultEvent.query
    if not is_admin(user):
        query = query.join(
            TelemetrySession,
            TelemetrySession.id == TelemetryFaultEvent.session_id,
        ).filter(TelemetrySession.created_by == str(getattr(user, 'id', None)))
    return query

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

def _asset_summary(kind_counts: dict, user) -> dict:
    csv_total = int(kind_counts.get("csv", 0))
    ulg_total = int(kind_counts.get("ulg", 0))
    prepared_ulg = 0
    ulg_assets = Asset.query.filter_by(kind="ulg").filter(*_owned_filters(user, Asset.created_by)).all()
    for asset in ulg_assets:
        if asset.is_prepared:
            prepared_ulg += 1
    return {
        "total": int(Asset.query.filter(*_owned_filters(user, Asset.created_by)).count()),
        "csv_total": csv_total,
        "ulg_total": ulg_total,
        "prepared_ulg_total": prepared_ulg,
        "diagnosable_total": csv_total + prepared_ulg,
    }

def _task_summary(run_counts: dict, recent_runs: list[RunRecord], user) -> dict:
    today = datetime.now().date()
    today_diagnosis_total = 0
    for run in RunRecord.query.filter_by(task=OFFLINE_MULTICLASS_TASK).filter(*_owned_filters(user, RunRecord.created_by)).all():
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

def _fault_summary(user) -> dict:
    reports = DiagnosisReport.query.order_by(DiagnosisReport.created_at.desc()).filter(*_owned_filters(user, DiagnosisReport.created_by)).all()
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

def _warning_summary(user) -> dict:
    events = _fault_events_scoped(user).order_by(TelemetryFaultEvent.created_at.desc()).all()
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

def _risk_trend(user) -> dict:
    since = datetime.now() - timedelta(hours=24)
    points = []
    for report in DiagnosisReport.query.filter(DiagnosisReport.created_at >= since).filter(*_owned_filters(user, DiagnosisReport.created_by)).all():
        points.append(
            {
                "time": report.created_at.strftime("%m-%d %H:%M") if report.created_at else "-",
                "risk": _risk_value_from_report(report),
                "source": "report",
                "label": _label_text(report.prediction_label),
            }
        )
    for event in _fault_events_scoped(user).filter(TelemetryFaultEvent.created_at >= since).all():
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

def _field_contract(runtime_status: dict, user) -> dict:
    coverage = runtime_status.get("feature_coverage")
    source = "当前输入"
    schema_version = runtime_status.get("schema_version")

    if not coverage:
        latest_session = (
            TelemetrySession.query.order_by(TelemetrySession.created_at.desc())
            .filter(*_session_owned_filters(user))
            .first()
        )
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

def _telemetry_summary(runtime_status: dict, user) -> dict:
    latest_session = (
        TelemetrySession.query.order_by(TelemetrySession.created_at.desc())
        .filter(*_session_owned_filters(user))
        .first()
    )
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

def _recent_evidence(user) -> list[dict]:
    evidence_items = []
    events = _fault_events_scoped(user).order_by(TelemetryFaultEvent.created_at.desc()).limit(5).all()
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

    runs = RunRecord.query.order_by(RunRecord.created_at.desc()).filter(*_owned_filters(user, RunRecord.created_by)).limit(10).all()
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

def _similar_reports(dominant_label: str | None, user) -> list[dict]:
    query = DiagnosisReport.query
    if not is_admin(user):
        query = query.filter(DiagnosisReport.created_by == getattr(user, 'id', None))
    if dominant_label:
        query = query.filter(DiagnosisReport.prediction_label == dominant_label)
    reports = query.order_by(DiagnosisReport.created_at.desc()).limit(3).all()
    if not reports and dominant_label:
        reports = (
            DiagnosisReport.query.order_by(DiagnosisReport.created_at.desc())
            .filter(*_owned_filters(user, DiagnosisReport.created_by))
            .limit(3)
            .all()
        )
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

def get_overview(current_user):
    kind_counts = dict(
        db.session.query(Asset.kind, db.func.count(Asset.id))
        .filter(*_owned_filters(current_user, Asset.created_by))
        .group_by(Asset.kind)
        .all()
    )
    run_counts = dict(
        db.session.query(RunRecord.status, db.func.count(RunRecord.id))
        .filter(*_owned_filters(current_user, RunRecord.created_by))
        .group_by(RunRecord.status)
        .all()
    )
    recent_runs = (
        RunRecord.query.order_by(RunRecord.created_at.desc())
        .filter(*_owned_filters(current_user, RunRecord.created_by))
        .limit(10)
        .all()
    )

    # 统计成功任务的预测故障类型
    fault_counts = {
        "normal": 0,
        "motor": 0,
        "sensor": 0,
    }

    successful_runs = RunRecord.query.filter(
        RunRecord.status == "success",
        RunRecord.task == OFFLINE_MULTICLASS_TASK,
    ).filter(*_owned_filters(current_user, RunRecord.created_by)).all()
    for run in successful_runs:
        result = run.get_result()
        if isinstance(result, dict) and "prediction" in result:
            label = result["prediction"].get("label", "")
            if label in fault_counts:
                fault_counts[label] += 1

    runtime_status = _runtime_telemetry_status()
    asset_summary = _asset_summary(kind_counts, current_user)
    task_summary = _task_summary(run_counts, recent_runs, current_user)
    warning_summary = _warning_summary(current_user)
    fault_summary = _fault_summary(current_user)

    return jsonify(
        {
            "success": True,
            "data": {
                "assets_total": int(Asset.query.filter(*_owned_filters(current_user, Asset.created_by)).count()),
                "assets_by_kind": {
                    "ulg": int(kind_counts.get("ulg", 0)),
                    "csv": int(kind_counts.get("csv", 0)),
                },
                "runs_total": int(RunRecord.query.filter(*_owned_filters(current_user, RunRecord.created_by)).count()),
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
                "risk_trend": _risk_trend(current_user),
                "model_summary": _model_summary(),
                "field_contract": _field_contract(runtime_status, current_user),
                "telemetry_summary": _telemetry_summary(runtime_status, current_user),
                "recent_evidence": _recent_evidence(current_user),
                "similar_reports": _similar_reports(fault_summary.get("dominant_label"), current_user),
            },
        }
    )

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


def register_routes(bp):
    bp.add_url_rule(
        "/overview", endpoint="get_overview", view_func=token_required(get_overview), methods=["GET"]
    )
    bp.add_url_rule(
        "/models", endpoint="get_models", view_func=token_required(get_models), methods=["GET"]
    )
