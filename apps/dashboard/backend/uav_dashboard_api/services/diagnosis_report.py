from __future__ import annotations

import csv
from datetime import datetime
from pathlib import Path

from flask import current_app

from uav_dashboard_api import db
from uav_dashboard_api.models.asset import Asset
from uav_dashboard_api.models.diagnosis_report import DiagnosisReport
from uav_dashboard_api.models.run_record import RunRecord
from uav_dashboard_api.utils.path_store import resolve_local_path

OFFLINE_MULTICLASS_TASK = "offline_multiclass"
OFFLINE_CLASS_SET = ["normal", "motor", "sensor"]
REPORT_INTENDED_USE = "飞行日志 / CSV 回放离线故障诊断辅助分析"
RULES_VERSION_DEFAULT = "rule_v1"
CSV_QUALITY_SAMPLE_ROWS = 2000


LEGACY_RECOMMENDATIONS = {
    "normal": [
        "No immediate fault detected.",
        "Keep routine inspection and calibration schedule.",
    ],
    "battery_failure": [
        "Inspect battery health, voltage sag, and wiring harness.",
        "Replace battery pack if warning/fault persists.",
    ],
    "motor_failure": [
        "Check motor temperature, vibration, and phase current.",
        "Inspect ESC output and replace motor if abnormal.",
    ],
    "motor": [
        "Check motor temperature, vibration, and phase current.",
        "Inspect ESC output and replace motor if abnormal.",
    ],
    "prop_imbalance": [
        "Inspect propeller damage and balancing status.",
        "Replace propellers and verify motor shaft alignment.",
    ],
    "propeller": [
        "Inspect propeller damage and balancing status.",
        "Replace propellers and verify motor shaft alignment.",
    ],
    "sensor": [
        "Inspect IMU and related sensor calibration status.",
        "Check sensor wiring, mounting, and telemetry consistency.",
    ],
    "thrust_loss": [
        "Verify thrust command/output consistency and airframe load.",
        "Inspect propulsion chain for drag or actuator saturation.",
    ],
}

LABEL_TEXT_MAP = {
    "normal": "状态正常",
    "battery_failure": "电池故障",
    "motor_failure": "电机故障",
    "motor": "电机故障",
    "prop_imbalance": "螺旋桨故障",
    "propeller": "螺旋桨故障",
    "sensor": "传感器故障",
    "thrust_loss": "推力异常",
}

SEVERITY_TEXT_MAP = {
    "none": ("正常", "风险可控"),
    "light": ("轻度", "轻度风险"),
    "medium": ("中度", "中度风险"),
    "heavy": ("重度", "高风险"),
    "unknown": ("未知", "待人工确认"),
}

PRIORITY_TEXT_MAP = {
    "P0": "立即处理",
    "P1": "优先处理",
    "P2": "安排检查",
    "P3": "持续观察",
}

METRIC_TEXT_MAP = {
    "prediction_confidence": "模型判断置信度",
    "motor_imbalance_ratio": "电机输出均衡性异常",
    "esc_rpm_imbalance_ratio": "ESC 转速一致性异常",
    "imbalanced_prop_metric_p95": "螺旋桨失衡特征偏高",
    "gyro_vibration_p95": "陀螺振动偏高",
    "accel_vibration_p95": "加速度振动偏高",
    "battery_warning_max": "电池告警触发",
    "battery_faults_max": "电池故障计数偏高",
    "battery_remaining_min": "剩余电量偏低",
    "battery_voltage_drop_ratio": "电压跌落偏大",
    "fault_intensity_mean": "故障强度均值偏高",
    "fault_severity_p95": "高分位故障严重度偏高",
    "fault_severity_max": "峰值故障严重度偏高",
    "thrust_saturation_ratio": "推力饱和比例偏高",
    "unalloc_thrust_z_p95": "未分配推力异常",
    "vz_down_p95": "下降速度异常偏高",
}

def _normalize_probs(raw_probs: dict | None) -> dict[str, float]:
    output = {}
    for key, value in (raw_probs or {}).items():
        try:
            output[str(key)] = float(value)
        except (TypeError, ValueError):
            continue
    return output


def _build_topk(probs: dict[str, float], topk: int = 3) -> list[dict]:
    ranked = sorted(probs.items(), key=lambda item: item[1], reverse=True)
    return [{"label": label, "prob": prob} for label, prob in ranked[: max(1, topk)]]


def _normalize_severity(raw: dict | None) -> dict:
    if not isinstance(raw, dict):
        return {"level": "unknown", "score": None, "method": "legacy", "evidence": []}
    level = str(raw.get("level", "unknown")).strip() or "unknown"
    score = raw.get("score")
    try:
        score = None if score is None else float(score)
    except (TypeError, ValueError):
        score = None
    method = str(raw.get("method", "rule_v1")).strip() or "rule_v1"
    evidence = raw.get("evidence")
    if not isinstance(evidence, list):
        evidence = []
    return {
        "level": level,
        "score": score,
        "method": method,
        "evidence": evidence,
    }


def _normalize_recommendations(raw, prediction_label: str) -> list[dict]:
    rows: list[dict] = []
    if isinstance(raw, list):
        for item in raw:
            if isinstance(item, str):
                title = item.strip()
                if title:
                    rows.append(
                        {
                            "title": title,
                            "priority": "P2",
                            "reason_codes": ["legacy_text"],
                        }
                    )
                continue
            if not isinstance(item, dict):
                continue
            title = str(item.get("title", "")).strip()
            if not title:
                continue
            reason_codes = item.get("reason_codes")
            if not isinstance(reason_codes, list):
                reason_codes = []
            rows.append(
                {
                    "title": title,
                    "priority": str(item.get("priority", "P2")).strip() or "P2",
                    "reason_codes": [
                        str(code) for code in reason_codes if str(code).strip()
                    ],
                }
            )
    if rows:
        return rows
    return [
        {"title": text, "priority": "P2", "reason_codes": ["legacy_default"]}
        for text in LEGACY_RECOMMENDATIONS.get(
            prediction_label,
            [
                "Review sensor data and run additional verification.",
                "Confirm anomaly with maintenance and flight logs.",
            ],
        )
    ]


def _safe_int(value) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _safe_float(value) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _resolve_input_csv(result: dict, debug: dict, asset: Asset | None) -> str:
    for candidate in (
        result.get("input_csv"),
        debug.get("csv_path"),
        (asset.get_meta() or {}).get("prepared_csv_path") if asset else None,
    ):
        text = str(candidate or "").strip()
        if text:
            return text
    return ""


def _csv_row_looks_like_header(row: list[str]) -> bool:
    non_empty = [str(cell).strip() for cell in row if str(cell).strip()]
    if not non_empty:
        return False
    numeric_cells = 0
    for cell in non_empty:
        try:
            float(cell)
        except ValueError:
            continue
        numeric_cells += 1
    return numeric_cells < max(1, len(non_empty) // 2)


def _count_csv_samples(input_csv: str) -> int | None:
    if not input_csv:
        return None
    try:
        resolved = Path(resolve_local_path(input_csv, current_app.config))
    except Exception:
        return None

    if not resolved.exists() or not resolved.is_file():
        return None

    try:
        with resolved.open("r", encoding="utf-8-sig", newline="") as fp:
            reader = csv.reader(fp)
            first_row: list[str] | None = None
            row_count = 0
            for row in reader:
                if not any(str(cell).strip() for cell in row):
                    continue
                if first_row is None:
                    first_row = row
                    if _csv_row_looks_like_header(row):
                        continue
                row_count += 1
            return row_count if first_row is not None else 0
    except Exception:
        return None


def _inspect_csv_quality(input_csv: str) -> dict:
    if not input_csv:
        return {}
    try:
        resolved = Path(resolve_local_path(input_csv, current_app.config))
    except Exception:
        return {}
    if not resolved.exists() or not resolved.is_file():
        return {}

    try:
        with resolved.open("r", encoding="utf-8-sig", newline="") as fp:
            reader = csv.reader(fp)
            first_row: list[str] | None = None
            feature_count = 0
            sampled_rows: list[list[str]] = []
            empty_markers = {"", "nan", "none", "null"}

            for row in reader:
                if not any(str(cell).strip() for cell in row):
                    continue
                if first_row is None:
                    first_row = row
                    feature_count = len(row)
                    if _csv_row_looks_like_header(row):
                        continue

                values = list(row[:feature_count])
                if len(values) < feature_count:
                    values.extend([""] * (feature_count - len(values)))
                sampled_rows.append(values)
                if len(sampled_rows) >= CSV_QUALITY_SAMPLE_ROWS:
                    break
    except Exception:
        return {}

    if first_row is None or feature_count <= 0:
        return {}

    total_cells = len(sampled_rows) * feature_count
    missing_cells = 0
    non_empty_by_col = [0] * feature_count
    numeric_by_col = [0] * feature_count
    invalid_by_col = [0] * feature_count

    for values in sampled_rows:
        for index, value in enumerate(values[:feature_count]):
            text = str(value).strip()
            if text.lower() in empty_markers:
                missing_cells += 1
                continue
            non_empty_by_col[index] += 1
            try:
                float(text)
            except ValueError:
                invalid_by_col[index] += 1
            else:
                numeric_by_col[index] += 1

    numeric_columns = [
        index
        for index, non_empty_count in enumerate(non_empty_by_col)
        if non_empty_count > 0 and numeric_by_col[index] >= max(1, int(non_empty_count * 0.5))
    ]
    invalid_denominator = sum(non_empty_by_col[index] for index in numeric_columns)
    invalid_cells = sum(invalid_by_col[index] for index in numeric_columns)

    return {
        "sampled_rows": len(sampled_rows),
        "feature_count": feature_count,
        "missing_value_ratio": (missing_cells / total_cells) if total_cells else None,
        "invalid_value_ratio": (invalid_cells / invalid_denominator) if invalid_denominator else 0.0,
        "sample_limit": CSV_QUALITY_SAMPLE_ROWS,
    }


def _build_data_quality(result: dict, asset: Asset | None) -> dict:
    debug = result.get("debug") if isinstance(result.get("debug"), dict) else {}
    warnings: list[str] = []

    input_csv = _resolve_input_csv(result, debug, asset)
    csv_quality = _inspect_csv_quality(input_csv)
    sample_count = _safe_int(debug.get("sample_count"))
    if sample_count is None and input_csv:
        sample_count = _count_csv_samples(input_csv)
    if sample_count is None:
        sample_count = _safe_int(csv_quality.get("sampled_rows"))

    window_count = _safe_int(debug.get("window_count"))
    feature_count = _safe_int(debug.get("feature_dim"))
    if feature_count is None:
        feature_count = _safe_int(debug.get("in_channels"))
    if feature_count is None:
        feature_count = _safe_int(csv_quality.get("feature_count"))

    missing_columns = debug.get("missing_columns")
    if not isinstance(missing_columns, list):
        missing_columns = []

    missing_value_ratio = _safe_float(debug.get("missing_value_ratio"))
    if missing_value_ratio is None:
        missing_value_ratio = _safe_float(csv_quality.get("missing_value_ratio"))
    invalid_value_ratio = _safe_float(debug.get("invalid_value_ratio"))
    if invalid_value_ratio is None:
        invalid_value_ratio = _safe_float(csv_quality.get("invalid_value_ratio"))

    time_range = debug.get("time_range")
    if not isinstance(time_range, dict):
        time_range = {"start": None, "end": None}

    if not input_csv:
        warnings.append("未记录输入 CSV 路径，数据追溯能力受限。")
    if sample_count is None:
        if debug.get("seq_len") not in (None, ""):
            warnings.append("仅记录模型序列长度，未记录真实样本数量；不将序列长度作为样本数量。")
        else:
            warnings.append("未记录样本数量，无法完整评估输入覆盖度。")
    elif sample_count <= 0:
        warnings.append("样本数量为 0，无法完整评估输入覆盖度。")
    if feature_count is None:
        warnings.append("未记录特征维度，部分质量指标待补充。")
    elif feature_count <= 0:
        warnings.append("特征数量为 0，部分质量指标待补充。")
    if missing_columns:
        warnings.append(f"存在缺失列：{', '.join(str(item) for item in missing_columns[:5])}。")
    if missing_value_ratio is not None and missing_value_ratio > 0.1:
        warnings.append(f"缺失值占比较高（约 {_format_percent(missing_value_ratio)}）。")
    if invalid_value_ratio is not None and invalid_value_ratio > 0.05:
        warnings.append(f"异常值占比较高（约 {_format_percent(invalid_value_ratio)}）。")

    if missing_columns or (
        missing_value_ratio is not None and missing_value_ratio > 0.1
    ) or (invalid_value_ratio is not None and invalid_value_ratio > 0.05):
        level = "poor"
    elif (
        not input_csv
        or sample_count is None
        or sample_count <= 0
        or feature_count is None
        or feature_count <= 0
    ):
        level = "warning"
    elif input_csv and sample_count is not None and feature_count is not None:
        level = "good"
    else:
        level = "unknown"

    return {
        "level": level,
        "sample_count": sample_count if sample_count is not None else None,
        "window_count": window_count if window_count is not None else 0,
        "feature_count": feature_count if feature_count is not None else 0,
        "missing_columns": missing_columns,
        "missing_value_ratio": missing_value_ratio,
        "invalid_value_ratio": invalid_value_ratio,
        "time_range": {
            "start": time_range.get("start"),
            "end": time_range.get("end"),
        },
        "warnings": warnings,
    }


def _build_traceability(
    *,
    run: RunRecord,
    asset: Asset | None,
    model_name: str,
    result: dict,
    severity: dict,
    generated_at: str,
) -> dict:
    debug = result.get("debug") if isinstance(result.get("debug"), dict) else {}
    rules_version = str(severity.get("method") or RULES_VERSION_DEFAULT).strip() or RULES_VERSION_DEFAULT
    return {
        "run_id": run.id,
        "asset_id": run.asset_id,
        "asset_filename": asset.filename if asset else "",
        "input_csv": _resolve_input_csv(result, debug, asset),
        "model_name": model_name,
        "model_version": str(debug.get("model_version") or "").strip(),
        "model_hash": str(debug.get("model_hash") or "").strip(),
        "rules_version": rules_version,
        "generated_at": generated_at,
    }


def _build_model_card(result: dict, model_name: str) -> dict:
    labels = result.get("labels")
    class_set = []
    if isinstance(labels, list):
        class_set = [str(item).strip() for item in labels if str(item).strip()]
    if not class_set:
        class_set = list(OFFLINE_CLASS_SET)

    debug = result.get("debug") if isinstance(result.get("debug"), dict) else {}
    metrics = {}
    for key in ("best_val_f1", "best_val_acc", "test_f1", "test_acc"):
        value = _safe_float(debug.get(key))
        if value is not None:
            metrics[key] = value

    limitations = []
    if not debug.get("model_dir"):
        limitations.append("未记录模型目录，无法核对模型文件版本。")
    if debug.get("norm_source") == "sample_zscore":
        limitations.append("本次推理使用样本内归一化，跨工况对比时需谨慎。")

    return {
        "task": OFFLINE_MULTICLASS_TASK,
        "class_set": class_set,
        "intended_use": REPORT_INTENDED_USE,
        "limitations": limitations,
        "metrics": metrics,
        "model_name": model_name,
    }


def _build_report_confidence(
    *,
    confidence: float,
    severity: dict,
    data_quality: dict,
) -> dict:
    confidence = _safe_float(confidence) or 0.0
    reasons: list[str] = []
    quality_level = str(data_quality.get("level") or "unknown").strip() or "unknown"
    evidence = severity.get("evidence")
    rule_evidence_count = 0
    model_evidence_count = 0
    if isinstance(evidence, list):
        for item in evidence:
            if not isinstance(item, dict) or item.get("status") != "used":
                continue
            if item.get("source") == "model":
                model_evidence_count += 1
            else:
                rule_evidence_count += 1

    if quality_level == "poor":
        reasons.append("输入数据质量较差，结论支撑度不足。")
    elif quality_level == "warning":
        reasons.append("输入数据质量信息不完整，建议结合原始日志复核。")
    elif quality_level == "good":
        reasons.append("输入数据质量记录完整，可支撑当前结论。")

    if confidence >= 0.85:
        reasons.append(f"模型主分类置信度较高（{_format_percent(confidence)}）。")
    elif confidence >= 0.65:
        reasons.append(f"模型主分类置信度中等（{_format_percent(confidence)}）。")
    else:
        reasons.append(f"模型主分类置信度偏低（{_format_percent(confidence)}）。")

    if severity.get("method") == "model_confidence_fallback":
        reasons.append("规则指标不足，本次风险等级由模型置信度兜底生成。")
    elif rule_evidence_count > 0:
        reasons.append(f"规则证据链包含 {rule_evidence_count} 条有效指标支撑。")
    else:
        reasons.append("规则证据链较少，建议人工复核关键参数。")

    if quality_level == "poor" or confidence < 0.55:
        level = "low"
    elif quality_level == "good" and confidence >= 0.8 and rule_evidence_count > 0:
        level = "high"
    elif quality_level in {"good", "warning"} and confidence >= 0.65:
        level = "medium"
    else:
        level = "unknown"

    evidence_bonus = 0.2 if rule_evidence_count else (0.08 if model_evidence_count else 0.0)
    score = round(
        min(
            1.0,
            max(
                0.0,
                confidence * 0.7
                + evidence_bonus
                + (0.1 if quality_level == "good" else 0.0),
            ),
        ),
        3,
    )

    return {
        "level": level,
        "score": score,
        "reasons": reasons,
    }


def _build_review_defaults() -> dict:
    return {
        "status": "pending",
        "reviewer": "",
        "reviewed_at": "",
        "conclusion": "",
        "maintenance_notes": "",
    }


def _evidence_sentence(item: dict) -> str:
    metric = _metric_text(item.get("metric", ""))
    if item.get("status") == "used":
        source = "模型判断" if item.get("source") == "model" else "规则指标"
        value = _format_metric_value(item.get("value"))
        score = _format_metric_value(item.get("score")) if item.get("score") is not None else "-"
        if item.get("source") == "model":
            value = _format_percent(item.get("value"))
        return f"{source}：{metric}为 {value}，规则得分 {score}。"
    reason = str(item.get("reason_text") or "").strip()
    if not reason:
        reason = "模型未提供可用置信度。" if item.get("source") == "model" else "未从当前 CSV 中提取到对应规则指标。"
    prefix = "模型证据缺失" if item.get("source") == "model" else "规则指标缺失"
    return f"{prefix}：{metric}，{reason}"


def _build_readable_summary(
    *,
    prediction_label: str,
    confidence: float,
    severity: dict,
    topk: list[dict],
    recommendations: list[dict],
    data_quality: dict,
) -> dict:
    label_text = _fault_label_text(prediction_label)
    severity_level = str(severity.get("level") or "unknown").strip() or "unknown"
    severity_badge, severity_risk = _severity_text(severity_level)
    confidence_text = _format_percent(confidence)

    if prediction_label == "normal" or severity_level == "none":
        conclusion = f"本次离线诊断未发现明显故障，模型对正常状态的判断置信度为 {confidence_text}。"
        risk = "当前风险可控，建议保持例行飞前/飞后检查并持续留存飞行日志。"
    else:
        conclusion = f"本次离线诊断识别为{label_text}，主分类置信度为 {confidence_text}。"
        if severity_level == "heavy":
            risk = f"风险等级为高风险，建议优先复核相关部件并根据任务要求考虑暂停继续飞行。"
        elif severity_level == "medium":
            risk = f"风险等级为中度风险，建议尽快结合原始日志和维护记录复核。"
        elif severity_level == "light":
            risk = f"风险等级为需关注，建议安排针对性检查并继续观察后续日志。"
        else:
            risk = f"风险等级仍需人工确认，建议结合原始日志复核。"

    evidence_items = [_probability_summary(topk)]
    for item in (severity.get("evidence") or [])[:4]:
        if isinstance(item, dict):
            evidence_items.append(_evidence_sentence(item))

    recommendation_items = []
    for item in recommendations[:4]:
        if not isinstance(item, dict):
            continue
        title = str(item.get("title") or "").strip()
        if not title:
            continue
        recommendation_items.append(f"[{_priority_text(item.get('priority', 'P2'))}] {title}")

    cautions = []
    if severity.get("method") == "model_confidence_fallback":
        cautions.append("规则指标不足，本次风险等级按模型置信度兜底生成，应结合原始日志复核。")
    warnings = data_quality.get("warnings")
    if isinstance(warnings, list):
        cautions.extend(str(item) for item in warnings[:2] if str(item).strip())

    return {
        "conclusion": conclusion,
        "risk": risk,
        "evidence": evidence_items,
        "recommendations": recommendation_items,
        "cautions": cautions,
        "severity_text": severity_risk,
        "severity_badge": severity_badge,
    }


def _build_summary(
    run: RunRecord,
    asset: Asset | None,
    model_name: str,
    prediction_label: str,
    confidence: float,
    probs: dict[str, float],
    key_params: dict,
    severity: dict,
    recommendations: list[dict],
    recommendation_meta: dict,
    result: dict | None = None,
) -> dict:
    generated_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    result = result if isinstance(result, dict) else {}
    data_quality = _build_data_quality(result, asset)
    traceability = _build_traceability(
        run=run,
        asset=asset,
        model_name=model_name,
        result=result,
        severity=severity,
        generated_at=generated_at,
    )
    model_card = _build_model_card(result, model_name)
    report_confidence = _build_report_confidence(
        confidence=confidence,
        severity=severity,
        data_quality=data_quality,
    )
    topk = _build_topk(probs, topk=3)
    readable = _build_readable_summary(
        prediction_label=prediction_label,
        confidence=confidence,
        severity=severity,
        topk=topk,
        recommendations=recommendations,
        data_quality=data_quality,
    )
    return {
        "basic": {
            "run_id": run.id,
            "asset_id": run.asset_id,
            "asset_filename": asset.filename if asset else "",
            "model_name": model_name,
            "run_created_at": run.created_at.strftime("%Y-%m-%d %H:%M:%S")
            if run.created_at
            else "",
            "report_generated_at": generated_at,
        },
        "conclusion": {
            "label": prediction_label,
            "confidence": confidence,
        },
        "severity": severity,
        "topk": topk,
        "key_params": key_params,
        "recommendations": recommendations,
        "recommendation_meta": recommendation_meta,
        "readable": readable,
        "data_quality": data_quality,
        "traceability": traceability,
        "model_card": model_card,
        "report_confidence": report_confidence,
        "review": _build_review_defaults(),
    }


def _fault_label_text(label: str) -> str:
    return LABEL_TEXT_MAP.get(str(label or "").strip(), str(label or "待确认异常") or "待确认异常")


def _severity_text(level: str) -> tuple[str, str]:
    return SEVERITY_TEXT_MAP.get(str(level or "unknown").strip(), SEVERITY_TEXT_MAP["unknown"])


def _priority_text(priority: str) -> str:
    return PRIORITY_TEXT_MAP.get(str(priority or "P2").strip().upper(), "安排检查")


def _metric_text(metric: str) -> str:
    value = str(metric or "").strip()
    return METRIC_TEXT_MAP.get(value, value.replace("_", " ") or "未知指标")


def _format_percent(value: float | None) -> str:
    try:
        return f"{float(value) * 100:.3f}%"
    except (TypeError, ValueError):
        return "-"


def _format_metric_value(value) -> str:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return "-" if value in (None, "") else str(value)
    if abs(numeric) >= 100:
        return f"{numeric:.1f}"
    if abs(numeric) >= 1:
        return f"{numeric:.3f}"
    return f"{numeric:.4f}"


def _probability_summary(rows: list[dict]) -> str:
    if not rows:
        return "模型未提供完整的类别概率分布。"
    top1 = rows[0]
    if len(rows) == 1:
        return f"模型当前主要判断为{_fault_label_text(top1.get('label', ''))}，对应概率约为{_format_percent(top1.get('prob'))}。"
    top2 = rows[1]
    return (
        f"模型当前主要判断为{_fault_label_text(top1.get('label', ''))}（{_format_percent(top1.get('prob'))}），"
        f"次高可能为{_fault_label_text(top2.get('label', ''))}（{_format_percent(top2.get('prob'))}）。"
    )


def _build_key_params(result: dict) -> dict:
    debug = result.get("debug") if isinstance(result.get("debug"), dict) else {}
    key_params = {}

    for key in ("asset_kind", "input_csv", "model"):
        value = result.get(key)
        if value not in (None, ""):
            key_params[key] = value

    labels = result.get("labels")
    if isinstance(labels, list) and labels:
        key_params["label_count"] = len(labels)
        key_params["labels"] = labels[:5]

    topk = result.get("topk")
    if isinstance(topk, list) and topk:
        key_params["topk"] = topk[:3]

    for key in (
        "window_size",
        "stride",
        "sequence_length",
        "feature_dim",
        "sample_count",
    ):
        value = debug.get(key)
        if value not in (None, ""):
            key_params[key] = value

    severity = result.get("severity")
    if isinstance(severity, dict):
        if severity.get("level"):
            key_params["severity_level"] = severity.get("level")
        if severity.get("score") is not None:
            key_params["severity_score"] = severity.get("score")
        if severity.get("method"):
            key_params["severity_method"] = severity.get("method")

    recommendation_meta = result.get("recommendation_meta")
    if isinstance(recommendation_meta, dict) and recommendation_meta:
        key_params["recommendation_strategy"] = recommendation_meta.get("strategy")
        key_params["recommendation_fallback_used"] = recommendation_meta.get(
            "fallback_used"
        )

    return key_params


def create_or_get_diagnosis_report(
    run: RunRecord, created_by: int | None = None
) -> tuple[DiagnosisReport, bool]:
    existing = DiagnosisReport.query.filter_by(run_id=run.id).first()
    if existing is not None:
        return existing, False

    if run.task != OFFLINE_MULTICLASS_TASK:
        raise ValueError("仅支持离线分类诊断任务")
    if run.status != "success":
        raise ValueError("仅可为成功任务生成报告")

    result = run.get_result() or {}
    prediction = result.get("prediction") or {}
    if not prediction:
        raise ValueError("任务结果缺少 prediction")

    probs = _normalize_probs(result.get("probs"))
    prediction_label = str(prediction.get("label", "")).strip()
    if not prediction_label:
        raise ValueError("预测标签缺失")

    try:
        confidence = float(prediction.get("confidence", 0.0))
    except (TypeError, ValueError):
        confidence = 0.0

    params = run.get_params()
    model_name = str(
        params.get("model") or result.get("debug", {}).get("model_name") or ""
    ).strip()
    asset = db.session.get(Asset, run.asset_id) if run.asset_id else None
    key_params = _build_key_params(result)
    severity = _normalize_severity(result.get("severity"))
    recommendations = _normalize_recommendations(
        result.get("recommendations"), prediction_label=prediction_label
    )
    recommendation_meta = result.get("recommendation_meta")
    if not isinstance(recommendation_meta, dict):
        recommendation_meta = {}
    recommendation_meta = {
        "strategy": recommendation_meta.get("strategy")
        or severity.get("method")
        or "legacy",
        "fallback_used": bool(recommendation_meta.get("fallback_used", False)),
        "severity_level": recommendation_meta.get("severity_level")
        or severity.get("level"),
        "candidate_count": recommendation_meta.get("candidate_count")
        or len(recommendations),
    }
    unified_decision = (
        result.get("unified_decision")
        if isinstance(result.get("unified_decision"), dict)
        else {}
    )
    summary = _build_summary(
        run=run,
        asset=asset,
        model_name=model_name,
        prediction_label=prediction_label,
        confidence=confidence,
        probs=probs,
        key_params=key_params,
        severity=severity,
        recommendations=recommendations,
        recommendation_meta=recommendation_meta,
        result=result,
    )
    if unified_decision:
        summary["decision"] = unified_decision

    report = DiagnosisReport(
        run_id=run.id,
        asset_id=run.asset_id,
        model_name=model_name,
        prediction_label=prediction_label,
        confidence=confidence,
        created_by=created_by,
    )
    report.set_probs(probs)
    report.set_summary(summary)
    db.session.add(report)
    db.session.commit()
    return report, True
