from __future__ import annotations

from datetime import datetime
import json
from pathlib import Path
import shutil
import subprocess
import tempfile
import textwrap

from flask import current_app
from pypdf import PdfWriter
from pypdf.generic import DecodedStreamObject, DictionaryObject, NameObject

from uav_dashboard_api import db
from uav_dashboard_api.models.asset import Asset
from uav_dashboard_api.models.diagnosis_report import DiagnosisReport
from uav_dashboard_api.models.run_record import RunRecord
from uav_dashboard_api.utils.path_store import serialize_local_path
from uav_dashboard_api.utils.pdf_fonts import build_pdf_font_candidates

OFFLINE_MULTICLASS_TASK = "offline_multiclass"
OFFLINE_CLASS_SET = ["normal", "motor", "sensor"]
RULES_VERSION_DEFAULT = "rule_v1"
DATA_QUALITY_LEVEL_TEXT = {
    "good": "良好",
    "warning": "需关注",
    "poor": "较差",
    "unknown": "未知",
}
REPORT_CONFIDENCE_LEVEL_TEXT = {
    "high": "高",
    "medium": "中",
    "low": "低",
    "unknown": "未知",
}
REVIEW_STATUS_TEXT = {
    "pending": "待复核",
    "confirmed": "已确认",
    "false_alarm": "误报",
    "ignored": "已忽略",
}


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

REPORTLAB_FONT_CANDIDATES = build_pdf_font_candidates()

_REPORTLAB_PYTHON_CACHE = {"checked": False, "python": None}



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


def _escape_pdf_text(text: str) -> str:
    safe = text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
    return "".join(char if ord(char) < 128 else "?" for char in safe)


def _wrap_lines(lines: list[str], width: int = 90) -> list[str]:
    wrapped = []
    for line in lines:
        pieces = textwrap.wrap(line, width=width) if line else [""]
        wrapped.extend(pieces or [""])
    return wrapped


def _attach_helvetica_font(writer: PdfWriter, page):
    font = DictionaryObject()
    font.update(
        {
            NameObject("/Type"): NameObject("/Font"),
            NameObject("/Subtype"): NameObject("/Type1"),
            NameObject("/BaseFont"): NameObject("/Helvetica"),
            NameObject("/Encoding"): NameObject("/WinAnsiEncoding"),
        }
    )
    font_ref = writer._add_object(font)

    resources = page.get("/Resources")
    if resources is None:
        resources = DictionaryObject()
        page[NameObject("/Resources")] = resources

    fonts = resources.get("/Font")
    if fonts is None:
        fonts = DictionaryObject()
        resources[NameObject("/Font")] = fonts

    fonts[NameObject("/F1")] = font_ref


def _write_pdf_fallback(path: Path, lines: list[str]):
    writer = PdfWriter()
    page_width = 595
    page_height = 842
    line_height = 16
    top_margin = 790
    bottom_margin = 56

    pending_lines = _wrap_lines(lines, width=92)
    index = 0
    while index < len(pending_lines):
        page = writer.add_blank_page(width=page_width, height=page_height)
        _attach_helvetica_font(writer, page)

        y = top_margin
        commands = []
        while index < len(pending_lines) and y > bottom_margin:
            content = _escape_pdf_text(pending_lines[index])
            commands.append(f"BT /F1 11 Tf 48 {y} Td ({content}) Tj ET")
            y -= line_height
            index += 1

        stream = DecodedStreamObject()
        stream.set_data("\n".join(commands).encode("latin-1", errors="replace"))
        page[NameObject("/Contents")] = writer._add_object(stream)

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as fp:
        writer.write(fp)


def _reportlab_renderer_script_path() -> Path:
    return Path(__file__).resolve().parents[1] / "utils" / "reportlab_pdf_renderer.py"


def _find_reportlab_python() -> str | None:
    if _REPORTLAB_PYTHON_CACHE["checked"]:
        return _REPORTLAB_PYTHON_CACHE["python"]

    candidates = []
    for item in ("/usr/bin/python3", shutil.which("python3"), shutil.which("python")):
        if item and item not in candidates:
            candidates.append(item)

    for candidate in candidates:
        try:
            subprocess.run(
                [candidate, "-c", "import reportlab"],
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        except Exception:
            continue
        _REPORTLAB_PYTHON_CACHE["checked"] = True
        _REPORTLAB_PYTHON_CACHE["python"] = candidate
        return candidate

    _REPORTLAB_PYTHON_CACHE["checked"] = True
    _REPORTLAB_PYTHON_CACHE["python"] = None
    return None


def _render_pdf_with_reportlab(path: Path, lines: list[str]) -> bool:
    renderer = _reportlab_renderer_script_path()
    python_executable = _find_reportlab_python()
    if not renderer.exists() or not python_executable:
        return False

    payload = {
        "lines": lines,
        "font_candidates": REPORTLAB_FONT_CANDIDATES,
    }
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", suffix=".json", delete=False) as fp:
        json.dump(payload, fp, ensure_ascii=False)
        payload_path = Path(fp.name)

    try:
        subprocess.run(
            [python_executable, str(renderer), str(payload_path), str(path)],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        return True
    except Exception as exc:  # pragma: no cover - fallback path
        current_app.logger.warning("reportlab PDF 渲染失败，回退简版 PDF: %s", exc)
        return False
    finally:
        payload_path.unlink(missing_ok=True)


def _write_pdf(path: Path, lines: list[str]):
    if _render_pdf_with_reportlab(path, lines):
        return
    _write_pdf_fallback(path, lines)


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


def _build_data_quality(result: dict, asset: Asset | None) -> dict:
    debug = result.get("debug") if isinstance(result.get("debug"), dict) else {}
    warnings: list[str] = []

    sample_count = _safe_int(debug.get("sample_count"))
    if sample_count is None:
        sample_count = _safe_int(debug.get("seq_len"))

    window_count = _safe_int(debug.get("window_count"))
    feature_count = _safe_int(debug.get("feature_dim"))
    if feature_count is None:
        feature_count = _safe_int(debug.get("in_channels"))

    missing_columns = debug.get("missing_columns")
    if not isinstance(missing_columns, list):
        missing_columns = []

    missing_value_ratio = _safe_float(debug.get("missing_value_ratio"))
    invalid_value_ratio = _safe_float(debug.get("invalid_value_ratio"))

    time_range = debug.get("time_range")
    if not isinstance(time_range, dict):
        time_range = {"start": None, "end": None}

    input_csv = _resolve_input_csv(result, debug, asset)
    if not input_csv:
        warnings.append("未记录输入 CSV 路径，数据追溯能力受限。")
    if sample_count is None:
        warnings.append("未记录样本数量，无法完整评估输入覆盖度。")
    if feature_count is None:
        warnings.append("未记录特征维度，部分质量指标待补充。")
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
    elif not input_csv or sample_count is None or feature_count is None:
        level = "warning"
    elif input_csv and sample_count is not None:
        level = "good"
    else:
        level = "unknown"

    return {
        "level": level,
        "sample_count": sample_count if sample_count is not None else 0,
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
        "intended_use": "RflySim/RflyMAD 离线日志故障诊断辅助分析",
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
    reasons: list[str] = []
    quality_level = str(data_quality.get("level") or "unknown").strip() or "unknown"
    evidence = severity.get("evidence")
    evidence_count = len(evidence) if isinstance(evidence, list) else 0

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

    if evidence_count > 0:
        reasons.append(f"规则证据链包含 {evidence_count} 条指标支撑。")
    else:
        reasons.append("规则证据链较少，建议人工复核关键参数。")

    if quality_level == "poor" or confidence < 0.55:
        level = "low"
    elif quality_level == "good" and confidence >= 0.8 and evidence_count > 0:
        level = "high"
    elif quality_level in {"good", "warning"} and confidence >= 0.65:
        level = "medium"
    else:
        level = "unknown"

    score = round(min(1.0, max(0.0, confidence * 0.7 + (0.2 if evidence_count else 0.05) + (0.1 if quality_level == "good" else 0.0))), 3)

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
        "topk": _build_topk(probs, topk=3),
        "key_params": key_params,
        "recommendations": recommendations,
        "recommendation_meta": recommendation_meta,
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
        return f"{float(value) * 100:.1f}%"
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


def _format_key_param_value(value) -> str:
    if isinstance(value, list):
        return ", ".join(_format_key_param_value(item) for item in value)
    if isinstance(value, dict):
        return ", ".join(f"{k}={_format_key_param_value(v)}" for k, v in value.items())
    return _format_metric_value(value)


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


def _recommendation_reason_text(reason_codes) -> str:
    if not isinstance(reason_codes, list) or not reason_codes:
        return "建议结合飞行日志、维护记录和关键参数进行人工复核。"
    labels = []
    for code in reason_codes:
        code_text = str(code or "").strip()
        if not code_text:
            continue
        if code_text in {"legacy_text", "legacy_default"}:
            labels.append("既有经验规则")
        else:
            labels.append(_metric_text(code_text))
    return f"关注点：{'、'.join(labels)}" if labels else "建议结合飞行日志、维护记录和关键参数进行人工复核。"


def _pdf_text(value, fallback: str = "-") -> str:
    text = str(value or "").strip()
    return text if text else fallback


def _pdf_data_quality_lines(data_quality: dict) -> list[str]:
    if not isinstance(data_quality, dict):
        data_quality = {}
    level = str(data_quality.get("level") or "unknown").strip() or "unknown"
    level_text = DATA_QUALITY_LEVEL_TEXT.get(level, "未知")
    lines = [
        f"- 数据质量等级：{level_text}",
        f"- 样本数量：{data_quality.get('sample_count') if data_quality.get('sample_count') else '待补充'}",
        f"- 窗口数量：{data_quality.get('window_count') if data_quality.get('window_count') else '待补充'}",
        f"- 特征数量：{data_quality.get('feature_count') if data_quality.get('feature_count') else '待补充'}",
    ]
    missing_ratio = data_quality.get("missing_value_ratio")
    invalid_ratio = data_quality.get("invalid_value_ratio")
    lines.append(
        f"- 缺失值占比：{_format_percent(missing_ratio) if missing_ratio is not None else '待补充'}"
    )
    lines.append(
        f"- 异常值占比：{_format_percent(invalid_ratio) if invalid_ratio is not None else '待补充'}"
    )
    time_range = data_quality.get("time_range")
    if isinstance(time_range, dict):
        start = _pdf_text(time_range.get("start"), "待补充")
        end = _pdf_text(time_range.get("end"), "待补充")
        lines.append(f"- 时间范围：{start} ~ {end}")
    warnings = data_quality.get("warnings")
    if isinstance(warnings, list) and warnings:
        for warning in warnings[:3]:
            lines.append(f"- 提示：{warning}")
    else:
        lines.append("- 提示：无额外数据质量告警。")
    return lines


def _pdf_traceability_lines(traceability: dict) -> list[str]:
    if not isinstance(traceability, dict):
        traceability = {}
    return [
        f"- 任务 ID：{_pdf_text(traceability.get('run_id'))}",
        f"- 资产 ID：{_pdf_text(traceability.get('asset_id'))}",
        f"- 资产文件：{_pdf_text(traceability.get('asset_filename'), '未知')}",
        f"- 输入文件：{_pdf_text(traceability.get('input_csv'), '待补充')}",
        f"- 模型名称：{_pdf_text(traceability.get('model_name'), '未知')}",
        f"- 模型版本：{_pdf_text(traceability.get('model_version'), '待补充')}",
        f"- 模型哈希：{_pdf_text(traceability.get('model_hash'), '待补充')}",
        f"- 规则版本：{_pdf_text(traceability.get('rules_version'), RULES_VERSION_DEFAULT)}",
        f"- 报告生成时间：{_pdf_text(traceability.get('generated_at'), '待补充')}",
    ]


def _pdf_review_lines(review: dict) -> list[str]:
    if not isinstance(review, dict):
        review = {}
    status = str(review.get("status") or "pending").strip() or "pending"
    status_text = REVIEW_STATUS_TEXT.get(status, "待复核")
    lines = [
        f"- 复核状态：{status_text}",
        f"- 复核人：{_pdf_text(review.get('reviewer'), '待补充')}",
        f"- 复核时间：{_pdf_text(review.get('reviewed_at'), '待补充')}",
        f"- 复核结论：{_pdf_text(review.get('conclusion'), '待补充')}",
        f"- 维修备注：{_pdf_text(review.get('maintenance_notes'), '待补充')}",
    ]
    return lines


def _build_pdf_lines(summary: dict, probs: dict[str, float]) -> list[str]:
    basic = summary.get("basic", {})
    conclusion = summary.get("conclusion", {})
    severity = (
        summary.get("severity", {}) if isinstance(summary.get("severity"), dict) else {}
    )
    topk = summary.get("topk", [])
    key_params = summary.get("key_params", {})
    recommendations = summary.get("recommendations", [])
    recommendation_meta = (
        summary.get("recommendation_meta", {})
        if isinstance(summary.get("recommendation_meta"), dict)
        else {}
    )
    data_quality = (
        summary.get("data_quality", {})
        if isinstance(summary.get("data_quality"), dict)
        else {}
    )
    traceability = (
        summary.get("traceability", {})
        if isinstance(summary.get("traceability"), dict)
        else {}
    )
    model_card = (
        summary.get("model_card", {})
        if isinstance(summary.get("model_card"), dict)
        else {}
    )
    report_confidence = (
        summary.get("report_confidence", {})
        if isinstance(summary.get("report_confidence"), dict)
        else {}
    )
    review = summary.get("review", {}) if isinstance(summary.get("review"), dict) else {}

    probability_rows = [
        {"label": item.get("label", ""), "prob": item.get("prob", 0.0)}
        for item in topk
        if isinstance(item, dict)
    ]
    if not probability_rows:
        probability_rows = [
            {"label": label, "prob": prob}
            for label, prob in sorted(probs.items(), key=lambda entry: entry[1], reverse=True)[:3]
        ]

    label = str(conclusion.get("label", "")).strip()
    label_text = _fault_label_text(label)
    severity_level = str(severity.get("level", "unknown")).strip() or "unknown"
    severity_badge, severity_risk = _severity_text(severity_level)
    confidence_text = _format_percent(conclusion.get("confidence", 0.0))
    severity_score = severity.get("score")
    severity_score_text = _format_metric_value(severity_score) if severity_score is not None else "-"

    if label == "normal" or severity_level == "none":
        headline = "本次离线诊断未发现明显故障，当前状态正常。"
        risk_text = f"当前结果显示风险可控，模型对正常状态的判断置信度为 {confidence_text}，建议保持例行检查。"
    elif severity_level == "heavy":
        headline = f"本次离线诊断识别为{label_text}，当前属于高风险状态。"
        risk_text = f"当前结果属于高风险，模型以 {confidence_text} 的置信度判断为{label_text}，建议立即复核并优先安排检修。"
    elif severity_level == "medium":
        headline = f"本次离线诊断识别为{label_text}，已经出现较明显异常迹象。"
        risk_text = f"当前结果属于中度风险，模型以 {confidence_text} 的置信度判断为{label_text}，建议尽快结合飞行日志和维护记录复核。"
    elif severity_level == "light":
        headline = f"本次离线诊断提示{label_text}，建议继续跟踪变化。"
        risk_text = f"当前结果提示轻度异常，模型以 {confidence_text} 的置信度判断为{label_text}，建议持续观察并安排针对性检查。"
    else:
        headline = f"本次离线诊断提示{label_text}，但风险等级仍需人工确认。"
        risk_text = f"当前报告提示{label_text}，但风险分级暂不稳定，建议人工复核关键参数。"

    confidence_level = str(report_confidence.get("level") or "unknown").strip() or "unknown"
    confidence_level_text = REPORT_CONFIDENCE_LEVEL_TEXT.get(confidence_level, "未知")
    confidence_score_text = (
        _format_metric_value(report_confidence.get("score"))
        if report_confidence.get("score") is not None
        else "待补充"
    )
    class_set = model_card.get("class_set")
    class_set_text = (
        " / ".join(str(item) for item in class_set)
        if isinstance(class_set, list) and class_set
        else " / ".join(OFFLINE_CLASS_SET)
    )

    lines = [
        "基于数据驱动的无人机故障检测系统",
        "离线诊断报告",
        f"生成时间：{basic.get('report_generated_at', '')}",
        f"任务 ID：{basic.get('run_id', '')}    资产 ID：{basic.get('asset_id', '')}",
        f"资产文件：{basic.get('asset_filename', '')}",
        f"诊断模型：{basic.get('model_name', '')}",
        "",
        "一、诊断结论",
        headline,
        risk_text,
        "",
        "二、数据质量",
        *_pdf_data_quality_lines(data_quality),
        "",
        "三、主要判断依据",
        f"- 模型判断：{_probability_summary(probability_rows)}",
    ]

    severity_evidence = severity.get("evidence", [])
    if isinstance(severity_evidence, list) and severity_evidence:
        for item in severity_evidence[:3]:
            if not isinstance(item, dict):
                continue
            metric_text = _metric_text(item.get("metric", ""))
            value_text = _format_metric_value(item.get("value"))
            score_text = _format_metric_value(item.get("score")) if item.get("score") is not None else "-"
            lines.append(
                f"- {metric_text}：当前观测值 {value_text}，规则贡献分 {score_text}，状态 {item.get('status', 'unknown')}。"
            )

    elif key_params:
        for key, value in list(key_params.items())[:3]:
            lines.append(f"- {_metric_text(key)}：摘要值 {_format_key_param_value(value)}。")

    lines.extend(
        [
            "",
            "四、风险等级与可信度",
            f"- 风险等级：{severity_risk}（{severity_badge}）",
            f"- 严重度分数：{severity_score_text}",
            f"- 模型置信度：{confidence_text}",
            f"- 报告可信度：{confidence_level_text}（综合分 {confidence_score_text}）",
        ]
    )
    confidence_reasons = report_confidence.get("reasons")
    if isinstance(confidence_reasons, list) and confidence_reasons:
        for reason in confidence_reasons[:4]:
            lines.append(f"- 可信度说明：{reason}")
    else:
        lines.append("- 可信度说明：待补充")

    lines.extend(["", "五、建议动作"])
    if recommendations:
        for item in recommendations:
            if isinstance(item, dict):
                title = str(item.get("title", "")).strip()
                if not title:
                    continue
                priority = _priority_text(item.get("priority", "P2"))
                reason_text = _recommendation_reason_text(item.get("reason_codes"))
                lines.append(f"- [{priority}] {title}")
                lines.append(f"  {reason_text}")
            elif item:
                lines.append(f"- [安排检查] {item}")
    else:
        lines.append("- [安排检查] 继续按既定巡检周期执行例行检查")

    lines.extend(
        [
            "",
            "六、模型与规则追溯",
            *_pdf_traceability_lines(traceability),
            f"- 任务类型：{_pdf_text(model_card.get('task'), OFFLINE_MULTICLASS_TASK)}",
            f"- 类别集合：{class_set_text}",
            f"- 适用场景：{_pdf_text(model_card.get('intended_use'), '待补充')}",
            "",
            "七、人工复核记录",
            *_pdf_review_lines(review),
            "",
            "八、技术摘要",
        ]
    )
    lines.append(f"- 原始标签：{label}")
    lines.append(f"- 强度分数：{severity_score_text}")
    lines.append(f"- 评估方法：{severity.get('method', 'rule_v1')}")
    if recommendation_meta:
        lines.append(
            f"- 推荐策略：{recommendation_meta.get('strategy', '-')}, fallback_used={recommendation_meta.get('fallback_used', False)}"
        )

    for key, value in list(key_params.items())[:5]:
        lines.append(f"- {_metric_text(key)}：{_format_key_param_value(value)}")

    return lines


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

    report_filename = (
        f"diagnosis_run_{run.id}_{datetime.now().strftime('%Y%m%d%H%M%S')}.pdf"
    )
    report_path = Path(current_app.config["REPORTS_DIR"]) / report_filename
    _write_pdf(report_path, _build_pdf_lines(summary, probs))

    report = DiagnosisReport(
        run_id=run.id,
        asset_id=run.asset_id,
        model_name=model_name,
        prediction_label=prediction_label,
        confidence=confidence,
        pdf_path=serialize_local_path(report_path),
        created_by=created_by,
    )
    report.set_probs(probs)
    report.set_summary(summary)
    db.session.add(report)
    db.session.commit()
    return report, True
