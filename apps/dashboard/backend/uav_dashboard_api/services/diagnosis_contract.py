from __future__ import annotations

from typing import Any


CANONICAL_SEVERITY_LEVELS = {"none", "light", "medium", "heavy", "unknown"}


def canonical_severity_from_probability(probability: float | None) -> str:
    try:
        prob = float(probability)
    except (TypeError, ValueError):
        return "unknown"
    if prob >= 0.9:
        return "heavy"
    if prob >= 0.75:
        return "medium"
    if prob >= 0.0:
        return "light"
    return "unknown"


def legacy_severity_from_canonical(level: str | None) -> str:
    mapping = {
        "none": "low",
        "light": "low",
        "medium": "medium",
        "heavy": "critical",
        "unknown": "medium",
    }
    return mapping.get(str(level or "unknown"), "medium")


def normalize_recommendations(
    items: list[dict[str, Any]] | list[str] | None,
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for item in items or []:
        if isinstance(item, str):
            title = item.strip()
            if title:
                output.append(
                    {"title": title, "priority": "P2", "reason_codes": ["legacy_text"]}
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
        output.append(
            {
                "title": title,
                "priority": str(item.get("priority", "P2")).strip() or "P2",
                "reason_codes": [
                    str(code) for code in reason_codes if str(code).strip()
                ],
            }
        )
    return output


def build_time_range(
    *, start_time: float | None, end_time: float | None, duration: float | None = None
) -> dict[str, float | None]:
    if duration is None and start_time is not None and end_time is not None:
        duration = max(0.0, float(end_time) - float(start_time))
    return {
        "start_time": None if start_time is None else round(float(start_time), 2),
        "end_time": None if end_time is None else round(float(end_time), 2),
        "duration": None if duration is None else round(float(duration), 2),
    }


def build_diagnosis_payload(
    *,
    source: str,
    label: str,
    confidence: float | None,
    severity: str | None,
    recommendations: list[dict[str, Any]] | list[str] | None,
    model: Any,
    time_range: dict[str, Any] | None,
    lead_time_s: float | None = None,
    evidence_summary: dict[str, Any] | None = None,
) -> dict[str, Any]:
    try:
        confidence_value = None if confidence is None else float(confidence)
    except (TypeError, ValueError):
        confidence_value = None

    severity_level = str(severity or "unknown").strip() or "unknown"
    if severity_level not in CANONICAL_SEVERITY_LEVELS:
        severity_level = "unknown"

    return {
        "source": str(source).strip() or "unknown",
        "label": str(label).strip() or "unknown",
        "confidence": confidence_value,
        "probability": confidence_value,
        "severity": severity_level,
        "recommendations": normalize_recommendations(recommendations),
        "model": model,
        "time_range": time_range,
        "lead_time_s": None if lead_time_s is None else round(float(lead_time_s), 2),
        "evidence_summary": evidence_summary,
    }
