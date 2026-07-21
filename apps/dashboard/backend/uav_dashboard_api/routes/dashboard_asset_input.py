from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

from flask import current_app

from uav_dashboard_api import db
from uav_dashboard_api.models.asset import Asset
from uav_dashboard_api.models.telemetry_session import TelemetrySession
from uav_dashboard_api.utils.path_store import resolve_local_path, serialize_local_path


CSV_QUALITY_SAMPLE_ROWS = 2000

class AssetNotPreparedError(RuntimeError):
    """ULG 资产尚未prepare。"""

class SessionCsvUnavailableError(RuntimeError):
    """遥测会话未提供可分析CSV。"""

def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fp:
        while True:
            chunk = fp.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()

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

def _format_percent(value: float) -> str:
    rounded = round(value, 1)
    if rounded.is_integer():
        return f"{int(rounded)}%"
    return f"{rounded}%"

def _build_csv_quality_meta(csv_path: str | Path | None) -> dict:
    if not csv_path:
        return {}
    path = Path(csv_path)
    if not path.exists() or not path.is_file():
        return {}

    try:
        with path.open("r", encoding="utf-8-sig", newline="") as fp:
            reader = csv.reader(fp)
            header = next(reader, None)
            if not header:
                return {}

            column_count = len(header)
            missing_cells = 0
            total_cells = 0
            sampled_rows = 0
            empty_markers = {"", "nan", "none", "null"}

            for row in reader:
                sampled_rows += 1
                values = list(row[:column_count])
                if len(values) < column_count:
                    values.extend([""] * (column_count - len(values)))
                total_cells += column_count
                missing_cells += sum(
                    1 for value in values if str(value).strip().lower() in empty_markers
                )
                if sampled_rows >= CSV_QUALITY_SAMPLE_ROWS:
                    break
    except Exception:
        return {}

    if column_count <= 0 or total_cells <= 0:
        return {
            "field_completeness": "暂无",
            "field_completeness_detail": {
                "column_count": column_count,
                "sampled_rows": sampled_rows,
                "missing_cells": missing_cells,
            },
        }

    completeness = max(0.0, min(100.0, (1 - missing_cells / total_cells) * 100))
    return {
        "field_completeness": _format_percent(completeness),
        "field_completeness_detail": {
            "column_count": column_count,
            "sampled_rows": sampled_rows,
            "missing_cells": missing_cells,
            "sample_limit": CSV_QUALITY_SAMPLE_ROWS,
        },
    }

def _coerce_float(value) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None

def _downsample_curve(curve: list[dict], max_points: int = 1000) -> list[dict]:
    if len(curve) <= max_points:
        return curve
    step = max(1, len(curve) // max_points)
    return curve[::step]

def _read_fault_prob_curve_json(path_value: str | Path | None) -> list[dict]:
    resolved = resolve_local_path(path_value, current_app.config)
    if not resolved:
        return []
    path = Path(resolved)
    if not path.exists() or not path.is_file():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return []
    if not isinstance(data, list):
        return []
    curve = []
    for item in data:
        if not isinstance(item, dict):
            continue
        time_s = _coerce_float(item.get("time_s"))
        fault_prob = _coerce_float(item.get("fault_prob"))
        if time_s is None or fault_prob is None:
            continue
        curve.append({"time_s": time_s, "fault_prob": fault_prob})
    return _downsample_curve(curve)

def _read_fault_prob_curve_csv(path_value: str | Path | None) -> list[dict]:
    resolved = resolve_local_path(path_value, current_app.config)
    if not resolved:
        return []
    path = Path(resolved)
    if not path.exists() or not path.is_file():
        return []
    curve = []
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as fp:
            reader = csv.DictReader(fp)
            if not reader.fieldnames or "fault_prob" not in reader.fieldnames:
                return []
            time_field = next(
                (
                    name
                    for name in ("time_s", "trueTime", "timestamp", "Timestamp")
                    if name in reader.fieldnames
                ),
                None,
            )
            for idx, row in enumerate(reader):
                fault_prob = _coerce_float(row.get("fault_prob"))
                if fault_prob is None:
                    continue
                time_s = _coerce_float(row.get(time_field)) if time_field else None
                if time_s is None:
                    time_s = float(idx)
                curve.append({"time_s": time_s, "fault_prob": fault_prob})
    except Exception:
        return []
    return _downsample_curve(curve)

def _resolve_asset_telemetry_session(asset: Asset):
    meta = asset.get_meta()
    session_id = str(meta.get("session_id") or "").strip()
    if session_id:
        session = db.session.get(TelemetrySession, session_id)
        if session is not None:
            return session

    sessions = (
        TelemetrySession.query.filter(TelemetrySession.has_detection.is_(True))
        .order_by(TelemetrySession.created_at.desc())
        .all()
    )
    for session in sessions:
        source_asset_id = session.get_meta().get("source_asset_id")
        if source_asset_id is not None and str(source_asset_id) == str(asset.id):
            return session

    candidates = [
        asset.stored_path,
        meta.get("prepared_csv_path"),
    ]
    for candidate in candidates:
        if not candidate:
            continue
        session = (
            TelemetrySession.query.filter(
                TelemetrySession.has_detection.is_(True),
                TelemetrySession.source_path == candidate,
            )
            .order_by(TelemetrySession.created_at.desc())
            .first()
        )
        if session is not None:
            return session
        session = (
            TelemetrySession.query.filter(
                TelemetrySession.has_detection.is_(True),
                TelemetrySession.csv_path == candidate,
            )
            .order_by(TelemetrySession.created_at.desc())
            .first()
        )
        if session is not None:
            return session

    resolved_candidates = set()
    for candidate in candidates:
        resolved = resolve_local_path(candidate, current_app.config)
        if resolved:
            try:
                resolved_candidates.add(str(Path(resolved).resolve()))
            except Exception:
                resolved_candidates.add(str(Path(resolved)))
    if not resolved_candidates:
        return None

    for session in sessions:
        session_candidates = [session.source_path, session.csv_path]
        for session_candidate in session_candidates:
            resolved = resolve_local_path(session_candidate, current_app.config)
            if not resolved:
                continue
            try:
                session_path = str(Path(resolved).resolve())
            except Exception:
                session_path = str(Path(resolved))
            if session_path in resolved_candidates:
                return session
    return None

def _asset_payload_with_quality(asset: Asset, *, fill_missing_quality: bool = True) -> dict:
    payload = asset.to_dict()
    meta = dict(payload.get("meta") or {})
    if fill_missing_quality and asset.kind == "csv" and not meta.get("field_completeness"):
        try:
            csv_path = _resolved_csv_path(asset, strict_prepare=False)
        except Exception:
            csv_path = ""
        quality_meta = _build_csv_quality_meta(csv_path)
        if quality_meta:
            meta.update(quality_meta)
            payload["meta"] = meta
    return payload

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
    meta = {
        "prepared_csv_path": str(csv_file),
        "source_type": "telemetry_session",
        "session_id": session.id,
    }
    meta.update(_build_csv_quality_meta(csv_file))
    asset.set_meta(meta)
    db.session.add(asset)
    db.session.flush()
    return asset
