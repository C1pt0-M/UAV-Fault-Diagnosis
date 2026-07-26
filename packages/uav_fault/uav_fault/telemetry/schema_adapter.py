from __future__ import annotations

import math
from typing import Any

import numpy as np

from uav_fault.data.build_sequence_cache import SEQ_COLUMN_ALIASES, SEQ_COLUMNS

CANONICAL_SCHEMA_VERSION = "rfly_seq_v1"
ACTUATOR_COLUMNS = [
    "actuator_ctrl_1", "actuator_ctrl_2", "actuator_ctrl_3", "actuator_ctrl_4",
    "actuator_out_1", "actuator_out_2", "actuator_out_3", "actuator_out_4",
]
ACTUATOR_OUTPUT_COLUMNS = (
    "actuator_out_1",
    "actuator_out_2",
    "actuator_out_3",
    "actuator_out_4",
)
BASE_COLUMNS = [name for name, _ in SEQ_COLUMN_ALIASES]
DERIVED_COLUMNS = [name for name in SEQ_COLUMNS if name not in BASE_COLUMNS]
PX4_MAP = {
    "actuator_ctrl_1": "actuator_ctrl_1", "actuator_ctrl_2": "actuator_ctrl_2", "actuator_ctrl_3": "actuator_ctrl_3", "actuator_ctrl_4": "actuator_ctrl_4",
    "gyro_x": "gyro_x", "gyro_y": "gyro_y", "gyro_z": "gyro_z",
    "accel_x": "accel_x", "accel_y": "accel_y", "accel_z": "accel_z",
    "baro_alt": "baro_alt_meter", "baro_temp": "baro_temp", "baro_pressure": "baro_pressure_pa",
    "vel_x": "vx", "vel_y": "vy", "vel_z": "vz",
    "mag_x": "mag_x", "mag_y": "mag_y", "mag_z": "mag_z",
    "att_q_w": "q_w", "att_q_x": "q_x", "att_q_y": "q_y", "att_q_z": "q_z",
}
LIVE_CANONICAL_ALIASES = {
    "actuator_out_1": ("motor1",),
    "actuator_out_2": ("motor2",),
    "actuator_out_3": ("motor3",),
    "actuator_out_4": ("motor4",),
    "baro_alt": ("baro_alt_meter",),
    "baro_pressure": ("baro_pressure_pa",),
    "vel_x": ("vx",),
    "vel_y": ("vy",),
    "vel_z": ("vz",),
    "att_q_w": ("q_w",),
    "att_q_x": ("q_x",),
    "att_q_y": ("q_y",),
    "att_q_z": ("q_z",),
}


def _safe_float(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if np.isfinite(out) else None


def _window_has_signal(records: list[dict[str, Any]], columns: tuple[str, ...], *, eps: float) -> bool:
    for row in records:
        for column in columns:
            value = _safe_float(row.get(column))
            if value is not None and abs(value) > eps:
                return True
    return False


def _window_mean_abs(records: list[dict[str, Any]], column: str) -> float:
    values = [
        abs(value)
        for row in records
        for value in [_safe_float(row.get(column))]
        if value is not None
    ]
    return float(sum(values) / len(values)) if values else 0.0


def _window_mean(records: list[dict[str, Any]], columns: tuple[str, ...]) -> float:
    values = [
        value
        for row in records
        for column in columns
        for value in [_safe_float(row.get(column))]
        if value is not None
    ]
    return float(sum(values) / len(values)) if values else 0.0


def assess_canonical_input_contract(
    records: list[dict[str, Any]] | tuple[dict[str, Any], ...],
    *,
    coverage: dict[str, list[str]] | None = None,
    actuator_source: Any = None,
    window_size: int = 150,
) -> dict[str, Any]:
    """Assess whether canonical telemetry is trustworthy enough for online inference."""
    coverage = coverage or {}
    recent_window = list(records)[-int(window_size) :] if records else []
    default_count = len(coverage.get("default", []))
    proxy_count = len(coverage.get("proxy", []))
    if actuator_source is None and recent_window:
        actuator_source = recent_window[-1].get("actuator_source")

    zero_signal_groups: list[str] = []
    if not recent_window:
        zero_signal_groups.append("empty_window")
    else:
        if not _window_has_signal(recent_window, ACTUATOR_OUTPUT_COLUMNS, eps=1.0):
            zero_signal_groups.append("actuator_outputs")

        if _window_mean_abs(recent_window, "accel_z") < 1.0:
            zero_signal_groups.append("imu_accel_z")

        if (
            not _window_has_signal(recent_window, ("vel_x", "vel_y", "vel_z"), eps=1e-3)
            and not _window_has_signal(recent_window, ("baro_alt",), eps=1e-3)
        ):
            zero_signal_groups.append("motion_reference")

        if (
            _window_mean(recent_window, ACTUATOR_OUTPUT_COLUMNS) < 1050.0
            and _window_mean_abs(recent_window, "baro_alt") < 0.5
        ):
            zero_signal_groups.append("preflight_idle")

    detection_blocked = bool(
        {
            "empty_window",
            "actuator_outputs",
            "imu_accel_z",
            "motion_reference",
            "preflight_idle",
        }.intersection(zero_signal_groups)
    )
    if detection_blocked:
        contract_status = "degraded"
        reason = "untrusted_zero_signals"
    elif zero_signal_groups:
        contract_status = "degraded"
        reason = "weak_motion_reference"
    elif default_count > 0:
        contract_status = "degraded"
        reason = "missing_features"
    elif actuator_source == "proxy" or proxy_count > 0:
        contract_status = "degraded"
        reason = "proxy_actuator_source"
    else:
        contract_status = "healthy"
        reason = "ok"

    return {
        "status": contract_status,
        "reason": reason,
        "actuator_source": actuator_source,
        "default_count": default_count,
        "proxy_count": proxy_count,
        "zero_signal_groups": zero_signal_groups,
        "detection_blocked": detection_blocked,
    }


def _quat_from_euler(roll_deg: Any, pitch_deg: Any, yaw_deg: Any) -> tuple[float, float, float, float] | None:
    roll = _safe_float(roll_deg)
    pitch = _safe_float(pitch_deg)
    yaw = _safe_float(yaw_deg)
    if roll is None or pitch is None or yaw is None:
        return None
    roll, pitch, yaw = map(math.radians, (roll, pitch, yaw))
    cr, sr = math.cos(roll / 2), math.sin(roll / 2)
    cp, sp = math.cos(pitch / 2), math.sin(pitch / 2)
    cy, sy = math.cos(yaw / 2), math.sin(yaw / 2)
    return (
        cr * cp * cy + sr * sp * sy,
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
    )


def _euler_from_quat(
    q_w: Any, q_x: Any, q_y: Any, q_z: Any
) -> tuple[float, float, float] | None:
    quat = [_safe_float(value) for value in (q_w, q_x, q_y, q_z)]
    if any(value is None for value in quat):
        return None
    w, x, y, z = quat
    norm = math.sqrt(w * w + x * x + y * y + z * z)
    if norm <= 1e-8:
        return None
    w, x, y, z = (value / norm for value in (w, x, y, z))

    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = math.atan2(sinr_cosp, cosr_cosp)

    sinp = 2.0 * (w * y - z * x)
    pitch = math.asin(max(-1.0, min(1.0, sinp)))

    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = math.atan2(siny_cosp, cosy_cosp)
    return (roll, pitch, yaw)


def normalize_time_columns(df):
    if "time_s" in df.columns:
        return df
    df = df.copy()
    if "trueTime" in df.columns:
        df["time_s"] = np.asarray(df["trueTime"], dtype=float)
    elif "Timestamp" in df.columns:
        df["time_s"] = np.asarray(df["Timestamp"], dtype=float) * 1e-6
    return df


def _apply_derived(out: dict[str, Any], source_map: dict[str, str]) -> tuple[dict[str, Any], dict[str, list[str]]]:
    values = [float(_safe_float(out.get(name)) or 0.0) for name in ACTUATOR_COLUMNS]
    mean = float(np.mean(values)) if values else 0.0
    std = float(np.std(values)) if values else 0.0
    ax = float(_safe_float(out.get("accel_x")) or 0.0)
    ay = float(_safe_float(out.get("accel_y")) or 0.0)
    az = float(_safe_float(out.get("accel_z")) or 0.0)
    acc_norm = float(np.sqrt(ax * ax + ay * ay + az * az))
    out.update({
        "actuator_mean": mean,
        "actuator_std": std,
        "acc_norm": acc_norm,
        "actuator_effort": float(mean / max(acc_norm, 0.1)),
    })
    for column in SEQ_COLUMNS:
        value = _safe_float(out.get(column))
        out[column] = float(value) if value is not None else 0.0
    coverage = {"mapped": [], "derived": [], "default": []}
    for column in SEQ_COLUMNS:
        tag = source_map.get(column, "derived" if column in DERIVED_COLUMNS else "default")
        coverage.setdefault(tag, []).append(column)
    for column, tag in source_map.items():
        if column not in SEQ_COLUMNS:
            coverage.setdefault(tag, []).append(column)
    return out, coverage


def canonicalize_rfly_frame(row: dict[str, Any]) -> tuple[dict[str, Any], dict[str, list[str]]]:
    out = dict(row)
    source_map: dict[str, str] = {}
    actuator_source = str(row.get("actuator_source") or "").strip()
    if actuator_source:
        out["actuator_source"] = actuator_source
        source_map["actuator_source"] = "mapped"
    if _safe_float(out.get("time_s")) is not None:
        source_map["time_s"] = "mapped"
    else:
        true_time = _safe_float(row.get("trueTime"))
        timestamp_us = _safe_float(row.get("Timestamp"))
        if true_time is not None:
            out["time_s"] = true_time
            source_map["time_s"] = "mapped"
        elif timestamp_us is not None:
            out["time_s"] = timestamp_us * 1e-6
            source_map["time_s"] = "mapped"
    if _safe_float(out.get("seq")) is not None:
        source_map["seq"] = "mapped"
    else:
        seq = _safe_float(row.get("Index"))
        if seq is not None:
            out["seq"] = seq
            source_map["seq"] = "mapped"
    for name, aliases in SEQ_COLUMN_ALIASES:
        if _safe_float(out.get(name)) is not None:
            source_map[name] = "mapped"
            continue
        for alias in (*LIVE_CANONICAL_ALIASES.get(name, ()), *aliases):
            if _safe_float(row.get(alias)) is not None:
                out[name] = row.get(alias)
                source_map[name] = "mapped"
                break
    if actuator_source == "proxy":
        for column in ("actuator_out_1", "actuator_out_2", "actuator_out_3", "actuator_out_4"):
            if _safe_float(out.get(column)) is not None:
                source_map[column] = "proxy"
    elif actuator_source == "px4_controls":
        for column in ("actuator_out_1", "actuator_out_2", "actuator_out_3", "actuator_out_4"):
            if _safe_float(out.get(column)) is not None:
                source_map[column] = "derived"
    if not all(key in out for key in ("att_q_w", "att_q_x", "att_q_y", "att_q_z")):
        quat = _quat_from_euler(row.get("roll"), row.get("pitch"), row.get("yaw"))
        if quat is not None:
            out.update(dict(zip(("att_q_w", "att_q_x", "att_q_y", "att_q_z"), quat)))
            for key in ("att_q_w", "att_q_x", "att_q_y", "att_q_z"):
                source_map.setdefault(key, "derived")
    if any(_safe_float(out.get(key)) is None for key in ("roll", "pitch", "yaw")):
        euler = _euler_from_quat(
            out.get("att_q_w"),
            out.get("att_q_x"),
            out.get("att_q_y"),
            out.get("att_q_z"),
        )
        if euler is not None:
            for key, value in zip(("roll", "pitch", "yaw"), euler):
                if _safe_float(out.get(key)) is None:
                    out[key] = value
                    source_map.setdefault(key, "derived")
    return _apply_derived(out, source_map)


def canonicalize_rfly_records(df) -> list[dict[str, Any]]:
    rows = df.to_dict(orient="records") if hasattr(df, "to_dict") else list(df)
    return [canonicalize_rfly_frame(row)[0] for row in rows]


def adapt_px4_live_frame(frame: dict[str, Any]) -> tuple[dict[str, Any], dict[str, list[str]]]:
    out = {"time_s": frame.get("time_s"), "seq": frame.get("seq")}
    source_map: dict[str, str] = {}
    if _safe_float(frame.get("time_s")) is not None:
        source_map["time_s"] = "mapped"
    if _safe_float(frame.get("seq")) is not None:
        source_map["seq"] = "mapped"
    for idx in range(1, 5):
        canonical = f"actuator_out_{idx}"
        value = _safe_float(frame.get(canonical))
        if value is not None:
            out[canonical] = value
            source_map[canonical] = "mapped"
            continue
        value = _safe_float(frame.get(f"motor{idx}"))
        if value is not None:
            out[canonical] = value
            source_map[canonical] = "mapped"
    for canonical, live_name in PX4_MAP.items():
        value = _safe_float(frame.get(live_name))
        if value is not None:
            out[canonical] = value
            source_map[canonical] = "mapped"
    quat = _quat_from_euler(frame.get("roll"), frame.get("pitch"), frame.get("yaw"))
    if quat is not None:
        for key, value in zip(("att_q_w", "att_q_x", "att_q_y", "att_q_z"), quat):
            if key in source_map:
                continue
            out[key] = value
            source_map[key] = "derived"
    return _apply_derived(out, source_map)
