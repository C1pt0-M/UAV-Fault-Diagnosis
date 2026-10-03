from __future__ import annotations

import math
from typing import Any


PX4_SHARED_RAW_MAP = {
    "gyro_x": ["gyro_x"],
    "gyro_y": ["gyro_y"],
    "gyro_z": ["gyro_z"],
    "accel_x": ["accel_x"],
    "accel_y": ["accel_y"],
    "accel_z": ["accel_z"],
    "baro_alt": ["baro_alt", "baro_alt_meter"],
    "baro_pressure": ["baro_pressure", "baro_pressure_pa"],
    "att_q0": ["att_q0", "q_w"],
    "att_q1": ["att_q1", "q_x"],
    "att_q2": ["att_q2", "q_y"],
    "att_q3": ["att_q3", "q_z"],
    "vel_x": ["vel_x", "vx"],
    "vel_y": ["vel_y", "vy"],
    "vel_z": ["vel_z", "vz"],
    "mag_x": ["mag_x"],
    "mag_y": ["mag_y"],
    "mag_z": ["mag_z"],
    "rpm_1": ["rpm_1", "esc_rpm_1"],
    "rpm_2": ["rpm_2", "esc_rpm_2"],
    "rpm_3": ["rpm_3", "esc_rpm_3"],
    "rpm_4": ["rpm_4", "esc_rpm_4"],
}


def _safe_float(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def _quat_from_euler(
    roll_rad: Any, pitch_rad: Any, yaw_rad: Any
) -> tuple[float, float, float, float] | None:
    """Quaternion from roll/pitch/yaw in RADIANS; no unit conversion (see the
    same function in uav_fault.telemetry.schema_adapter for why)."""
    roll = _safe_float(roll_rad)
    pitch = _safe_float(pitch_rad)
    yaw = _safe_float(yaw_rad)
    if roll is None or pitch is None or yaw is None:
        return None
    cr, sr = math.cos(roll / 2), math.sin(roll / 2)
    cp, sp = math.cos(pitch / 2), math.sin(pitch / 2)
    cy, sy = math.cos(yaw / 2), math.sin(yaw / 2)
    return (
        cr * cp * cy + sr * sp * sy,
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
    )


def adapt_px4_row_to_shared_raw(row: dict[str, Any]) -> dict[str, float]:
    out: dict[str, float] = {}
    out["time_s"] = float(_safe_float(row.get("time_s")) or 0.0)

    for target, aliases in PX4_SHARED_RAW_MAP.items():
        value = None
        for alias in aliases:
            value = _safe_float(row.get(alias))
            if value is not None:
                break
        out[target] = float(value) if value is not None else 0.0

    if all(out[name] == 0.0 for name in ("att_q0", "att_q1", "att_q2", "att_q3")):
        quat = _quat_from_euler(row.get("roll"), row.get("pitch"), row.get("yaw"))
        if quat is not None:
            out["att_q0"], out["att_q1"], out["att_q2"], out["att_q3"] = map(
                float, quat
            )

    return out
