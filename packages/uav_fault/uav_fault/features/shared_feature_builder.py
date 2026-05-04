from __future__ import annotations

import math
from typing import Any


def _safe_float(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def _vector_norm(values: list[float | None]) -> float:
    safe = [float(v or 0.0) for v in values]
    return math.sqrt(sum(v * v for v in safe))


def _quat_to_tilt_angle(q0: float, q1: float, q2: float, q3: float) -> float:
    sinp = 2.0 * (q0 * q2 - q3 * q1)
    sinp = max(-1.0, min(1.0, sinp))
    pitch = math.asin(sinp)
    sinr_cosp = 2.0 * (q0 * q1 + q2 * q3)
    cosr_cosp = 1.0 - 2.0 * (q1 * q1 + q2 * q2)
    roll = math.atan2(sinr_cosp, cosr_cosp)
    return math.sqrt(roll * roll + pitch * pitch)


def build_point_features(frame: dict[str, Any]) -> dict[str, float]:
    gyro = [_safe_float(frame.get(name)) for name in ("gyro_x", "gyro_y", "gyro_z")]
    accel = [_safe_float(frame.get(name)) for name in ("accel_x", "accel_y", "accel_z")]
    vel = [_safe_float(frame.get(name)) for name in ("vel_x", "vel_y", "vel_z")]
    q = [
        _safe_float(frame.get(name))
        for name in ("att_q0", "att_q1", "att_q2", "att_q3")
    ]

    q0, q1, q2, q3 = [float(v or 0.0) for v in q]
    return {
        "gyro_norm": _vector_norm(gyro),
        "accel_norm": _vector_norm(accel),
        "velocity_norm": _vector_norm(vel),
        "tilt_angle": _quat_to_tilt_angle(q0, q1, q2, q3),
    }


def _std(values: list[float]) -> float:
    if not values:
        return 0.0
    mean = sum(values) / len(values)
    return math.sqrt(sum((v - mean) ** 2 for v in values) / len(values))


def build_sequence_features(
    frames: list[dict[str, Any]], *, window_size: int = 10
) -> list[dict[str, float]]:
    enriched: list[dict[str, float]] = []
    for idx, frame in enumerate(frames):
        current = {k: float(v) for k, v in build_point_features(frame).items()}

        now_t = _safe_float(frame.get("time_s"))
        if idx == 0:
            current["baro_alt_rate"] = 0.0
            current["baro_pressure_rate"] = 0.0
        else:
            prev = frames[idx - 1]
            prev_t = _safe_float(prev.get("time_s"))
            dt = (now_t - prev_t) if now_t is not None and prev_t is not None else None
            if dt is None or dt <= 0:
                current["baro_alt_rate"] = 0.0
                current["baro_pressure_rate"] = 0.0
            else:
                alt_now = float(_safe_float(frame.get("baro_alt")) or 0.0)
                alt_prev = float(_safe_float(prev.get("baro_alt")) or 0.0)
                p_now = float(_safe_float(frame.get("baro_pressure")) or 0.0)
                p_prev = float(_safe_float(prev.get("baro_pressure")) or 0.0)
                current["baro_alt_rate"] = (alt_now - alt_prev) / dt
                current["baro_pressure_rate"] = (p_now - p_prev) / dt

        start = max(0, idx - window_size + 1)
        window = enriched[start:idx] + [current]
        for name in ("gyro_norm", "accel_norm", "velocity_norm", "tilt_angle"):
            values = [item[name] for item in window]
            suffix = "tilt_std_w" if name == "tilt_angle" else f"{name}_std_w"
            current[suffix] = _std(values)

        enriched.append(current)

    return enriched
