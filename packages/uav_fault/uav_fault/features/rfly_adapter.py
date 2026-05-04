from __future__ import annotations

from typing import Any


RFLY_SHARED_RAW_MAP = {
    "gyro_x": ["_sensor_combined_0_gyro_rad[0]"],
    "gyro_y": ["_sensor_combined_0_gyro_rad[1]"],
    "gyro_z": ["_sensor_combined_0_gyro_rad[2]"],
    "accel_x": ["_sensor_combined_0_accelerometer_m_s2[0]"],
    "accel_y": ["_sensor_combined_0_accelerometer_m_s2[1]"],
    "accel_z": ["_sensor_combined_0_accelerometer_m_s2[2]"],
    "baro_alt": ["_vehicle_air_data_0_baro_alt_meter"],
    "baro_pressure": ["_vehicle_air_data_0_baro_pressure_pa"],
    "att_q0": ["_vehicle_attitude_0_q[0]"],
    "att_q1": ["_vehicle_attitude_0_q[1]"],
    "att_q2": ["_vehicle_attitude_0_q[2]"],
    "att_q3": ["_vehicle_attitude_0_q[3]"],
    "vel_x": ["_vehicle_local_position_0_vx"],
    "vel_y": ["_vehicle_local_position_0_vy"],
    "vel_z": ["_vehicle_local_position_0_vz"],
    "mag_x": ["_vehicle_magnetometer_0_magnetometer_ga[0]"],
    "mag_y": ["_vehicle_magnetometer_0_magnetometer_ga[1]"],
    "mag_z": ["_vehicle_magnetometer_0_magnetometer_ga[2]"],
    "rpm_1": ["TrueState_data_motorRPMs[1]"],
    "rpm_2": ["TrueState_data_motorRPMs[2]"],
    "rpm_3": ["TrueState_data_motorRPMs[3]"],
    "rpm_4": ["TrueState_data_motorRPMs[4]"],
}


def _safe_float(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out


def adapt_rfly_row_to_shared_raw(row: dict[str, Any]) -> dict[str, float]:
    out: dict[str, float] = {}

    time_s = _safe_float(row.get("trueTime"))
    if time_s is None:
        ts = _safe_float(row.get("Timestamp"))
        time_s = ts * 1e-6 if ts is not None else 0.0
    out["time_s"] = float(time_s)

    for target, aliases in RFLY_SHARED_RAW_MAP.items():
        value = None
        for alias in aliases:
            value = _safe_float(row.get(alias))
            if value is not None:
                break
        out[target] = float(value) if value is not None else 0.0

    return out
