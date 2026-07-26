from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


ACTUATORS = [f"actuator_out_{idx}" for idx in range(1, 5)]
REQUIRED = [
    "time_s",
    *ACTUATORS,
    "accel_x",
    "accel_y",
    "accel_z",
    "baro_pressure_pa",
    "baro_alt_meter",
    "baro_temp",
    "pos_z",
    "mag_x",
    "mag_y",
    "mag_z",
    "actuator_ctrl_1",
    "actuator_ctrl_2",
    "actuator_ctrl_3",
    "actuator_ctrl_4",
    "pymav_highres_imu_seen",
    "pymav_actuator_control_seen",
    "armed",
]


def _number(row: dict[str, str], key: str) -> float | None:
    try:
        return float(row.get(key, ""))
    except (TypeError, ValueError):
        return None


def validate(
    path: Path,
    *,
    min_rows: int = 3000,
    require_horizontal_motion: bool = False,
    target_altitude_m: float = 0.0,
) -> dict:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        columns = list(reader.fieldnames or [])
        rows = list(reader)

    missing = [column for column in REQUIRED if column not in columns]
    times = [value for row in rows if (value := _number(row, "time_s")) is not None]
    positive_dt = [b - a for a, b in zip(times, times[1:]) if b > a]
    sample_hz = (
        (len(times) - 1) / (times[-1] - times[0])
        if len(times) >= 2 and times[-1] > times[0]
        else 0.0
    )

    pressure = [
        value
        for row in rows
        if (value := _number(row, "baro_pressure_pa")) is not None
    ]
    pressure_valid_ratio = (
        sum(80_000.0 <= value <= 110_000.0 for value in pressure) / len(pressure)
        if pressure
        else 0.0
    )

    actuator_values = [
        value
        for row in rows
        for column in ACTUATORS
        if (value := _number(row, column)) is not None and value > 0.0
    ]
    actuator_valid_ratio = (
        sum(900.0 <= value <= 1900.0 for value in actuator_values)
        / len(actuator_values)
        if actuator_values
        else 0.0
    )
    actuator_range = (
        max(actuator_values) - min(actuator_values) if actuator_values else 0.0
    )

    accel_z = [
        value for row in rows if (value := _number(row, "accel_z")) is not None
    ]
    accel_valid_ratio = (
        sum(5.0 <= abs(value) <= 15.0 for value in accel_z) / len(accel_z)
        if accel_z
        else 0.0
    )
    pos_z = [value for row in rows if (value := _number(row, "pos_z")) is not None]
    altitude_range_m = max(pos_z) - min(pos_z) if pos_z else 0.0
    max_altitude_m = max((-value for value in pos_z), default=0.0)
    target_altitude_reached = (
        target_altitude_m <= 0.0
        or 0.9 * target_altitude_m <= max_altitude_m <= 1.25 * target_altitude_m
    )
    horizontal_x = 0.0
    horizontal_y = 0.0
    horizontal_x_values = [0.0]
    horizontal_y_values = [0.0]
    horizontal_speed_max = 0.0
    for previous, current in zip(rows, rows[1:]):
        t0 = _number(previous, "time_s")
        t1 = _number(current, "time_s")
        vx = _number(current, "vx")
        vy = _number(current, "vy")
        if vx is None or vy is None:
            continue
        horizontal_speed_max = max(horizontal_speed_max, (vx * vx + vy * vy) ** 0.5)
        if t0 is None or t1 is None or not 0.0 < t1 - t0 < 0.1:
            continue
        horizontal_x += vx * (t1 - t0)
        horizontal_y += vy * (t1 - t0)
        horizontal_x_values.append(horizontal_x)
        horizontal_y_values.append(horizontal_y)
    horizontal_x_range_m = max(horizontal_x_values) - min(horizontal_x_values)
    horizontal_y_range_m = max(horizontal_y_values) - min(horizontal_y_values)
    horizontal_motion_present = (
        max(horizontal_x_range_m, horizontal_y_range_m) >= 5.0
        and horizontal_speed_max >= 0.5
    )
    baro_alt = [
        value for row in rows if (value := _number(row, "baro_alt_meter")) is not None
    ]
    baro_altitude_range_m = max(baro_alt) - min(baro_alt) if baro_alt else 0.0
    mag_norm = []
    for row in rows:
        values = [_number(row, key) for key in ("mag_x", "mag_y", "mag_z")]
        if all(value is not None for value in values):
            mag_norm.append(sum(float(value) ** 2 for value in values) ** 0.5)
    actuator_controls = [
        value
        for row in rows
        for key in ("actuator_ctrl_1", "actuator_ctrl_2", "actuator_ctrl_3", "actuator_ctrl_4")
        if (value := _number(row, key)) is not None
    ]
    actuator_control_range = (
        max(actuator_controls) - min(actuator_controls) if actuator_controls else 0.0
    )
    highres_values = [
        _number(row, "pymav_highres_imu_seen") or 0.0 for row in rows
    ]
    actuator_control_values = [
        _number(row, "pymav_actuator_control_seen") or 0.0 for row in rows
    ]
    highres_imu_seen_ratio = (
        sum(value > 0.5 for value in highres_values) / len(highres_values)
        if highres_values else 0.0
    )
    actuator_control_seen_ratio = (
        sum(value > 0.5 for value in actuator_control_values)
        / len(actuator_control_values)
        if actuator_control_values else 0.0
    )
    armed_values = [_number(row, "armed") or 0.0 for row in rows]
    armed_indices = [idx for idx, value in enumerate(armed_values) if value > 0.5]
    armed_phase_present = bool(armed_indices)
    prearm_duration_s = 0.0
    armed_duration_s = 0.0
    postarm_duration_s = 0.0
    if armed_indices and times:
        first_armed = armed_indices[0]
        last_armed = armed_indices[-1]
        prearm_duration_s = max(0.0, times[first_armed] - times[0])
        armed_duration_s = max(0.0, times[last_armed] - times[first_armed])
        postarm_duration_s = max(0.0, times[-1] - times[last_armed])

    checks = {
        "required_columns": not missing,
        "minimum_rows": len(rows) >= min_rows,
        "sample_rate_48_52_hz": 48.0 <= sample_hz <= 52.0,
        "pressure_valid": pressure_valid_ratio >= 0.99,
        "actuator_pwm_valid": actuator_valid_ratio >= 0.99,
        "actuator_dynamic": actuator_range >= 10.0,
        "imu_accel_z_valid": accel_valid_ratio >= 0.95,
        "flight_motion_present": altitude_range_m >= 0.5,
        "highres_imu_stream_seen": highres_imu_seen_ratio >= 0.99,
        "actuator_control_stream_seen": actuator_control_seen_ratio >= 0.99,
        "baro_altitude_dynamic": baro_altitude_range_m >= 0.5,
        "magnetometer_signal_present": bool(mag_norm) and max(mag_norm) >= 0.01,
        "actuator_control_dynamic": actuator_control_range >= 0.01,
        "horizontal_motion_present": (
            horizontal_motion_present if require_horizontal_motion else True
        ),
        "target_altitude_reached": target_altitude_reached,
        "armed_phase_present": armed_phase_present,
        "prearm_duration_reasonable": armed_phase_present and prearm_duration_s <= 30.0,
    }
    return {
        "status": "pass" if all(checks.values()) else "fail",
        "file": str(path.resolve()),
        "rows": len(rows),
        "sample_hz": round(sample_hz, 3),
        "duration_s": round(times[-1] - times[0], 3) if len(times) >= 2 else 0.0,
        "pressure_valid_ratio": round(pressure_valid_ratio, 6),
        "actuator_valid_ratio": round(actuator_valid_ratio, 6),
        "actuator_range": round(actuator_range, 3),
        "accel_z_valid_ratio": round(accel_valid_ratio, 6),
        "altitude_range_m": round(altitude_range_m, 3),
        "target_altitude_m": round(target_altitude_m, 3),
        "max_altitude_m": round(max_altitude_m, 3),
        "baro_altitude_range_m": round(baro_altitude_range_m, 3),
        "mag_norm_max": round(max(mag_norm), 6) if mag_norm else 0.0,
        "actuator_control_range": round(actuator_control_range, 6),
        "highres_imu_seen": bool(highres_imu_seen_ratio >= 0.99),
        "actuator_control_seen": bool(actuator_control_seen_ratio >= 0.99),
        "highres_imu_seen_ratio": round(highres_imu_seen_ratio, 6),
        "actuator_control_seen_ratio": round(actuator_control_seen_ratio, 6),
        "horizontal_motion_required": require_horizontal_motion,
        "horizontal_speed_max_m_s": round(horizontal_speed_max, 3),
        "horizontal_x_range_m": round(horizontal_x_range_m, 3),
        "horizontal_y_range_m": round(horizontal_y_range_m, 3),
        "prearm_duration_s": round(prearm_duration_s, 3),
        "armed_duration_s": round(armed_duration_s, 3),
        "postarm_duration_s": round(postarm_duration_s, 3),
        "missing_columns": missing,
        "checks": checks,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate a no-fault RflySim CSV")
    parser.add_argument("--csv", required=True)
    parser.add_argument("--session-name", required=True)
    parser.add_argument("--maneuver", required=True)
    parser.add_argument("--target-altitude-m", type=float, default=0.0)
    parser.add_argument("--target-speed-m-s", type=float, default=0.0)
    parser.add_argument("--min-rows", type=int, default=3000)
    args = parser.parse_args()

    csv_path = Path(args.csv)
    report = validate(
        csv_path,
        min_rows=args.min_rows,
        require_horizontal_motion=args.target_speed_m_s > 0.0,
        target_altitude_m=args.target_altitude_m,
    )
    quality_path = csv_path.with_suffix(".quality.json")
    quality_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    metadata = {
        "session_name": args.session_name,
        "label": "normal",
        "platform": "RflySim",
        "airframe": "iris",
        "map": "LowGPU",
        "udp_mode": 2,
        "maneuver": args.maneuver,
        "target_altitude_m": args.target_altitude_m,
        "target_speed_m_s": args.target_speed_m_s,
        "schema_version": "rfly_seq_v1",
        "quality_status": report["status"],
        "rows": report["rows"],
        "sample_hz": report["sample_hz"],
    }
    csv_path.with_suffix(".meta.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["status"] == "pass" else 2


if __name__ == "__main__":
    raise SystemExit(main())
