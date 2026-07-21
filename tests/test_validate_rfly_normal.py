from __future__ import annotations

import csv
import importlib.util
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = PROJECT_ROOT / "scripts" / "data" / "validate_rfly_normal.py"
SPEC = importlib.util.spec_from_file_location("validate_rfly_normal", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
validator = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(validator)


def _write_session(
    path: Path,
    *,
    pressure: float = 101325.0,
    supplement_seen: bool = True,
    warmup_rows: int = 0,
    horizontal_motion: bool = True,
) -> None:
    columns = [
        "time_s",
        "actuator_out_1",
        "actuator_out_2",
        "actuator_out_3",
        "actuator_out_4",
        "accel_x",
        "accel_y",
        "accel_z",
        "baro_pressure_pa",
        "baro_alt_meter",
        "baro_temp",
        "pos_z",
        "vx",
        "vy",
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
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for idx in range(3000):
            delta = 20 if (idx // 100) % 2 else 0
            row_supplement_seen = supplement_seen and idx >= warmup_rows
            writer.writerow(
                {
                    "time_s": idx * 0.02,
                    "actuator_out_1": 1300,
                    "actuator_out_2": 1310,
                    "actuator_out_3": 1290,
                    "actuator_out_4": 1305 + delta,
                    "accel_x": 0.0,
                    "accel_y": 0.0,
                    "accel_z": -9.8,
                    "baro_pressure_pa": pressure,
                    "baro_alt_meter": 5.0 * idx / 2999,
                    "baro_temp": 25.0,
                    "pos_z": -5.0 * idx / 2999,
                    "vx": (1.0 if idx < 1500 else -1.0) if horizontal_motion else 0.0,
                    "vy": 0.5 if horizontal_motion else 0.0,
                    "mag_x": 0.25 if row_supplement_seen else 0.0,
                    "mag_y": -0.02 if row_supplement_seen else 0.0,
                    "mag_z": 0.43 if row_supplement_seen else 0.0,
                    "actuator_ctrl_1": 0.01 if row_supplement_seen else 0.0,
                    "actuator_ctrl_2": -0.01 if row_supplement_seen else 0.0,
                    "actuator_ctrl_3": 0.0,
                    "actuator_ctrl_4": (0.3 + delta / 100.0) if row_supplement_seen else 0.0,
                    "pymav_highres_imu_seen": int(row_supplement_seen),
                    "pymav_actuator_control_seen": int(row_supplement_seen),
                    "armed": int(500 <= idx <= 2500),
                }
            )


def test_validate_rfly_normal_accepts_complete_50hz_flight(tmp_path):
    path = tmp_path / "normal.csv"
    _write_session(path)

    report = validator.validate(path)

    assert report["status"] == "pass"
    assert report["sample_hz"] == 50.0
    assert report["checks"]["actuator_pwm_valid"] is True
    assert report["checks"]["pressure_valid"] is True


def test_validate_rfly_normal_rejects_pressure_unit_error(tmp_path):
    path = tmp_path / "bad-pressure.csv"
    _write_session(path, pressure=10_132_500.0)

    report = validator.validate(path)

    assert report["status"] == "fail"
    assert report["checks"]["pressure_valid"] is False


def test_validate_rfly_normal_rejects_missing_pymav_supplement(tmp_path):
    path = tmp_path / "missing-supplement.csv"
    _write_session(path, supplement_seen=False)

    report = validator.validate(path)

    assert report["status"] == "fail"
    assert report["checks"]["highres_imu_stream_seen"] is False
    assert report["checks"]["actuator_control_stream_seen"] is False
    assert report["checks"]["magnetometer_signal_present"] is False
    assert report["checks"]["actuator_control_dynamic"] is False


def test_validate_rfly_normal_rejects_pymav_warmup_rows(tmp_path):
    path = tmp_path / "warmup.csv"
    _write_session(path, warmup_rows=100)

    report = validator.validate(path)

    assert report["status"] == "fail"
    assert report["highres_imu_seen_ratio"] < 0.99
    assert report["actuator_control_seen_ratio"] < 0.99


def test_validate_rfly_normal_checks_horizontal_motion_when_required(tmp_path):
    path = tmp_path / "hover-only.csv"
    _write_session(path, horizontal_motion=False)

    report = validator.validate(path, require_horizontal_motion=True)

    assert report["status"] == "fail"
    assert report["checks"]["horizontal_motion_present"] is False


def test_validate_rfly_normal_checks_target_altitude(tmp_path):
    path = tmp_path / "too-low.csv"
    _write_session(path)

    report = validator.validate(path, target_altitude_m=10.0)

    assert report["status"] == "fail"
    assert report["checks"]["target_altitude_reached"] is False


def test_validate_rfly_normal_rejects_excessive_prearm_wait(tmp_path):
    path = tmp_path / "long-prearm.csv"
    _write_session(path)
    rows = list(csv.DictReader(path.open(newline="", encoding="utf-8")))
    for idx, row in enumerate(rows):
        row["armed"] = int(2000 <= idx <= 2500)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    report = validator.validate(path)

    assert report["status"] == "fail"
    assert report["prearm_duration_s"] > 30.0
    assert report["checks"]["prearm_duration_reasonable"] is False
