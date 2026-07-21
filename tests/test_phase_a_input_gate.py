import csv
import json
from pathlib import Path

from scripts.data import phase_a_input_gate


def _full_px4_row(time_s: float, seq: int, *, gap_scale: float = 1.0) -> dict:
    return {
        "time_s": time_s,
        "seq": seq,
        "motor1": 0.10 + 0.001 * seq,
        "motor2": 0.20 + 0.001 * seq,
        "motor3": 0.30 + 0.001 * seq,
        "motor4": 0.40 + 0.001 * seq,
        "actuator_ctrl_1": 0.01 * gap_scale,
        "actuator_ctrl_2": 0.02 * gap_scale,
        "actuator_ctrl_3": 0.03 * gap_scale,
        "actuator_ctrl_4": 0.04 * gap_scale,
        "gyro_x": 0.1 * seq,
        "gyro_y": 0.2 * seq,
        "gyro_z": 0.3 * seq,
        "accel_x": 1.0,
        "accel_y": 2.0,
        "accel_z": 9.8,
        "baro_alt_meter": 10.0 + 0.01 * seq,
        "baro_temp": 25.0,
        "baro_pressure_pa": 101325.0,
        "vel_x": 0.5,
        "vel_y": -0.1,
        "vel_z": -0.2,
        "mag_x": 0.01,
        "mag_y": 0.02,
        "mag_z": 0.03,
        "q_w": 1.0,
        "q_x": 0.0,
        "q_y": 0.0,
        "q_z": 0.0,
    }


def _write_csv(csv_path: Path, rows: list[dict]) -> None:
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def test_phase_a_input_gate_writes_expected_outputs(tmp_path):
    csv_path = tmp_path / "px4_live.csv"
    out_dir = tmp_path / "out"
    rows = [_full_px4_row(i * 0.02, i) for i in range(160)]
    _write_csv(csv_path, rows)

    summary = phase_a_input_gate.run_phase_a_input_gate(
        csv_path=csv_path,
        output_dir=out_dir,
        window_size=150,
        max_default_ratio=0.30,
        max_gap_s=0.20,
    )

    assert summary["rows"] == 160
    assert abs(summary["dt_median"] - 0.02) < 1e-9
    assert summary["valid_window_ratio"] == 1.0
    assert summary["mapped_ratio"] > 0.5
    assert 0.0 < summary["default_ratio"] < 0.2

    summary_path = out_dir / "phase_a_summary.json"
    coverage_path = out_dir / "phase_a_feature_coverage.csv"
    risk_path = out_dir / "phase_a_risk_fields.csv"
    assert summary_path.exists()
    assert coverage_path.exists()
    assert risk_path.exists()

    with open(summary_path, "r", encoding="utf-8") as f:
        saved = json.load(f)
    assert saved["rows"] == 160
    assert saved["window_size"] == 150

    with open(coverage_path, "r", encoding="utf-8", newline="") as f:
        coverage_rows = list(csv.DictReader(f))
    assert "feature" in coverage_rows[0]
    assert "mapped_count" in coverage_rows[0]
    actuator_out_1 = next(
        row for row in coverage_rows if row["feature"] == "actuator_out_1"
    )
    actuator_mean = next(
        row for row in coverage_rows if row["feature"] == "actuator_mean"
    )
    assert int(actuator_out_1["mapped_count"]) == 160
    assert int(actuator_mean["derived_count"]) == 160

    with open(risk_path, "r", encoding="utf-8", newline="") as f:
        risk_rows = list(csv.DictReader(f))
    assert set(["feature", "missing_rate", "constant_rate", "risk_note"]).issubset(
        risk_rows[0].keys()
    )
    assert "thrust_z" in {row["feature"] for row in risk_rows}


def test_phase_a_input_gate_marks_windows_invalid_when_time_gap_too_large(tmp_path):
    csv_path = tmp_path / "px4_gap.csv"
    out_dir = tmp_path / "out_gap"
    rows = []
    for i in range(160):
        time_s = i * 0.02
        if i >= 80:
            time_s += 1.0
        rows.append(_full_px4_row(time_s, i, gap_scale=1.0 + i * 0.001))
    _write_csv(csv_path, rows)

    summary = phase_a_input_gate.run_phase_a_input_gate(
        csv_path=csv_path,
        output_dir=out_dir,
        window_size=150,
        max_default_ratio=0.30,
        max_gap_s=0.20,
    )

    assert summary["rows"] == 160
    assert summary["valid_window_ratio"] == 0.0
    assert summary["dt_p95"] >= 0.02


def test_phase_a_input_gate_uses_timestamp_when_time_s_missing(tmp_path):
    csv_path = tmp_path / "px4_timestamp.csv"
    out_dir = tmp_path / "out_ts"
    rows = []
    for i in range(160):
        row = _full_px4_row(0.0, i)
        row.pop("time_s")
        row["Timestamp"] = int(i * 20000)
        rows.append(row)
    _write_csv(csv_path, rows)

    summary = phase_a_input_gate.run_phase_a_input_gate(
        csv_path=csv_path,
        output_dir=out_dir,
        window_size=150,
        max_default_ratio=0.30,
        max_gap_s=0.20,
    )

    assert abs(summary["dt_median"] - 0.02) < 1e-9
    assert summary["valid_window_ratio"] == 1.0


def test_phase_a_input_gate_reports_invalid_time_axis_instead_of_silent_arange(
    tmp_path,
):
    csv_path = tmp_path / "px4_bad_time.csv"
    out_dir = tmp_path / "out_bad_time"
    rows = [_full_px4_row(i * 0.02, i) for i in range(160)]
    rows[40]["time_s"] = "bad"
    _write_csv(csv_path, rows)

    summary = phase_a_input_gate.run_phase_a_input_gate(
        csv_path=csv_path,
        output_dir=out_dir,
        window_size=150,
        max_default_ratio=0.30,
        max_gap_s=0.20,
    )

    assert summary["time_axis_valid"] == 0
    assert summary["valid_window_ratio"] == 0.0
    assert summary["dt_median"] is None
    assert summary["estimated_hz"] is None


def test_phase_a_input_gate_risk_output_covers_broader_key_fields(tmp_path):
    csv_path = tmp_path / "px4_keys.csv"
    out_dir = tmp_path / "out_keys"
    rows = [_full_px4_row(i * 0.02, i) for i in range(160)]
    _write_csv(csv_path, rows)

    phase_a_input_gate.run_phase_a_input_gate(
        csv_path=csv_path,
        output_dir=out_dir,
        window_size=150,
        max_default_ratio=0.30,
        max_gap_s=0.20,
    )

    risk_path = out_dir / "phase_a_risk_fields.csv"
    with open(risk_path, "r", encoding="utf-8", newline="") as f:
        risk_rows = list(csv.DictReader(f))
    features = {row["feature"] for row in risk_rows}
    assert "gyro_x" in features
    assert "accel_z" in features
    assert "vel_x" in features
    assert "baro_alt" in features
    assert "att_q_w" in features
    assert "mag_x" in features
    assert "thrust_z" in features

    gyro_x = next(row for row in risk_rows if row["feature"] == "gyro_x")
    accel_z = next(row for row in risk_rows if row["feature"] == "accel_z")
    thrust_z = next(row for row in risk_rows if row["feature"] == "thrust_z")

    assert float(gyro_x["missing_rate"]) == 0.0
    assert float(accel_z["missing_rate"]) == 0.0
    assert float(accel_z["constant_rate"]) == 1.0
    assert gyro_x["risk_note"] == "key_field"
    assert thrust_z["risk_note"] in {"mostly_default", "observe"}
