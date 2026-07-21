import importlib.util
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = PROJECT_ROOT / "scripts" / "data" / "merge_buaa_ulog_case.py"
spec = importlib.util.spec_from_file_location("merge_buaa_ulog_case", MODULE_PATH)
assert spec is not None
assert spec.loader is not None
merge_buaa_ulog_case = importlib.util.module_from_spec(spec)
spec.loader.exec_module(merge_buaa_ulog_case)


def _write_csv(path: Path, rows: list[dict]) -> None:
    pd.DataFrame(rows).to_csv(path, index=False)


def test_merge_case_outputs_rflymad_schema_with_explicit_no_fault_labels(tmp_path):
    case_dir = tmp_path / "TestCase_189_1419000188"
    log_dir = case_dir / "Log"
    log_dir.mkdir(parents=True)
    (case_dir / "TestInfo.csv").write_text(
        "Case ID,189\n"
        "Control Sequence,\"['2,1,0.5,1,15' '1,1,5']\"\n"
        "Fault ID,[]\n",
        encoding="utf-8",
    )

    _write_csv(
        log_dir / "14_06_21_sensor_combined_0.csv",
        [
            {
                "timestamp": 1000,
                "gyro_rad[0]": 0.1,
                "gyro_rad[1]": 0.2,
                "gyro_rad[2]": 0.3,
                "accelerometer_m_s2[0]": 0.0,
                "accelerometer_m_s2[1]": 0.1,
                "accelerometer_m_s2[2]": -9.8,
            },
            {
                "timestamp": 2000,
                "gyro_rad[0]": 0.2,
                "gyro_rad[1]": 0.3,
                "gyro_rad[2]": 0.4,
                "accelerometer_m_s2[0]": 0.1,
                "accelerometer_m_s2[1]": 0.2,
                "accelerometer_m_s2[2]": -9.7,
            },
        ],
    )
    _write_csv(
        log_dir / "14_06_21_actuator_outputs_0.csv",
        [
            {"timestamp": 1000, "output[0]": 1100.0, "output[1]": 1110.0, "output[2]": 1090.0, "output[3]": 1105.0},
            {"timestamp": 2000, "output[0]": 1200.0, "output[1]": 1210.0, "output[2]": 1190.0, "output[3]": 1205.0},
        ],
    )
    _write_csv(
        log_dir / "14_06_21_estimator_attitude_0.csv",
        [{"timestamp": 1000, "q[0]": 1.0, "q[1]": 0.0, "q[2]": 0.0, "q[3]": 0.0}],
    )
    _write_csv(
        log_dir / "14_06_21_vehicle_magnetometer_0.csv",
        [{"timestamp": 1000, "magnetometer_ga[0]": 0.27, "magnetometer_ga[1]": -0.03, "magnetometer_ga[2]": 0.47}],
    )
    _write_csv(
        log_dir / "14_06_21_vehicle_gps_position_0.csv",
        [{"timestamp": 1000, "vel_n_m_s": 0.5, "vel_e_m_s": -0.1, "vel_d_m_s": -0.2}],
    )
    _write_csv(
        log_dir / "14_06_21_vehicle_global_position_0.csv",
        [{"timestamp": 1000, "lat": 40.0, "lon": 116.0, "alt": 12.0}],
    )

    merged = merge_buaa_ulog_case.merge_case(case_dir, target_hz=None)

    assert list(merged.columns) == [
        *merge_buaa_ulog_case.RFlyMAD_COLUMNS,
        "is_fault_active",
        "flight_phase",
    ]
    assert len(merged) == 2
    assert merged["is_fault_active"].tolist() == [0, 0]
    assert merged["flight_phase"].tolist() == ["normal", "normal"]
    assert merged["_actuator_outputs_0_output[0]"].tolist() == [1100.0, 1200.0]
    assert merged["_vehicle_air_data_0_baro_pressure_pa"].tolist() == [101325.0, 101325.0]
    assert merged["_vehicle_local_position_0_vx"].tolist() == [0.5, 0.5]
    assert merged["_vehicle_local_position_0_vz"].tolist() == [-0.2, -0.2]
    assert merged["_vehicle_attitude_0_q[0]"].tolist() == [1.0, 1.0]
    assert merged["_vehicle_magnetometer_0_magnetometer_ga[2]"].tolist() == [0.47, 0.47]
    assert merged["UAVState_data_fault_state"].tolist() == [0, 0]
