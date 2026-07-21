from pathlib import Path
import json

import pandas as pd


from uav_fault.data.make_manifest_v2 import build_manifest_v2


def _write_csv(path: Path, rows: list[dict]):
    pd.DataFrame(rows).to_csv(path, index=False)


def test_build_manifest_v2_creates_main_6class_manifest_and_summary(tmp_path: Path):
    csv_dir = tmp_path / "processed_official_50hz"
    csv_dir.mkdir()

    _write_csv(
        csv_dir / "Case_1000000000.csv",
        [
            {
                "Index": 0,
                "Timestamp": 1000000,
                "trueTime": 1.0,
                "_sensor_combined_0_gyro_rad[0]": 0.1,
                "_sensor_combined_0_gyro_rad[1]": 0.2,
                "_sensor_combined_0_gyro_rad[2]": 0.3,
                "_sensor_combined_0_accelerometer_m_s2[0]": 1.0,
                "_sensor_combined_0_accelerometer_m_s2[1]": 2.0,
                "_sensor_combined_0_accelerometer_m_s2[2]": 3.0,
                "_vehicle_air_data_0_baro_alt_meter": 10.0,
                "_vehicle_air_data_0_baro_pressure_pa": 100000.0,
                "_vehicle_attitude_0_q[0]": 1.0,
                "_vehicle_attitude_0_q[1]": 0.0,
                "_vehicle_attitude_0_q[2]": 0.0,
                "_vehicle_attitude_0_q[3]": 0.0,
                "_vehicle_local_position_0_vx": 0.0,
                "_vehicle_local_position_0_vy": 0.0,
                "_vehicle_local_position_0_vz": 0.0,
                "_vehicle_magnetometer_0_magnetometer_ga[0]": 0.1,
                "_vehicle_magnetometer_0_magnetometer_ga[1]": 0.2,
                "_vehicle_magnetometer_0_magnetometer_ga[2]": 0.3,
                "TrueState_data_fault_state": 0,
                "UAVState_data_cmd": "cmd_a",
                "TrueState_data_motorRPMs[1]": 1000.0,
                "TrueState_data_motorRPMs[2]": 1001.0,
                "TrueState_data_motorRPMs[3]": 1002.0,
                "TrueState_data_motorRPMs[4]": 1003.0,
            },
            {
                "Index": 1,
                "Timestamp": 1020000,
                "trueTime": 1.02,
                "_sensor_combined_0_gyro_rad[0]": 0.1,
                "_sensor_combined_0_gyro_rad[1]": 0.2,
                "_sensor_combined_0_gyro_rad[2]": 0.3,
                "_sensor_combined_0_accelerometer_m_s2[0]": 1.0,
                "_sensor_combined_0_accelerometer_m_s2[1]": 2.0,
                "_sensor_combined_0_accelerometer_m_s2[2]": 3.0,
                "_vehicle_air_data_0_baro_alt_meter": 10.0,
                "_vehicle_air_data_0_baro_pressure_pa": 100000.0,
                "_vehicle_attitude_0_q[0]": 1.0,
                "_vehicle_attitude_0_q[1]": 0.0,
                "_vehicle_attitude_0_q[2]": 0.0,
                "_vehicle_attitude_0_q[3]": 0.0,
                "_vehicle_local_position_0_vx": 0.0,
                "_vehicle_local_position_0_vy": 0.0,
                "_vehicle_local_position_0_vz": 0.0,
                "_vehicle_magnetometer_0_magnetometer_ga[0]": 0.1,
                "_vehicle_magnetometer_0_magnetometer_ga[1]": 0.2,
                "_vehicle_magnetometer_0_magnetometer_ga[2]": 0.3,
                "TrueState_data_fault_state": 1,
                "UAVState_data_cmd": "cmd_b",
                "TrueState_data_motorRPMs[1]": 1004.0,
                "TrueState_data_motorRPMs[2]": 1005.0,
                "TrueState_data_motorRPMs[3]": 1006.0,
                "TrueState_data_motorRPMs[4]": 1007.0,
            },
        ],
    )

    _write_csv(
        csv_dir / "Case_1403000001.csv",
        [
            {
                "Index": 0,
                "Timestamp": 2000000,
                "trueTime": 2.0,
                "_sensor_combined_0_gyro_rad[0]": 0.1,
                "_sensor_combined_0_gyro_rad[1]": 0.2,
                "_sensor_combined_0_gyro_rad[2]": 0.3,
                "_sensor_combined_0_accelerometer_m_s2[0]": 1.0,
                "_sensor_combined_0_accelerometer_m_s2[1]": 2.0,
                "_sensor_combined_0_accelerometer_m_s2[2]": 3.0,
                "_vehicle_air_data_0_baro_alt_meter": 10.0,
                "_vehicle_air_data_0_baro_pressure_pa": 100000.0,
                "_vehicle_attitude_0_q[0]": 1.0,
                "_vehicle_attitude_0_q[1]": 0.0,
                "_vehicle_attitude_0_q[2]": 0.0,
                "_vehicle_attitude_0_q[3]": 0.0,
                "_vehicle_local_position_0_vx": 0.0,
                "_vehicle_local_position_0_vy": 0.0,
                "_vehicle_local_position_0_vz": 0.0,
                "_vehicle_magnetometer_0_magnetometer_ga[0]": 0.1,
                "_vehicle_magnetometer_0_magnetometer_ga[1]": 0.2,
                "_vehicle_magnetometer_0_magnetometer_ga[2]": 0.3,
            }
        ],
    )

    output_path = tmp_path / "manifest_v2.csv"
    summary_path = tmp_path / "manifest_v2.summary.json"

    build_manifest_v2(
        csv_dir=csv_dir,
        output_path=output_path,
        summary_output_path=summary_path,
        schema_version="rflymad_manifest_v2",
        build_version="test-build",
    )

    manifest = pd.read_csv(output_path)
    assert list(manifest["filename"]) == ["Case_1000000000.csv", "Case_1403000001.csv"]

    motor_row = manifest.loc[manifest["filename"] == "Case_1000000000.csv"].iloc[0]
    assert motor_row["platform"] == "SIL"
    assert motor_row["flight_mode"] == "hover"
    assert motor_row["fault_type_canonical"] == "motor"
    assert motor_row["label_main_6class"] == "motor"
    assert motor_row["label_main_6class_id"] == 1
    assert motor_row["is_in_main_6class"] == 1
    assert motor_row["label_binary"] == "fault"
    assert motor_row["time_col"] == "trueTime"
    assert abs(motor_row["duration_sec"] - 0.02) < 1e-9
    assert abs(motor_row["sample_rate_hz_est"] - 50.0) < 1e-6
    assert motor_row["has_true_fault_state"] == 1
    assert motor_row["has_cmd"] == 1
    assert motor_row["has_motor_rpm"] == 1
    assert motor_row["fault_annotation_source"] == "TrueState_data_fault_state"
    assert motor_row["has_fault_annotation"] == 1
    assert motor_row["fault_start_idx"] == 1
    assert motor_row["fault_end_idx"] == 1
    assert abs(motor_row["fault_start_sec"] - 1.02) < 1e-9
    assert motor_row["quality_flag"] == "ok"

    wind_row = manifest.loc[manifest["filename"] == "Case_1403000001.csv"].iloc[0]
    assert wind_row["fault_type_canonical"] == "wind_affect"
    assert wind_row["label_main_6class"] == "wind"
    assert wind_row["label_main_6class_id"] == 4
    assert wind_row["is_in_main_6class"] == 1
    assert wind_row["label_binary"] == "fault"
    assert wind_row["has_fault_annotation"] == 0

    summary = json.loads(summary_path.read_text())
    assert summary["total_cases"] == 2
    assert summary["platform_counts"] == {"SIL": 2}
    assert summary["label_main_6class_counts"] == {"motor": 1, "wind": 1}
    assert summary["in_main_6class_count"] == 2


def test_build_manifest_v2_marks_low_voltage_outside_main_6class(tmp_path: Path):
    csv_dir = tmp_path / "processed_official_50hz"
    csv_dir.mkdir()

    _write_csv(
        csv_dir / "Case_1002000000.csv",
        [
            {
                "Index": 0,
                "Timestamp": 1000000,
                "trueTime": 1.0,
                "_sensor_combined_0_gyro_rad[0]": 0.1,
                "_sensor_combined_0_gyro_rad[1]": 0.2,
                "_sensor_combined_0_gyro_rad[2]": 0.3,
                "_sensor_combined_0_accelerometer_m_s2[0]": 1.0,
                "_sensor_combined_0_accelerometer_m_s2[1]": 2.0,
                "_sensor_combined_0_accelerometer_m_s2[2]": 3.0,
                "_vehicle_air_data_0_baro_alt_meter": 10.0,
                "_vehicle_air_data_0_baro_pressure_pa": 100000.0,
                "_vehicle_attitude_0_q[0]": 1.0,
                "_vehicle_attitude_0_q[1]": 0.0,
                "_vehicle_attitude_0_q[2]": 0.0,
                "_vehicle_attitude_0_q[3]": 0.0,
                "_vehicle_local_position_0_vx": 0.0,
                "_vehicle_local_position_0_vy": 0.0,
                "_vehicle_local_position_0_vz": 0.0,
                "_vehicle_magnetometer_0_magnetometer_ga[0]": 0.1,
                "_vehicle_magnetometer_0_magnetometer_ga[1]": 0.2,
                "_vehicle_magnetometer_0_magnetometer_ga[2]": 0.3,
            }
        ],
    )

    output_path = tmp_path / "manifest_v2.csv"
    summary_path = tmp_path / "manifest_v2.summary.json"
    build_manifest_v2(
        csv_dir, output_path, summary_path, "rflymad_manifest_v2", "test-build"
    )

    row = pd.read_csv(output_path).iloc[0]
    assert row["fault_type_canonical"] == "low_voltage"
    assert pd.isna(row["label_main_6class"])
    assert pd.isna(row["label_main_6class_id"])
    assert row["is_in_main_6class"] == 0


def test_build_manifest_v2_merges_testinfo_metadata(tmp_path: Path):
    csv_dir = tmp_path / "processed_official_50hz"
    csv_dir.mkdir()

    _write_csv(
        csv_dir / "Case_1000000000.csv",
        [
            {
                "Index": 0,
                "Timestamp": 1000000,
                "trueTime": 1.0,
                "_sensor_combined_0_gyro_rad[0]": 0.1,
                "_sensor_combined_0_gyro_rad[1]": 0.2,
                "_sensor_combined_0_gyro_rad[2]": 0.3,
                "_sensor_combined_0_accelerometer_m_s2[0]": 1.0,
                "_sensor_combined_0_accelerometer_m_s2[1]": 2.0,
                "_sensor_combined_0_accelerometer_m_s2[2]": 3.0,
                "_vehicle_air_data_0_baro_alt_meter": 10.0,
                "_vehicle_air_data_0_baro_pressure_pa": 100000.0,
                "_vehicle_attitude_0_q[0]": 1.0,
                "_vehicle_attitude_0_q[1]": 0.0,
                "_vehicle_attitude_0_q[2]": 0.0,
                "_vehicle_attitude_0_q[3]": 0.0,
                "_vehicle_local_position_0_vx": 0.0,
                "_vehicle_local_position_0_vy": 0.0,
                "_vehicle_local_position_0_vz": 0.0,
                "_vehicle_magnetometer_0_magnetometer_ga[0]": 0.1,
                "_vehicle_magnetometer_0_magnetometer_ga[1]": 0.2,
                "_vehicle_magnetometer_0_magnetometer_ga[2]": 0.3,
            }
        ],
    )

    testinfo_path = tmp_path / "sil_testinfo_metadata.csv"
    pd.DataFrame(
        [
            {
                "platform": "SIL",
                "flight_mode": "hover",
                "fault_type_canonical": "motor",
                "case_dir": "TestCase_1_1000000000",
                "control_sequence_raw": "['cmd1' 'cmd2']",
                "fault_id_numeric": 123450,
                "fault_params_json": "[1.0, 0.9]",
                "fault_injection_time_sec": 12.0,
                "test_end_time_sec": 34.0,
                "raw_collection_hz": 131.0,
            }
        ]
    ).to_csv(testinfo_path, index=False)

    output_path = tmp_path / "manifest_v2.csv"
    summary_path = tmp_path / "manifest_v2.summary.json"

    build_manifest_v2(
        csv_dir=csv_dir,
        output_path=output_path,
        summary_output_path=summary_path,
        schema_version="rflymad_manifest_v2",
        build_version="test-build",
        testinfo_metadata_paths=[testinfo_path],
    )

    row = pd.read_csv(output_path).iloc[0]
    assert row["control_sequence_raw"] == "['cmd1' 'cmd2']"
    assert row["fault_id_numeric"] == 123450
    assert row["fault_params_json"] == "[1.0, 0.9]"
    assert row["fault_injection_time_sec"] == 12.0
    assert row["test_end_time_sec"] == 34.0
    assert row["raw_collection_hz"] == 131.0


def test_build_manifest_v2_prefers_existing_bucket_case_num(tmp_path: Path):
    csv_dir = tmp_path / "processed_official_50hz"
    csv_dir.mkdir()

    _write_csv(
        csv_dir / "Case_1000000000.csv",
        [
            {
                "Index": 0,
                "Timestamp": 1000000,
                "trueTime": 1.0,
                "_sensor_combined_0_gyro_rad[0]": 0.1,
                "_sensor_combined_0_gyro_rad[1]": 0.2,
                "_sensor_combined_0_gyro_rad[2]": 0.3,
                "_sensor_combined_0_accelerometer_m_s2[0]": 1.0,
                "_sensor_combined_0_accelerometer_m_s2[1]": 2.0,
                "_sensor_combined_0_accelerometer_m_s2[2]": 3.0,
                "_vehicle_air_data_0_baro_alt_meter": 10.0,
                "_vehicle_air_data_0_baro_pressure_pa": 100000.0,
                "_vehicle_attitude_0_q[0]": 1.0,
                "_vehicle_attitude_0_q[1]": 0.0,
                "_vehicle_attitude_0_q[2]": 0.0,
                "_vehicle_attitude_0_q[3]": 0.0,
                "_vehicle_local_position_0_vx": 0.0,
                "_vehicle_local_position_0_vy": 0.0,
                "_vehicle_local_position_0_vz": 0.0,
                "_vehicle_magnetometer_0_magnetometer_ga[0]": 0.1,
                "_vehicle_magnetometer_0_magnetometer_ga[1]": 0.2,
                "_vehicle_magnetometer_0_magnetometer_ga[2]": 0.3,
            }
        ],
    )

    _write_csv(
        csv_dir / "Case_1000000001.csv",
        [
            {
                "Index": 0,
                "Timestamp": 1000000,
                "trueTime": 1.0,
                "_sensor_combined_0_gyro_rad[0]": 0.1,
                "_sensor_combined_0_gyro_rad[1]": 0.2,
                "_sensor_combined_0_gyro_rad[2]": 0.3,
                "_sensor_combined_0_accelerometer_m_s2[0]": 1.0,
                "_sensor_combined_0_accelerometer_m_s2[1]": 2.0,
                "_sensor_combined_0_accelerometer_m_s2[2]": 3.0,
                "_vehicle_air_data_0_baro_alt_meter": 10.0,
                "_vehicle_air_data_0_baro_pressure_pa": 100000.0,
                "_vehicle_attitude_0_q[0]": 1.0,
                "_vehicle_attitude_0_q[1]": 0.0,
                "_vehicle_attitude_0_q[2]": 0.0,
                "_vehicle_attitude_0_q[3]": 0.0,
                "_vehicle_local_position_0_vx": 0.0,
                "_vehicle_local_position_0_vy": 0.0,
                "_vehicle_local_position_0_vz": 0.0,
                "_vehicle_magnetometer_0_magnetometer_ga[0]": 0.1,
                "_vehicle_magnetometer_0_magnetometer_ga[1]": 0.2,
                "_vehicle_magnetometer_0_magnetometer_ga[2]": 0.3,
            }
        ],
    )

    testinfo_path = tmp_path / "sil_testinfo_metadata.csv"
    pd.DataFrame(
        [
            {
                "platform": "SIL",
                "flight_mode": "hover",
                "fault_type_canonical": "motor",
                "case_dir": "TestCase_200_bar",
                "bucket_case_num": 0,
                "control_sequence_raw": "['mapped_to_case_0']",
            },
            {
                "platform": "SIL",
                "flight_mode": "hover",
                "fault_type_canonical": "motor",
                "case_dir": "TestCase_100_foo",
                "bucket_case_num": 1,
                "control_sequence_raw": "['mapped_to_case_1']",
            },
        ]
    ).to_csv(testinfo_path, index=False)

    output_path = tmp_path / "manifest_v2.csv"
    summary_path = tmp_path / "manifest_v2.summary.json"

    build_manifest_v2(
        csv_dir=csv_dir,
        output_path=output_path,
        summary_output_path=summary_path,
        schema_version="rflymad_manifest_v2",
        build_version="test-build",
        testinfo_metadata_paths=[testinfo_path],
    )

    manifest = pd.read_csv(output_path)
    row0 = manifest.loc[manifest["filename"] == "Case_1000000000.csv"].iloc[0]
    row1 = manifest.loc[manifest["filename"] == "Case_1000000001.csv"].iloc[0]
    assert row0["control_sequence_raw"] == "['mapped_to_case_0']"
    assert row1["control_sequence_raw"] == "['mapped_to_case_1']"
