from pathlib import Path

from uav_fault.online.run_detector import run_csv_mode


class _DummyDetector:
    def __init__(self) -> None:
        self.windows = []

    def predict(self, window):
        self.windows.append([dict(row) for row in window])
        return {
            "label": "normal",
            "confidence": 0.8,
            "fault_prob": 0.2,
            "raw_logits": [1.0, 0.0],
        }


def test_run_csv_mode_canonicalizes_new_processed_csv_before_inference(tmp_path: Path):
    csv_path = tmp_path / "case_new_schema.csv"
    csv_path.write_text(
        "\n".join(
            [
                "Index,Timestamp,trueTime,_actuator_controls_0_0_control[0],_actuator_outputs_0_output[0],_sensor_combined_0_gyro_rad[0],_sensor_combined_0_accelerometer_m_s2[0],_sensor_combined_0_accelerometer_m_s2[1],_sensor_combined_0_accelerometer_m_s2[2],_vehicle_air_data_0_baro_alt_meter,_vehicle_air_data_0_baro_temp_celcius,_vehicle_air_data_0_baro_pressure_pa,_vehicle_attitude_0_q[0],_vehicle_attitude_0_q[1],_vehicle_attitude_0_q[2],_vehicle_attitude_0_q[3],_vehicle_local_position_0_vx,_vehicle_local_position_0_vy,_vehicle_local_position_0_vz,_vehicle_magnetometer_0_magnetometer_ga[0],_vehicle_magnetometer_0_magnetometer_ga[1],_vehicle_magnetometer_0_magnetometer_ga[2]",
                "1,1500000,1.5,0.11,1200,0.01,1.0,2.0,-9.7,15.2,25.1,101300,1.0,0.0,0.0,0.0,0.3,0.4,-0.5,0.01,0.02,0.03",
                "2,2500000,2.5,0.12,1210,0.02,1.1,2.1,-9.6,15.4,25.2,101280,0.99,0.0,0.0,0.1,0.31,0.41,-0.51,0.011,0.021,0.031",
            ]
        ),
        encoding="utf-8",
    )

    detector = _DummyDetector()

    df, results = run_csv_mode(detector, str(csv_path), window_size=2, every_n=1)

    assert list(df["time_s"]) == [1.5, 2.5]
    assert [item["time_s"] for item in results] == [2.5]

    assert len(detector.windows) == 1
    first_row = detector.windows[0][0]
    second_row = detector.windows[0][1]
    assert first_row["time_s"] == 1.5
    assert second_row["time_s"] == 2.5
    assert first_row["seq"] == 1.0
    assert second_row["seq"] == 2.0
    assert first_row["actuator_ctrl_1"] == 0.11
    assert first_row["actuator_out_1"] == 1200.0
    assert first_row["gyro_x"] == 0.01
    assert first_row["accel_z"] == -9.7
    assert first_row["baro_alt"] == 15.2
    assert first_row["baro_temp"] == 25.1
    assert first_row["baro_pressure"] == 101300.0
    assert first_row["att_q_w"] == 1.0
    assert first_row["vel_x"] == 0.3
    assert first_row["mag_z"] == 0.03
    assert "actuator_mean" in first_row
