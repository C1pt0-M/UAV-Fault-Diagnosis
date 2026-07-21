from uav_fault.data.build_px4_shared_cache import (
    build_shared_feature_rows,
    resample_feature_matrix,
)


def test_build_shared_feature_rows_combines_raw_and_derived_features():
    rows = [
        {
            "time_s": 0.0,
            "gyro_x": 3.0,
            "gyro_y": 4.0,
            "gyro_z": 0.0,
            "accel_x": 0.0,
            "accel_y": 0.0,
            "accel_z": 2.0,
            "baro_alt": 10.0,
            "baro_pressure": 100000.0,
            "att_q0": 1.0,
            "att_q1": 0.0,
            "att_q2": 0.0,
            "att_q3": 0.0,
            "vel_x": 1.0,
            "vel_y": 2.0,
            "vel_z": 2.0,
        },
        {
            "time_s": 1.0,
            "gyro_x": 0.0,
            "gyro_y": 0.0,
            "gyro_z": 0.0,
            "accel_x": 0.0,
            "accel_y": 3.0,
            "accel_z": 4.0,
            "baro_alt": 12.0,
            "baro_pressure": 99998.0,
            "att_q0": 0.9238795325,
            "att_q1": 0.0,
            "att_q2": 0.3826834324,
            "att_q3": 0.0,
            "vel_x": 0.0,
            "vel_y": 0.0,
            "vel_z": 5.0,
        },
    ]

    feature_rows = build_shared_feature_rows(rows)

    assert len(feature_rows) == 2
    assert feature_rows[0]["gyro_x"] == 3.0
    assert feature_rows[0]["gyro_norm"] == 5.0
    assert feature_rows[1]["baro_alt_rate"] == 2.0
    assert feature_rows[1]["velocity_norm"] == 5.0
    assert "tilt_std_w" in feature_rows[1]


def test_resample_feature_matrix_resamples_to_target_len_and_column_order():
    rows = [
        {"gyro_x": 0.0, "gyro_norm": 1.0},
        {"gyro_x": 1.0, "gyro_norm": 2.0},
        {"gyro_x": 2.0, "gyro_norm": 3.0},
    ]

    matrix = resample_feature_matrix(rows, ["gyro_x", "gyro_norm"], target_len=5)

    assert matrix.shape == (5, 2)
    assert matrix[0, 0] == 0.0
    assert matrix[-1, 0] == 2.0
    assert matrix[0, 1] == 1.0
    assert matrix[-1, 1] == 3.0
