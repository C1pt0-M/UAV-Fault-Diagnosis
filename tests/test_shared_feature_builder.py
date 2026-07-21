from uav_fault.features.shared_feature_builder import (
    build_point_features,
    build_sequence_features,
)


def test_build_point_features_computes_core_norms_and_tilt():
    frame = {
        "gyro_x": 3.0,
        "gyro_y": 4.0,
        "gyro_z": 0.0,
        "accel_x": 0.0,
        "accel_y": 0.0,
        "accel_z": 2.0,
        "vel_x": 1.0,
        "vel_y": 2.0,
        "vel_z": 2.0,
        "baro_alt": 10.0,
        "baro_pressure": 100000.0,
        "att_q0": 1.0,
        "att_q1": 0.0,
        "att_q2": 0.0,
        "att_q3": 0.0,
    }

    derived = build_point_features(frame)

    assert derived["gyro_norm"] == 5.0
    assert derived["accel_norm"] == 2.0
    assert derived["velocity_norm"] == 3.0
    assert abs(derived["tilt_angle"]) < 1e-9


def test_build_sequence_features_computes_derivatives_and_window_std():
    frames = [
        {
            "time_s": 0.0,
            "gyro_x": 3.0,
            "gyro_y": 4.0,
            "gyro_z": 0.0,
            "accel_x": 0.0,
            "accel_y": 0.0,
            "accel_z": 2.0,
            "vel_x": 1.0,
            "vel_y": 2.0,
            "vel_z": 2.0,
            "baro_alt": 10.0,
            "baro_pressure": 100000.0,
            "att_q0": 1.0,
            "att_q1": 0.0,
            "att_q2": 0.0,
            "att_q3": 0.0,
        },
        {
            "time_s": 1.0,
            "gyro_x": 0.0,
            "gyro_y": 0.0,
            "gyro_z": 0.0,
            "accel_x": 0.0,
            "accel_y": 3.0,
            "accel_z": 4.0,
            "vel_x": 0.0,
            "vel_y": 0.0,
            "vel_z": 5.0,
            "baro_alt": 12.5,
            "baro_pressure": 99998.0,
            "att_q0": 0.9238795325,
            "att_q1": 0.0,
            "att_q2": 0.3826834324,
            "att_q3": 0.0,
        },
    ]

    seq = build_sequence_features(frames, window_size=2)

    assert len(seq) == 2
    assert seq[0]["baro_alt_rate"] == 0.0
    assert seq[1]["baro_alt_rate"] == 2.5
    assert seq[0]["baro_pressure_rate"] == 0.0
    assert seq[1]["baro_pressure_rate"] == -2.0
    assert seq[1]["gyro_norm_std_w"] == 2.5
    assert seq[1]["accel_norm_std_w"] == 1.5
    assert seq[1]["velocity_norm_std_w"] == 1.0
    assert seq[1]["tilt_angle"] > 0.7
    assert seq[1]["tilt_std_w"] > 0.3
