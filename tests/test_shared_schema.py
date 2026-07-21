from uav_fault.features.shared_schema import (
    DERIVED_FEATURES_V1,
    HIGH_RISK_RAW_FIELDS_V1,
    RAW_SHARED_FIELDS_V1,
    TRAINING_FEATURES_V1,
)


def test_raw_shared_fields_v1_contains_core_cross_platform_inputs():
    assert RAW_SHARED_FIELDS_V1[:3] == ["gyro_x", "gyro_y", "gyro_z"]
    assert "accel_x" in RAW_SHARED_FIELDS_V1
    assert "baro_pressure" in RAW_SHARED_FIELDS_V1
    assert "att_q0" in RAW_SHARED_FIELDS_V1
    assert "vel_z" in RAW_SHARED_FIELDS_V1


def test_high_risk_raw_fields_v1_marks_control_output_and_rpm_chains():
    assert HIGH_RISK_RAW_FIELDS_V1["actuator_controls"] == [
        "actuator_ctrl_1",
        "actuator_ctrl_2",
        "actuator_ctrl_3",
        "actuator_ctrl_4",
    ]
    assert HIGH_RISK_RAW_FIELDS_V1["actuator_outputs"] == [
        "actuator_out_1",
        "actuator_out_2",
        "actuator_out_3",
        "actuator_out_4",
    ]
    assert HIGH_RISK_RAW_FIELDS_V1["rpm_proxy"] == [
        "rpm_1",
        "rpm_2",
        "rpm_3",
        "rpm_4",
    ]


def test_derived_features_v1_contains_strongly_recommended_features():
    assert DERIVED_FEATURES_V1["gyro_norm"]["inputs"] == ["gyro_x", "gyro_y", "gyro_z"]
    assert DERIVED_FEATURES_V1["accel_norm"]["inputs"] == [
        "accel_x",
        "accel_y",
        "accel_z",
    ]
    assert DERIVED_FEATURES_V1["velocity_norm"]["inputs"] == ["vel_x", "vel_y", "vel_z"]
    assert DERIVED_FEATURES_V1["tilt_angle"]["inputs"] == [
        "att_q0",
        "att_q1",
        "att_q2",
        "att_q3",
    ]
    assert DERIVED_FEATURES_V1["baro_alt_rate"]["kind"] == "temporal_derivative"


def test_training_features_v1_extends_raw_inputs_with_recommended_derived_set():
    assert "gyro_x" in TRAINING_FEATURES_V1
    assert "baro_pressure" in TRAINING_FEATURES_V1
    assert "gyro_norm" in TRAINING_FEATURES_V1
    assert "accel_norm_std_w" in TRAINING_FEATURES_V1
    assert "tilt_std_w" in TRAINING_FEATURES_V1
