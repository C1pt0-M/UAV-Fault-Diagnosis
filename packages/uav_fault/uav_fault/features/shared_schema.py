from __future__ import annotations

RAW_SHARED_FIELDS_V1 = [
    "gyro_x",
    "gyro_y",
    "gyro_z",
    "accel_x",
    "accel_y",
    "accel_z",
    "baro_alt",
    "baro_pressure",
    "att_q0",
    "att_q1",
    "att_q2",
    "att_q3",
    "vel_x",
    "vel_y",
    "vel_z",
]


HIGH_RISK_RAW_FIELDS_V1 = {
    "actuator_controls": [
        "actuator_ctrl_1",
        "actuator_ctrl_2",
        "actuator_ctrl_3",
        "actuator_ctrl_4",
    ],
    "actuator_outputs": [
        "actuator_out_1",
        "actuator_out_2",
        "actuator_out_3",
        "actuator_out_4",
    ],
    "rpm_proxy": [
        "rpm_1",
        "rpm_2",
        "rpm_3",
        "rpm_4",
    ],
}


DERIVED_FEATURES_V1 = {
    "gyro_norm": {
        "kind": "vector_norm",
        "inputs": ["gyro_x", "gyro_y", "gyro_z"],
        "recommended": True,
    },
    "accel_norm": {
        "kind": "vector_norm",
        "inputs": ["accel_x", "accel_y", "accel_z"],
        "recommended": True,
    },
    "velocity_norm": {
        "kind": "vector_norm",
        "inputs": ["vel_x", "vel_y", "vel_z"],
        "recommended": True,
    },
    "baro_alt_rate": {
        "kind": "temporal_derivative",
        "inputs": ["baro_alt"],
        "recommended": True,
    },
    "baro_pressure_rate": {
        "kind": "temporal_derivative",
        "inputs": ["baro_pressure"],
        "recommended": True,
    },
    "tilt_angle": {
        "kind": "quaternion_tilt",
        "inputs": ["att_q0", "att_q1", "att_q2", "att_q3"],
        "recommended": True,
    },
    "gyro_norm_std_w": {
        "kind": "window_std",
        "inputs": ["gyro_norm"],
        "recommended": True,
    },
    "accel_norm_std_w": {
        "kind": "window_std",
        "inputs": ["accel_norm"],
        "recommended": True,
    },
    "velocity_norm_std_w": {
        "kind": "window_std",
        "inputs": ["velocity_norm"],
        "recommended": True,
    },
    "tilt_std_w": {
        "kind": "window_std",
        "inputs": ["tilt_angle"],
        "recommended": True,
    },
}


TRAINING_FEATURES_V1 = RAW_SHARED_FIELDS_V1 + list(DERIVED_FEATURES_V1.keys())
