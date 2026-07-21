from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = (
    PROJECT_ROOT
    / "packages"
    / "uav_fault"
    / "uav_fault"
    / "data"
    / "build_sequence_cache.py"
)
SPEC = importlib.util.spec_from_file_location("build_sequence_cache", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
cache = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(cache)


def test_extract_channels_reads_live_canonical_columns():
    frame = {
        "actuator_ctrl_1": 0.1,
        "actuator_ctrl_2": 0.2,
        "actuator_ctrl_3": 0.3,
        "actuator_ctrl_4": 0.4,
        "actuator_out_1": 1301.0,
        "actuator_out_2": 1302.0,
        "actuator_out_3": 1303.0,
        "actuator_out_4": 1304.0,
        "gyro_x": 0.01,
        "gyro_y": 0.02,
        "gyro_z": 0.03,
        "accel_x": 0.1,
        "accel_y": 0.2,
        "accel_z": -9.8,
        "baro_alt_meter": 55.0,
        "baro_temp": 15.0,
        "baro_pressure_pa": 100700.0,
        "roll": 0.0,
        "pitch": 0.0,
        "yaw": 0.0,
        "vx": 1.0,
        "vy": 2.0,
        "vz": 3.0,
        "mag_x": 0.25,
        "mag_y": -0.03,
        "mag_z": 0.47,
    }

    values = cache._extract_channels(pd.DataFrame([frame]))[0]
    by_name = dict(zip(cache.SEQ_COLUMNS, values))

    expected = {
        "actuator_ctrl_4": 0.4,
        "actuator_out_1": 1301.0,
        "accel_z": -9.8,
        "baro_alt": 55.0,
        "baro_pressure": 100700.0,
        "att_q_w": 1.0,
        "att_q_x": 0.0,
        "att_q_y": 0.0,
        "att_q_z": 0.0,
        "vel_x": 1.0,
        "vel_y": 2.0,
        "vel_z": 3.0,
        "mag_z": 0.47,
    }
    for name, value in expected.items():
        assert np.isclose(by_name[name], value)
