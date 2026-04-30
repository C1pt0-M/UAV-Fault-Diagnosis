from __future__ import annotations

import sys
from pathlib import Path


CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

from rfly_live_bridge import RflyLiveBridge  # noqa: E402


def test_build_frame_prefers_recv_time_when_sim_time_missing() -> None:
    bridge = RflyLiveBridge(forward_host="127.0.0.1", forward_port=19999)
    try:
        frame = bridge.build_frame(
            mav_state={
                "trueTimeStmp": 0.0,
                "uavTimeStmp": 0.0,
                "recv_time_s": 1.25,
                "uavAngEular": [0.1, 0.2, 0.3],
                "uavVelNED": [1.0, 2.0, 3.0],
                "uavPosNED": [4.0, 5.0, 6.0],
                "trueAccB": [0.4, 0.5, 0.6],
                "trueMotorRPMS": [1000.0, 2000.0, 3000.0, 4000.0],
            }
        )
    finally:
        bridge.close()

    assert frame["time_s"] == 1.25
    assert frame["recv_time_s"] == 1.25
    assert frame["motor1"] > 0.0
    assert frame["vx"] == 1.0
    assert frame["pos_z"] == 6.0


def test_build_frame_uses_sim_time_when_available() -> None:
    bridge = RflyLiveBridge(forward_host="127.0.0.1", forward_port=19998)
    try:
        frame = bridge.build_frame(
            mav_state={
                "trueTimeStmp": 12.5,
                "uavTimeStmp": 8.0,
                "recv_time_s": 1.0,
                "uavAngEular": [0.0, 0.0, 0.0],
                "uavVelNED": [0.0, 0.0, 0.0],
                "uavPosNED": [0.0, 0.0, 0.0],
                "trueAccB": [0.0, 0.0, 0.0],
                "trueMotorRPMS": [0.0, 0.0, 0.0, 0.0],
            }
        )
    finally:
        bridge.close()

    assert frame["time_s"] == 12.5
    assert frame["recv_time_s"] == 1.0


def test_build_frame_derives_rates_accel_and_motor_fallbacks() -> None:
    bridge = RflyLiveBridge(forward_host="127.0.0.1", forward_port=19997)
    try:
        first = bridge.build_frame(
            mav_state={
                "recv_time_s": 10.0,
                "uavAngEular": [0.0, 0.0, 0.0],
                "uavAngRate": [0.0, 0.0, 0.0],
                "uavVelNED": [0.0, 0.0, 0.0],
                "uavPosNED": [0.0, 0.0, 0.0],
                "trueAccB": [0.0, 0.0, 0.0],
                "trueMotorRPMS": [0.0, 0.0, 0.0, 0.0],
                "px4_controls": [0.1, 0.2, 0.3, 0.4],
            },
            pymav_frame={
            },
        )
        second = bridge.build_frame(
            mav_state={
                "recv_time_s": 10.1,
                "uavAngEular": [0.2, 0.1, -0.3],
                "uavAngRate": [0.0, 0.0, 0.0],
                "uavVelNED": [1.0, -2.0, 3.0],
                "uavPosNED": [0.0, 0.0, -1.0],
                "trueAccB": [0.0, 0.0, 0.0],
                "trueMotorRPMS": [0.0, 0.0, 0.0, 0.0],
                "px4_controls": [0.1, 0.2, 0.3, 0.4],
            },
            pymav_frame={
            },
        )
    finally:
        bridge.close()

    assert first["motor1"] > 0.0
    assert first["actuator_ctrl_1"] == 0.1
    assert first["actuator_ctrl_4"] == 0.4
    assert second["rollspeed"] > 0.0
    assert second["pitchspeed"] > 0.0
    assert second["yawspeed"] < 0.0
    assert second["accel_x"] > 0.0
    assert second["accel_y"] < 0.0
    assert second["accel_z"] > 0.0


def test_build_frame_populates_kg_mstca_canonical_fields() -> None:
    bridge = RflyLiveBridge(forward_host="127.0.0.1", forward_port=19996)
    try:
        frame = bridge.build_frame(
            mav_state={
                "recv_time_s": 20.0,
                "uavAngEular": [0.1, -0.2, 0.3],
                "uavAngRate": [0.01, -0.02, 0.03],
                "uavVelNED": [1.0, -2.0, 3.0],
                "uavPosNED": [4.0, 5.0, -6.0],
                "trueAccB": [0.2, -0.3, -9.6],
                "trueMotorRPMS": [0.0, 0.0, 0.0, 0.0],
                "px4_controls": [0.1, -0.2, 0.05, 0.4],
            }
        )
    finally:
        bridge.close()

    for key in [
        "actuator_out_1",
        "actuator_out_2",
        "actuator_out_3",
        "actuator_out_4",
        "att_q_w",
        "att_q_x",
        "att_q_y",
        "att_q_z",
        "vel_x",
        "vel_y",
        "vel_z",
        "baro_alt",
        "baro_temp",
        "baro_pressure",
        "mag_x",
        "mag_y",
        "mag_z",
    ]:
        assert key in frame

    assert frame["actuator_out_1"] > 900.0
    assert frame["actuator_out_4"] > 900.0
    assert frame["vel_x"] == 1.0
    assert frame["vel_y"] == -2.0
    assert frame["vel_z"] == 3.0
    assert frame["baro_pressure"] > 90000.0


def test_build_frame_marks_actuator_source_for_true_and_proxy_outputs() -> None:
    bridge = RflyLiveBridge(forward_host="127.0.0.1", forward_port=19995)
    try:
        true_frame = bridge.build_frame(
            mav_state={
                "recv_time_s": 30.0,
                "uavAngEular": [0.0, 0.0, 0.0],
                "uavAngRate": [0.0, 0.0, 0.0],
                "uavVelNED": [0.0, 0.0, 0.0],
                "uavPosNED": [0.0, 0.0, 0.0],
                "trueAccB": [0.0, 0.0, 0.0],
                "trueMotorRPMS": [1100.0, 1200.0, 1300.0, 1400.0],
            }
        )
        proxy_frame = bridge.build_frame(
            mav_state={
                "recv_time_s": 30.1,
                "uavAngEular": [0.0, 0.0, 0.0],
                "uavAngRate": [0.0, 0.0, 0.0],
                "uavVelNED": [0.0, 0.0, 0.0],
                "uavPosNED": [0.0, 0.0, 0.0],
                "trueAccB": [0.0, 0.0, 0.0],
                "trueMotorRPMS": [0.0, 0.0, 0.0, 0.0],
                "px4_controls": [0.0, 0.0, 0.0, 0.0],
            }
        )
    finally:
        bridge.close()

    assert true_frame["actuator_source"] == "true_motor_rpm"
    assert proxy_frame["actuator_source"] == "proxy"
