from __future__ import annotations

import math

import pytest

from uav_fault.models.cnn.lstm import RFLY_SEQ_COLUMNS_V2
from uav_fault.telemetry.schema_adapter import (
    ACTIVE_FEATURE_SCHEMA,
    SEQ_COLUMNS,
    canonicalize_rfly_frame,
)


def test_adapter_emits_the_active_schema() -> None:
    """The runtime now feeds the 26-channel schema that the offline contract uses."""
    assert ACTIVE_FEATURE_SCHEMA == "rfly_seq_columns_v2"
    assert SEQ_COLUMNS == list(RFLY_SEQ_COLUMNS_V2)
    assert len(SEQ_COLUMNS) == 26


def test_rfly_canonical_contract_populates_kg_mstca_features_and_proxy_tags() -> None:
    canonical, coverage = canonicalize_rfly_frame(
        {
            "seq": 1,
            "time_s": 0.02,
            "actuator_source": "proxy",
            "actuator_out_1": 1300.0,
            "actuator_out_2": 1310.0,
            "actuator_out_3": 1290.0,
            "actuator_out_4": 1305.0,
            "roll": 0.1,
            "pitch": -0.2,
            "yaw": 0.3,
            "rollspeed": 0.01,
            "pitchspeed": -0.02,
            "yawspeed": 0.03,
            "accel_x": 0.0,
            "accel_y": 0.1,
            "accel_z": -9.7,
            "vel_x": 1.0,
            "vel_y": -2.0,
            "vel_z": 3.0,
            "baro_alt": 12.0,
            "baro_temp": 25.0,
            "baro_pressure": 101325.0,
        }
    )

    assert len(SEQ_COLUMNS) == 26
    for column in SEQ_COLUMNS:
        assert column in canonical
        assert isinstance(canonical[column], float)
        assert math.isfinite(canonical[column])

    # 被剔除的环境相关通道不再属于 schema。它们仍可能作为已解析的源字段留在帧
    # 字典里（baro_temp 等由 SEQ_COLUMN_ALIASES 解析得到），推理引擎只按 columns
    # 取列，所以不影响输入。
    for dropped in (
        "baro_temp",
        "baro_pressure",
        "actuator_mean",
        "actuator_std",
        "actuator_effort",
        "baro_alt",
    ):
        assert dropped not in SEQ_COLUMNS

    # baro_alt_relative 在单帧阶段携带绝对气压高度，窗口减法由推理引擎完成。
    assert canonical["baro_alt_relative"] == 12.0
    assert canonical["acc_norm"] == pytest.approx(math.sqrt(0.1**2 + 9.7**2))

    assert canonical["actuator_source"] == "proxy"
    assert set(coverage["proxy"]) == {
        "actuator_out_1",
        "actuator_out_2",
        "actuator_out_3",
        "actuator_out_4",
    }
    assert "actuator_source" in coverage["mapped"]
    assert {"att_q_w", "att_q_x", "att_q_y", "att_q_z"}.issubset(
        set(coverage["derived"])
    )
