from __future__ import annotations

import math

from uav_fault.data.build_sequence_cache import SEQ_COLUMNS
from uav_fault.telemetry.schema_adapter import canonicalize_rfly_frame


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

    assert len(SEQ_COLUMNS) == 31
    for column in SEQ_COLUMNS:
        assert column in canonical
        assert isinstance(canonical[column], float)
        assert math.isfinite(canonical[column])

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
