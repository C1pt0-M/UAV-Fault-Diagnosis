from uav_fault.telemetry.receiver import _required_fields_ready


def test_required_fields_ready_rejects_warmup_frames():
    fields = ["pymav_highres_imu_seen", "pymav_actuator_control_seen"]

    assert not _required_fields_ready(
        {"pymav_highres_imu_seen": 1.0, "pymav_actuator_control_seen": 0.0},
        fields,
    )
    assert _required_fields_ready(
        {"pymav_highres_imu_seen": 1.0, "pymav_actuator_control_seen": 1.0},
        fields,
    )
