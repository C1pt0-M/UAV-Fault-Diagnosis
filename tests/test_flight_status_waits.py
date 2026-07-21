from uav_fault.telemetry.flight_status import ready_to_arm_from_health


def test_ready_to_arm_from_health_requires_global_home_and_armable():
    assert ready_to_arm_from_health(global_ok=True, home_ok=True, armable=True) is True
    assert (
        ready_to_arm_from_health(global_ok=True, home_ok=True, armable=False) is False
    )
    assert (
        ready_to_arm_from_health(global_ok=False, home_ok=True, armable=True) is False
    )
    assert (
        ready_to_arm_from_health(global_ok=True, home_ok=False, armable=True) is False
    )
