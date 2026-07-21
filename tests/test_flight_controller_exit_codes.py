from uav_fault.telemetry.flight_status import decide_exit_code


def test_decide_exit_code_returns_nonzero_when_arm_failed():
    assert (
        decide_exit_code(arm_ok=False, offboard_ok=False, mission_completed=False) == 2
    )


def test_decide_exit_code_returns_nonzero_when_offboard_failed_after_arm():
    assert (
        decide_exit_code(arm_ok=True, offboard_ok=False, mission_completed=False) == 3
    )


def test_decide_exit_code_returns_zero_when_mission_completed():
    assert decide_exit_code(arm_ok=True, offboard_ok=True, mission_completed=True) == 0
