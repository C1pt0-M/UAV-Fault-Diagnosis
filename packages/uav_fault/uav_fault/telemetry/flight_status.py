from __future__ import annotations


def decide_exit_code(
    *, arm_ok: bool, offboard_ok: bool, mission_completed: bool
) -> int:
    if not arm_ok:
        return 2
    if not offboard_ok:
        return 3
    if not mission_completed:
        return 4
    return 0


def ready_to_arm_from_health(*, global_ok: bool, home_ok: bool, armable: bool) -> bool:
    return global_ok and home_ok and armable
