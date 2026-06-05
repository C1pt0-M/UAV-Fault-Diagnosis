from __future__ import annotations

from uav_fault.online.realtime_feature_watch_service import RealtimeFeatureAlertEngine


def _row(scale: float = 1.0) -> dict[str, float]:
    return {
        "time_s": 1.0,
        "roll": 0.01 * scale,
        "pitch": -0.02 * scale,
        "yaw": 0.1 * scale,
        "rollspeed": 0.03 * scale,
        "pitchspeed": 0.02 * scale,
        "yawspeed": 0.01 * scale,
        "vx": 0.2 * scale,
        "vy": -0.1 * scale,
        "vz": 0.05 * scale,
        "accel_x": 0.1 * scale,
        "accel_y": 0.05 * scale,
        "accel_z": 0.02 * scale,
    }


def test_alert_engine_warmup_keeps_normal_stage() -> None:
    engine = RealtimeFeatureAlertEngine(warmup_frames=5, baseline_window=20)

    result = None
    for _ in range(4):
        result = engine.update(_row())

    assert result is not None
    assert result["baseline_ready"] is False
    assert result["warning_stage"] == "normal"


def test_alert_engine_escalates_on_large_motion_shift() -> None:
    engine = RealtimeFeatureAlertEngine(warmup_frames=10, baseline_window=30)

    for _ in range(12):
        engine.update(_row())

    result = engine.update(_row(scale=25.0))

    assert result["baseline_ready"] is True
    assert result["fault_prob"] > 0.0
    assert result["warning_stage"] in {"watch", "warning", "critical"}
