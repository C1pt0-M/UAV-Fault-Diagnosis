from __future__ import annotations

from uav_fault.online.realtime_feature_watcher import RealtimeFeatureWatcher


def _stable_row() -> dict[str, float]:
    return {
        "roll": 0.01,
        "pitch": -0.02,
        "yaw": 0.1,
        "rollspeed": 0.02,
        "pitchspeed": 0.01,
        "yawspeed": 0.03,
        "vx": 0.2,
        "vy": -0.1,
        "vz": 0.05,
        "accel_x": 0.1,
        "accel_y": 0.05,
        "accel_z": 0.02,
        "time_s": 1.0,
    }


def test_watcher_stays_in_warmup_before_baseline_ready() -> None:
    watcher = RealtimeFeatureWatcher(warmup_frames=5, baseline_window=20)

    result = None
    for _ in range(4):
        result = watcher.update(_stable_row())

    assert result is not None
    assert result["baseline_ready"] is False
    assert result["fault_prob"] == 0.0


def test_watcher_scores_anomalous_motion_higher_than_baseline() -> None:
    watcher = RealtimeFeatureWatcher(warmup_frames=20, baseline_window=60)

    for _ in range(25):
        baseline_result = watcher.update(_stable_row())

    anomaly_row = _stable_row()
    anomaly_row.update(
        {
            "roll": 0.8,
            "pitch": -0.6,
            "rollspeed": 1.2,
            "pitchspeed": 1.0,
            "yawspeed": 1.5,
            "vx": 2.5,
            "vy": -3.0,
            "vz": 1.8,
            "accel_x": 4.0,
            "accel_y": -3.5,
            "accel_z": 2.8,
        }
    )
    anomaly_result = watcher.update(anomaly_row)

    assert baseline_result["baseline_ready"] is True
    assert anomaly_result["baseline_ready"] is True
    assert anomaly_result["fault_prob"] > baseline_result["fault_prob"]
    assert anomaly_result["component_scores"]["attitude"] > 0.0
    assert anomaly_result["component_scores"]["angular_rate"] > 0.0
    assert anomaly_result["component_scores"]["acceleration"] > 0.0


def test_watcher_handles_missing_fields_as_zero_without_crashing() -> None:
    watcher = RealtimeFeatureWatcher(warmup_frames=3, baseline_window=10)

    for _ in range(3):
        watcher.update({"time_s": 1.0})

    result = watcher.update({"time_s": 2.0, "vx": 1.0})

    assert result["baseline_ready"] is True
    assert 0.0 <= result["fault_prob"] <= 1.0
