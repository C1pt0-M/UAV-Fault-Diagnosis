from __future__ import annotations

from collections import deque
from types import SimpleNamespace

from uav_fault.online.detection_policy import OnlineDetectionPolicy
from uav_fault.online.monitoring_session import OnlineMonitoringSession
from uav_fault.online.realtime_detection_service import RealtimeDetectionService


class _CanonicalRunner:
    window_size = 2
    aggregator = SimpleNamespace(onset_th=0.6)

    def __init__(self):
        self.last_window = None

    def update(self, *, frame, window, frame_idx):
        self.last_window = window
        return ({"label": "normal", "confidence": 0.9, "fault_prob": 0.1}, [])

    def flush(self):
        return []


def test_realtime_service_canonicalizes_simulation_frames_before_detection():
    runner = _CanonicalRunner()
    service = RealtimeDetectionService.__new__(RealtimeDetectionService)
    service.runner = runner
    service.policy = OnlineDetectionPolicy()
    service.detector = SimpleNamespace(window_size=2)
    service.aggregator = runner.aggregator
    service.monitoring_session = OnlineMonitoringSession()
    service.monitoring_session.start_detection(runner)
    service.canonical_window = deque(maxlen=2)
    service.input_contract_status = None
    service.frame_count = 0
    service.detection_count = 0
    service.fault_events = []

    for index in range(2):
        service._on_frame(
            {
                "time_s": index / 50.0,
                "motor1": 1200.0,
                "motor2": 1200.0,
                "motor3": 1200.0,
                "motor4": 1200.0,
                "accel_z": -9.8,
                "vx": 0.2,
                "baro_alt_meter": 1.0,
                "roll": 0.0,
                "pitch": 0.0,
                "yaw": 0.0,
            },
            deque(),
        )

    assert service.detection_count == 1
    assert runner.last_window is not None
    assert runner.last_window[-1]["actuator_out_1"] == 1200.0
    assert runner.last_window[-1]["vel_x"] == 0.2
    assert service.input_contract_status["detection_blocked"] is False
