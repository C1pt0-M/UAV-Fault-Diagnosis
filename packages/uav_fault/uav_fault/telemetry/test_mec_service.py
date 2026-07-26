from uav_fault.telemetry.mec_service import MecTelemetryService


def _service() -> MecTelemetryService:
    return MecTelemetryService(
        listen_host="127.0.0.1",
        listen_port=0,
        forward_host="127.0.0.1",
        forward_port=0,
        model_dir="unused",
        window_size=5,
        stride=1,
    )


def test_disarmed_frames_do_not_enter_mec_detection_window() -> None:
    service = _service()

    for frame_idx in range(20):
        decorated = service._handle_frame(
            {"time_s": frame_idx / 50.0, "armed": 0.0, "pos_z": 0.0}
        )

    assert len(service.window) == 0
    assert service.inference_count == 0
    assert decorated["mec_fault_prob"] == 0.0
    assert decorated["mec_alert_active"] == 0
    assert decorated["mec_contract_status"] == "waiting_for_arm"
    assert decorated["mec_model_status"] == "waiting_for_arm"


def test_arming_starts_a_fresh_mec_window() -> None:
    service = _service()
    service.window.append({"time_s": 0.0})
    service._publish_result(
        {"fault_prob": 0.99, "label": "fault", "alert_active": True}
    )

    service._handle_frame({"time_s": 0.02, "armed": 0.0, "pos_z": 0.0})
    decorated = service._handle_frame(
        {"time_s": 0.04, "armed": 1.0, "pos_z": -0.1}
    )

    assert len(service.window) == 1
    assert service.flight_active is True
    assert decorated["mec_fault_prob"] == 0.0
    assert decorated["mec_alert_active"] == 0
    assert decorated["mec_contract_status"] == "waiting_for_window"
    assert decorated["mec_model_status"] == "waiting_for_window"


def test_decorated_frame_exposes_inference_latency_statistics() -> None:
    service = _service()
    service.inference_samples.extend([2.0, 4.0, 8.0])
    service.inference_count = 3
    service._publish_result({"inference_ms": 8.0})

    decorated = service._decorate_frame({})

    assert decorated["mec_inference_ms"] == 8.0
    assert decorated["mec_avg_inference_ms"] == 4.667
    assert decorated["mec_p95_inference_ms"] == 4.0
    assert decorated["mec_inference_count"] == 3
