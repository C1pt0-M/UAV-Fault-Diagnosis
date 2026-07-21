from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
from uav_fault.online.fault_detector import FaultDetector


MODEL_DIR = str(PROJECT_ROOT / "assets" / "models" / "online_detector_cnn1d_rfly")


def test_fault_detector_uses_norm_stats_window_size_when_constructor_window_missing():
    detector = FaultDetector(model_dir=MODEL_DIR, window_size=None)

    assert detector.window_size == 150
    assert len(detector.columns) == 31


def test_fault_detector_predict_records_batch_matches_window_schedule_and_predict_contract():
    detector = FaultDetector(model_dir=MODEL_DIR)
    records = [
        {
            "actuator_ctrl_1": 0.0,
            "actuator_ctrl_2": 0.0,
            "actuator_ctrl_3": 0.0,
            "actuator_ctrl_4": 0.0,
            "actuator_out_1": 1200.0,
            "actuator_out_2": 1200.0,
            "actuator_out_3": 1200.0,
            "actuator_out_4": 1200.0,
            "gyro_x": 0.0,
            "gyro_y": 0.0,
            "gyro_z": 0.0,
            "accel_x": 0.0,
            "accel_y": 0.0,
            "accel_z": -9.8,
            "baro_alt": 10.0,
            "baro_temp": 25.0,
            "baro_pressure": 101325.0,
            "att_q_w": 1.0,
            "att_q_x": 0.0,
            "att_q_y": 0.0,
            "att_q_z": 0.0,
            "vel_x": 0.0,
            "vel_y": 0.0,
            "vel_z": 0.0,
            "mag_x": 0.0,
            "mag_y": 0.0,
            "mag_z": 0.0,
        }
        for _ in range(170)
    ]

    assert detector.predict(records[:149]) is None

    outputs = detector.predict_records_batch(records, every_n=10)

    assert [end for end, _ in outputs] == [150, 160, 170]
    for _, pred in outputs:
        assert set(pred.keys()) == {"label", "confidence", "fault_prob", "raw_logits"}
        assert pred["label"] in {"normal", "fault"}
        assert 0.0 <= pred["confidence"] <= 1.0
        assert 0.0 <= pred["fault_prob"] <= 1.0
        assert len(pred["raw_logits"]) == 2

    single = detector.predict(records[:150])
    assert single is not None
    assert set(single.keys()) == {"label", "confidence", "fault_prob", "raw_logits"}


def test_fault_detector_can_zero_out_excluded_features_in_feature_matrix():
    detector = FaultDetector(
        model_dir=MODEL_DIR,
        exclude_features=["actuator_out_1", "baro_temp", "mag_x"],
    )
    records = [
        {
            "actuator_ctrl_1": 0.0,
            "actuator_ctrl_2": 0.0,
            "actuator_ctrl_3": 0.0,
            "actuator_ctrl_4": 0.0,
            "actuator_out_1": 1200.0,
            "actuator_out_2": 1201.0,
            "actuator_out_3": 1202.0,
            "actuator_out_4": 1203.0,
            "gyro_x": 0.1,
            "gyro_y": 0.2,
            "gyro_z": 0.3,
            "accel_x": 1.0,
            "accel_y": 2.0,
            "accel_z": -9.8,
            "baro_alt": 10.0,
            "baro_temp": 25.0,
            "baro_pressure": 101325.0,
            "att_q_w": 1.0,
            "att_q_x": 0.0,
            "att_q_y": 0.0,
            "att_q_z": 0.0,
            "vel_x": 0.4,
            "vel_y": 0.5,
            "vel_z": 0.6,
            "mag_x": 0.7,
            "mag_y": 0.8,
            "mag_z": 0.9,
        }
    ]

    data = detector._build_feature_matrix(records)
    idx_out_1 = detector.columns.index("actuator_out_1")
    idx_baro_temp = detector.columns.index("baro_temp")
    idx_mag_x = detector.columns.index("mag_x")
    idx_out_2 = detector.columns.index("actuator_out_2")

    assert data[0, idx_out_1] == 0.0
    assert data[0, idx_baro_temp] == 0.0
    assert data[0, idx_mag_x] == 0.0
    assert data[0, idx_out_2] == 1201.0
