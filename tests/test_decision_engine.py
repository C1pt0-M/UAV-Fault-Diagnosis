from uav_fault.online.decision_engine import TemporalDecisionEngine
from uav_fault.online.decision_models import DecisionSignal


def test_decision_engine_transitions_preview_to_confirmed():
    engine = TemporalDecisionEngine()

    preview_signal = DecisionSignal(
        ts=10.0,
        frame_idx=1,
        fault_prob=0.66,
        label="fault",
        confidence=0.66,
        risk_score=0.58,
        risk_trend="rising",
        warning_stage="warning",
    )
    preview_outcome = engine.ingest_signal(
        preview_signal,
        onset_th=0.6,
        online_model={"name": "online_detector_CNN"},
    )

    assert preview_outcome is not None
    assert preview_outcome.episode_id is not None
    assert preview_outcome.warning_stage == "warning"
    assert preview_outcome.lead_time_s is None
    assert engine.current_state().preview_alert is not None

    class _FakeEvent:
        start_t = 11.0
        end_t = 13.0
        max_fault_prob = 0.92
        mean_fault_prob = 0.86
        n_points = 6

    confirmed_outcome = engine.confirm_event(_FakeEvent(), ts=13.0)
    assert confirmed_outcome is not None
    assert confirmed_outcome.state == "CONFIRMED_EVENT"
    assert confirmed_outcome.episode_id == preview_outcome.episode_id
    assert confirmed_outcome.warning_stage == "critical"
    assert confirmed_outcome.lead_time_s == 3.0


def test_decision_engine_recommendations_follow_stage():
    engine = TemporalDecisionEngine()
    signal = DecisionSignal(
        ts=5.0,
        frame_idx=1,
        fault_prob=0.4,
        label="fault",
        confidence=0.4,
        risk_score=0.4,
        risk_trend="rising",
        warning_stage="watch",
    )

    outcome = engine.ingest_signal(signal, onset_th=0.6, online_model=None)

    assert outcome is not None
    assert outcome.recommendations
    assert outcome.recommendations[0]["priority"] == "P2"
    assert outcome.recommendations[0]["reason_codes"] == ["realtime_observe"]


def test_decision_engine_can_attach_offline_diagnosis_and_finalize_decision():
    engine = TemporalDecisionEngine()
    signal = DecisionSignal(
        ts=8.0,
        frame_idx=1,
        fault_prob=0.7,
        label="fault",
        confidence=0.7,
        risk_score=0.61,
        risk_trend="rising",
        warning_stage="warning",
    )
    engine.ingest_signal(
        signal, onset_th=0.6, online_model={"name": "online_detector_CNN"}
    )

    class _FakeEvent:
        start_t = 8.5
        end_t = 10.5
        max_fault_prob = 0.95
        mean_fault_prob = 0.88
        n_points = 5

    engine.confirm_event(_FakeEvent(), ts=10.5)
    outcome = engine.attach_offline_diagnosis(
        {
            "fault_label": "motor",
            "severity": {"level": "heavy", "score": 0.86},
            "recommendations": [{"title": "inspect motor chain immediately", "priority": "P0"}],
            "evidence": [{"metric": "motor_imbalance", "status": "used"}],
        }
    )

    assert outcome is not None
    assert outcome.state == "DECIDED"
    assert outcome.episode_id is not None
    assert outcome.fault_label == "motor"
    assert outcome.severity == "heavy"
    assert outcome.evidence_summary is not None
    assert outcome.evidence_summary["source"] == "offline"
    assert outcome.evidence_summary["fault_label"] == "motor"
    assert outcome.recommendations[0]["priority"] == "P0"
