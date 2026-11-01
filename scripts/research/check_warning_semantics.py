"""Synthetic verification of current warning timing, not a performance test.

Run: python scripts/research/check_warning_semantics.py
No model loading, training, network, real flight data or runtime file edits.
"""
from pathlib import Path
import hashlib
import json
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'packages/uav_fault'))
from uav_fault.online.event_aggregator import FaultEventAggregator, event_timing
from uav_fault.online.monitoring_session import OnlineMonitoringSession
from uav_fault.online.risk_evaluator import RiskEvaluator


class SyntheticRunner:
    def __init__(self):
        self.aggregator = FaultEventAggregator()

    def flush(self):
        return self.aggregator.flush()


def run_case(probabilities, stop):
    runner = SyntheticRunner()
    session = OnlineMonitoringSession()
    session.start_detection(runner, model_summary={'source': 'synthetic probabilities, no model'})
    trace = []
    emitted = []
    for i, probability in enumerate(probabilities):
        t = i / 10
        events = runner.aggregator.update(t=t, fault_prob=probability, frame_idx=i * 5)
        update = session.ingest_external(result={'fault_prob': probability, 'label': 'fault', 'confidence': probability},
            new_events=events, frame_idx=i * 5, time_s=t, onset_th=runner.aggregator.onset_th)
        trace.append({'t': t, 'returned_events': len(events), 'preview_started_at': update.snapshot.preview_started_at,
            'warning_stage': update.snapshot.warning_stage, 'lead_time_s': update.snapshot.last_preview_lead_time_s})
        emitted.extend(events)
    if stop:
        emitted.extend(session.stop_detection().new_events)
    return {'trace': trace, 'events': [{'start_t': e.start_t, 'end_t': e.end_t,
        'confirmed_t': event_timing(e).get('confirmed_t'), 'emitted_t': event_timing(e).get('emitted_t'),
        'emission_reason': event_timing(e).get('emission_reason'), 'lead_time_s': getattr(e, 'lead_time_s', None)} for e in emitted]}


def main():
    flushed = run_case([0.8] * 11, stop=True)
    closed = run_case([0.8] * 9 + [0.1] * 7, stop=False)
    assert flushed['trace'][2]['returned_events'] == 0
    assert len(flushed['events']) == 1
    first = flushed['events'][0]
    assert first['confirmed_t'] == 0.2 and first['emitted_t'] == 1.0
    assert first['lead_time_s'] == 1.0 and first['emission_reason'] == 'flush'
    assert len(closed['events']) == 1
    second = closed['events'][0]
    assert second['confirmed_t'] == 0.2 and second['emitted_t'] > second['confirmed_t']
    assert second['lead_time_s'] is None
    risk = RiskEvaluator().update(fault_prob=0.8, has_preview=False, has_confirmed=False)
    assert risk['warning_stage'] == 'critical' and '稳定故障事件' in risk['warning_reason']
    files = ['packages/uav_fault/uav_fault/online/risk_evaluator.py',
        'packages/uav_fault/uav_fault/online/decision_engine.py',
        'packages/uav_fault/uav_fault/online/event_aggregator.py',
        'packages/uav_fault/uav_fault/online/monitoring_session.py']
    print(json.dumps({'status': 'PASS_CURRENT_WARNING_SEMANTICS_REPRODUCED',
        'scope': 'Synthetic scores only; checks current behavior, not desired behavior or prognostic validity.',
        'sustained_then_flush': flushed, 'high_then_clear': closed,
        'critical_without_confirmed_event': risk,
        'source_sha256': {p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in files},
        'no_training_model_inference_or_real_data': True}, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
