from collections import deque
from dataclasses import asdict
import threading
import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from scripts.eval import eval_telemetry_live
from uav_fault.online.detection_policy import OnlineDetectionPolicy, input_contract_check_due
from uav_fault.online import event_aggregator
from uav_fault.online.event_aggregator import event_timing
from uav_fault.online import monitoring_session, realtime_detection_service
from uav_fault.online.fault_detector import FaultDetector
from uav_fault.telemetry.mec_service import MecTelemetryService, _event_payload


@pytest.mark.parametrize('values', [
    {'inference_every': 0}, {'inference_every': True}, {'inference_every': 1.5},
    {'k_on': 0}, {'k_off': -1}, {'onset_th': float('nan')},
    {'onset_th': 1.1}, {'clear_th': -0.1}, {'clear_th': 0.7},
    {'min_event_s': -1}, {'merge_gap_s': float('inf')},
])
def test_invalid_policy_is_rejected(values):
    with pytest.raises(ValueError):
        OnlineDetectionPolicy(**values)


@pytest.mark.parametrize('every', [1, 5, 7, 10])
def test_stream_and_batch_window_phase_match(every):
    policy = OnlineDetectionPolicy(inference_every=every)
    expected = [i for i in range(1, 182) if i >= 150 and i % every == 0]
    actual = [i for i in range(1, 182) if policy.should_infer(
        frame_idx=i, window_length=min(i, 150), window_size=150,
    )]
    assert actual == list(policy.window_ends(181, 150)) == expected
    detector = FaultDetector.__new__(FaultDetector)
    detector.window_size = 150
    np.testing.assert_array_equal(detector._window_end_indices(181, every), expected)
    assert list(policy.window_ends(149, 150)) == []
    assert not policy.should_infer(frame_idx=0, window_length=150, window_size=150)


def test_cli_and_mec_keep_their_existing_default_intervals(monkeypatch):
    monkeypatch.setattr(sys, 'argv', ['realtime_detection_service', '--model', 'unused'])
    assert realtime_detection_service.parse_args().every == 1
    mec = MecTelemetryService(
        listen_host='127.0.0.1', listen_port=0, forward_host='127.0.0.1',
        forward_port=0, model_dir='unused',
    )
    assert mec.policy.inference_every == 5


def test_event_start_still_means_candidate_start_not_delivery_time():
    agg = OnlineDetectionPolicy().make_aggregator()
    deliveries = []
    for i in range(11):
        for event in agg.update(t=i / 5, fault_prob=0.9 if i <= 4 else 0.1):
            deliveries.append((event.start_t, event.end_t, i / 5))
    assert deliveries == [(0.0, 0.8, 1.6)]


def test_input_contract_cadence_preserves_warmup_first_check_and_force():
    def due(frame, length=150, previous=True, force=False):
        return input_contract_check_due(
            frame_idx=frame, window_length=length, min_window_size=150,
            every_frames=50, has_previous=previous, force=force,
        )

    assert not due(149, length=149, previous=False, force=True)
    assert due(151, previous=False)
    assert not due(151)
    assert due(200)
    assert due(151, force=True)
    assert input_contract_check_due(
        frame_idx=151, window_length=150, min_window_size=150, every_frames=1,
        has_previous=True,
    )


@pytest.mark.parametrize('every', [0, -1, True, 1.5])
def test_input_contract_cadence_rejects_invalid_intervals(every):
    with pytest.raises(ValueError):
        input_contract_check_due(
            frame_idx=150, window_length=150, min_window_size=150, every_frames=every,
        )


def test_event_timing_distinguishes_confirmation_from_output(monkeypatch):
    ticks = iter([1000.0, 1001.0])
    monkeypatch.setattr(event_aggregator, 'time', lambda: next(ticks))
    agg = OnlineDetectionPolicy().make_aggregator()
    events = []
    for i in range(11):
        events.extend(agg.update(t=i / 5, fault_prob=0.9 if i <= 4 else 0.1))
    assert len(events) == 1
    assert (events[0].start_t, events[0].end_t) == (0.0, 0.8)
    assert event_timing(events[0]) == {
        'confirmed_t': 0.4, 'emitted_t': 1.6,
        'confirmed_at_unix_s': 1000.0, 'emitted_at_unix_s': 1001.0,
        'emission_reason': 'update',
    }


def test_flush_and_merge_keep_the_first_confirmation_time(monkeypatch):
    ticks = iter([1000.0, 1010.0])
    monkeypatch.setattr(event_aggregator, 'time', lambda: next(ticks))
    agg = OnlineDetectionPolicy(merge_gap_s=2.0).make_aggregator()
    for i in range(11):
        assert agg.update(t=i / 5, fault_prob=0.1 if 4 <= i <= 6 else 0.9) == []
    events = agg.flush()
    assert len(events) == 1
    assert (events[0].start_t, events[0].end_t) == (0.0, 2.0)
    assert event_timing(events[0]) == {
        'confirmed_t': 0.4, 'emitted_t': 2.0,
        'confirmed_at_unix_s': 1000.0, 'emitted_at_unix_s': 1010.0,
        'emission_reason': 'flush',
    }
    assert agg.flush() == []


def test_short_events_are_still_filtered_out():
    agg = OnlineDetectionPolicy(min_event_s=5.0).make_aggregator()
    for i in range(4):
        assert agg.update(t=i / 5, fault_prob=0.9) == []
    assert agg.flush() == []


def test_legacy_and_invalid_event_timing_is_not_fabricated():
    assert event_timing(SimpleNamespace()) == {}
    assert event_timing(SimpleNamespace(meta={'timing': []})) == {}
    event = SimpleNamespace(meta={'timing': {
        'confirmed_t': '0.4', 'emitted_t': float('nan'),
        'confirmed_at_unix_s': True, 'emitted_at_unix_s': {'bad': 'value'},
        'emission_reason': 'invented', 'unknown': 42,
    }})
    assert event_timing(event) == {'confirmed_t': 0.4}


class _Detector:
    def __init__(self, model_dir='', device='cpu', window_size=150):
        self.window_size = window_size
        self.seen = []

    def predict(self, window):
        t = window[-1]['time_s']
        self.seen.append(t)
        p = 0.9 if 3.6 <= t <= 6.8 else 0.1
        return {'label': 'fault' if p > 0.5 else 'normal', 'confidence': 0.9, 'fault_prob': p}


def _frames():
    return [dict(time_s=i / 50, armed=1.0, motor1=1200.0, motor2=1210.0,
                 motor3=1190.0, motor4=1200.0, accel_z=-9.8, vx=0.2,
                 baro_alt_meter=1.0, roll=0.0, pitch=0.0, yaw=0.0)
            for i in range(520)]


def _stream_timing(timing):
    # Different processes/runs have different host clocks; only stream-clock
    # metadata can be compared for equality across replay and live transports.
    return {k: v for k, v in timing.items() if not k.endswith('_at_unix_s')}


def _event_values(events):
    values = []
    for event in events:
        value = asdict(event)
        value.pop('start_frame_idx')
        value.pop('end_frame_idx')
        value['meta']['timing'] = _stream_timing(event_timing(event))
        values.append(value)
    return values


def test_cli_mec_dashboard_and_replay_match_without_queue_drops(tmp_path, monkeypatch):
    from uav_dashboard_api.services.telemetry_manager import TelemetryProcessManager

    monkeypatch.setattr(monitoring_session, '_load_fault_detector', lambda: _Detector)
    frames = _frames()
    policy = OnlineDetectionPolicy(inference_every=7, onset_th=0.75, clear_th=0.35)

    class Receiver:
        def __init__(self, **kwargs):
            self.callbacks = []

    monkeypatch.setattr(realtime_detection_service, 'TelemetryReceiver', Receiver)
    cli = realtime_detection_service.RealtimeDetectionService(
        'unused', csv_path=str(tmp_path / 'unused.csv'), policy=policy,
    )
    for frame in frames:
        cli._on_frame(dict(frame), deque())
    cli_events = cli.fault_events + list(cli.monitoring_session.stop_detection().new_events)
    assert cli_events

    replay_detector = _Detector()
    canonical = eval_telemetry_live._records_for_backend(
        pd.DataFrame(frames), 'replay',
    )
    points = eval_telemetry_live._replay_points_legacy_gated(
        replay_detector, canonical, every_n=policy.inference_every,
    )
    replay_agg = policy.make_aggregator()
    replay_events = []
    for point in points:
        replay_events.extend(replay_agg.update(
            t=point['time_s'], fault_prob=point['fault_prob'], frame_idx=point['frame_idx'],
        ))
    replay_events.extend(replay_agg.flush())

    manager = TelemetryProcessManager(app=SimpleNamespace(config={'ONLINE_INFERENCE_STRIDE': 7}))
    manager._backend_name = 'replay'
    manager._active_model = {'name': 'dummy', 'task': 'online_binary'}
    manager._detector = monitoring_session.OnlineDetectionRunner('unused', policy=policy)
    manager._detecting = True
    manager._monitoring_session.start_detection(manager._detector)
    dashboard_events = []
    monkeypatch.setattr(manager, '_record_fault_event', dashboard_events.append)
    window = deque(maxlen=150)
    for frame in frames:
        window.append(frame)
        manager.process_transport_frame(frame, window, source_backend='replay')
    dashboard_events.extend(manager._monitoring_session.stop_detection().new_events)

    mec = MecTelemetryService(
        listen_host='127.0.0.1', listen_port=0, forward_host='127.0.0.1',
        forward_port=0, model_dir='unused', policy=policy,
    )
    mec.detector = _Detector()
    completed = threading.Event()
    publish = mec._publish_result

    def observe_publish(result):
        publish(result)
        if result.get('model_status') == 'ready' and 'inference_ms' in result:
            completed.set()

    monkeypatch.setattr(mec, '_publish_result', observe_publish)
    mec_events = []
    last_seq = 0
    worker = threading.Thread(target=mec._inference_worker, daemon=True)
    worker.start()
    try:
        for index, frame in enumerate(frames, 1):
            completed.clear()
            mec._handle_frame(dict(frame))
            if not policy.should_infer(frame_idx=index, window_length=min(index, 150), window_size=150):
                continue
            assert completed.wait(timeout=3), 'MEC inference did not finish'
            result = mec._result_snapshot()
            if result['event_seq'] > last_seq:
                mec_events.extend(result['events'])
                last_seq = result['event_seq']
    finally:
        mec.stop_event.set()
        worker.join(timeout=3)
    assert not worker.is_alive()
    assert mec.inference_drops == 0
    assert mec.aggregator.onset_th == policy.onset_th
    assert mec.aggregator.clear_th == policy.clear_th
    mec_events.extend(_event_payload(e) for e in mec.aggregator.flush())

    assert cli.detector.seen == replay_detector.seen == manager._detector.detector.seen == mec.detector.seen
    assert _event_values(cli_events) == _event_values(replay_events) == _event_values(dashboard_events)
    # The MEC transport payload intentionally omits aggregator metadata.
    expected = [dict(
        {k: v for k, v in e.items() if k != 'meta'}, timing=e['meta']['timing'],
    ) for e in _event_values(cli_events)]
    actual = []
    for event in mec_events:
        assert event['timing']['emitted_at_unix_s'] > 0
        actual.append(dict(
            {k: v for k, v in event.items() if k not in ('start_frame_idx', 'end_frame_idx', 'timing')},
            timing=_stream_timing(event['timing']),
        ))
    assert actual == expected


def test_dashboard_cached_and_forced_input_checks_keep_their_cadence(monkeypatch):
    from uav_dashboard_api.services import telemetry_manager as module

    manager = module.TelemetryProcessManager()
    checked = []

    def assess(*args, **kwargs):
        checked.append(manager._frame_count)
        return {'detection_blocked': manager._frame_count == 200}

    monkeypatch.setattr(module, 'assess_canonical_input_contract', assess)
    for index, length, force in [(149, 149, False), (150, 150, False),
                                  (151, 150, False), (200, 150, False), (201, 150, True)]:
        manager._frame_count = index - 1
        manager._handle_canonical_frame(
            frame={}, canonical={'time_s': index / 50}, detect_window=[{}] * length,
            coverage={}, async_inference=not force, force_contract_assessment=force,
        )
        if index in (151, 200, 201):
            assert manager._input_contract_status['detection_blocked'] == (index == 200)
    assert checked == [150, 200, 201]


def test_mec_timing_survives_dashboard_deserialization(monkeypatch):
    from uav_dashboard_api.services.telemetry_manager import TelemetryProcessManager

    agg = OnlineDetectionPolicy().make_aggregator()
    events = []
    for i in range(11):
        events.extend(agg.update(t=i / 5, fault_prob=0.9 if i <= 4 else 0.1))
    expected = event_timing(events[0])
    manager = TelemetryProcessManager()
    manager._backend_name = 'rfly'
    manager._detecting = True
    manager._mec_status = manager._initial_mec_status(enabled=True, location='mec', status='running')
    received = []
    monkeypatch.setattr(manager, '_record_fault_event', received.append)
    frame = dict(time_s=3.0, accel_z=-9.8, mec_enabled=1, mec_inference_count=1,
                 mec_fault_prob=0.1, mec_label='normal', mec_confidence=0.9,
                 mec_event_seq=1, mec_events=[_event_payload(events[0])])
    manager.process_transport_frame(frame, [frame], source_backend='rfly')
    assert len(received) == 1
    saved = manager._event_to_dict(received[0])
    assert saved['event_timing'] == expected
    assert saved['start_time'] == 0.0
    assert saved['end_time'] == 0.8
