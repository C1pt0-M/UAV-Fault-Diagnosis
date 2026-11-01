import sys
import types
from pathlib import Path

import numpy as np
import pandas as pd


from scripts.eval import eval_telemetry_live


def test_replay_summary_keeps_legacy_event_times_and_adds_output_timing(tmp_path, monkeypatch):
    csv_dir = tmp_path / 'csv'
    model_dir = tmp_path / 'model'
    csv_dir.mkdir()
    model_dir.mkdir()
    output = tmp_path / 'summary.csv'
    pd.DataFrame({
        'time_s': [i / 5 for i in range(11)],
        'actuator_out_1': 1200.0, 'actuator_out_2': 1200.0,
        'actuator_out_3': 1200.0, 'actuator_out_4': 1200.0,
        'accel_z': -9.8, 'vel_x': 0.2, 'baro_alt': 1.0,
    }).to_csv(csv_dir / 'sample.csv', index=False)

    class Detector:
        def __init__(self, **kwargs):
            self.window_size = 150
            self.config = {'model': 'dummy'}

    fake = types.ModuleType('uav_fault.online.fault_detector')
    fake.FaultDetector = Detector
    monkeypatch.setitem(sys.modules, 'uav_fault.online.fault_detector', fake)
    points = [dict(time_s=i / 5, frame_idx=i, fault_prob=0.9 if i <= 4 else 0.1,
                   label='fault' if i <= 4 else 'normal', confidence=0.9)
              for i in range(11)]
    monkeypatch.setattr(eval_telemetry_live, '_replay_points_fast', lambda *a, **kw: points)
    monkeypatch.setattr(sys, 'argv', [
        'eval_telemetry_live', '--csv-dir', str(csv_dir), '--model-dir', str(model_dir),
        '--out', str(output),
    ])
    assert eval_telemetry_live.main() == 0
    row = pd.read_csv(output).iloc[0]
    assert row['n_events'] == 1
    assert row['best_event_start_t'] == 0.0
    assert row['best_event_end_t'] == 0.8
    assert row['best_event_confirmed_t'] == 0.4
    assert row['best_event_emitted_t'] == 1.6
    assert row['best_event_confirmed_at_unix_s'] > 0
    assert row['best_event_emitted_at_unix_s'] > 0
    assert row['best_event_emission_reason'] == 'update'


class DummyEvent:
    def __init__(self, start_t: float, end_t: float):
        self.start_t = start_t
        self.end_t = end_t


def test_gt_event_metrics_detects_hit_and_latency():
    df = pd.DataFrame(
        {
            "time_s": np.arange(0.0, 5.0, 1.0),
            "is_fault_active": [0, 0, 1, 1, 0],
        }
    )
    events = [DummyEvent(2.5, 3.0)]

    metrics = eval_telemetry_live._gt_event_metrics(df, events)

    assert metrics["gt_has_labels"] == 1
    assert metrics["gt_fault"] == 1
    assert metrics["gt_event_count"] == 1
    assert metrics["hit"] == 1
    assert metrics["miss"] == 0
    assert metrics["false_alarm"] == 0
    assert metrics["gt_onset_t"] == 2.0
    assert metrics["first_detection_time"] == 2.5
    assert metrics["detection_latency_s"] == 0.5


def test_gt_event_metrics_detects_false_alarm_for_normal_file():
    df = pd.DataFrame(
        {
            "time_s": np.arange(0.0, 4.0, 1.0),
            "is_fault_active": [0, 0, 0, 0],
        }
    )
    events = [DummyEvent(1.0, 1.5)]

    metrics = eval_telemetry_live._gt_event_metrics(df, events)

    assert metrics["gt_fault"] == 0
    assert metrics["hit"] == 0
    assert metrics["miss"] == 0
    assert metrics["false_alarm"] == 1
    assert metrics["detection_latency_s"] is None


def test_gt_event_metrics_marks_unlabeled_input():
    df = pd.DataFrame({"time_s": [0.0, 1.0, 2.0]})

    metrics = eval_telemetry_live._gt_event_metrics(df, [])

    assert metrics["gt_has_labels"] == 0
    assert metrics["gt_fault"] is None
    assert metrics["hit"] is None
    assert metrics["false_alarm"] is None


def test_records_for_backend_px4_uses_px4_adapter_contract():
    df = pd.DataFrame(
        [
            {
                "time_s": 0.0,
                "seq": 1,
                "motor1": 0.1,
                "motor2": 0.2,
                "motor3": 0.3,
                "motor4": 0.4,
                "accel_x": 1.0,
                "accel_y": 2.0,
                "accel_z": 2.0,
                "roll": 0.0,
                "pitch": 0.0,
                "yaw": 0.0,
            }
        ]
    )

    records = eval_telemetry_live._records_for_backend(df, "px4")

    assert len(records) == 1
    assert records[0]["actuator_out_1"] == 0.1
    assert records[0]["acc_norm"] >= 0.0
    assert "motor1" not in records[0]


def test_records_for_backend_replay_keeps_rfly_canonicalization_path():
    df = pd.DataFrame(
        [
            {
                "time_s": 0.0,
                "_actuator_outputs_0_output[0]": 0.25,
                "roll": 1.0,
                "pitch": 2.0,
                "yaw": 3.0,
            }
        ]
    )

    records = eval_telemetry_live._records_for_backend(df, "replay")

    assert len(records) == 1
    assert records[0]["actuator_out_1"] == 0.25
    assert "acc_norm" in records[0]


def test_input_contract_columns_flags_zero_critical_signals():
    records = [
        {
            "actuator_out_1": 0.0,
            "actuator_out_2": 0.0,
            "actuator_out_3": 0.0,
            "actuator_out_4": 0.0,
            "accel_z": 0.0,
            "vel_x": 0.0,
            "vel_y": 0.0,
            "vel_z": 0.0,
            "baro_alt": 0.0,
        }
        for _ in range(150)
    ]

    columns = eval_telemetry_live._input_contract_columns(records)

    assert columns["input_contract_status"] == "degraded"
    assert columns["input_contract_reason"] == "untrusted_zero_signals"
    assert columns["input_contract_detection_blocked"] == 1
    assert "actuator_outputs" in columns["input_contract_zero_signal_groups"]
    assert "imu_accel_z" in columns["input_contract_zero_signal_groups"]


def test_input_contract_columns_allows_valid_actuator_and_imu_signals():
    records = [
        {
            "actuator_out_1": 1300.0,
            "actuator_out_2": 1310.0,
            "actuator_out_3": 1290.0,
            "actuator_out_4": 1305.0,
            "accel_z": -9.8,
            "vel_x": 0.1,
            "vel_y": 0.0,
            "vel_z": -0.2,
            "baro_alt": 12.0,
        }
        for _ in range(150)
    ]

    columns = eval_telemetry_live._input_contract_columns(records)

    assert columns["input_contract_detection_blocked"] == 0
    assert columns["input_contract_zero_signal_groups"] == ""


def test_replay_points_fast_skips_windows_blocked_by_input_contract():
    class _DummyDetector:
        window_size = 2

        def predict_records_batch(self, records, every_n, batch_size):
            return [
                (2, {"fault_prob": 1.0, "label": "fault", "confidence": 1.0}),
            ]

    records = [
        {
            "time_s": 0.0,
            "actuator_out_1": 1000.0,
            "actuator_out_2": 1000.0,
            "actuator_out_3": 1000.0,
            "actuator_out_4": 1000.0,
            "accel_z": -9.8,
            "vel_x": 0.0,
            "vel_y": 0.0,
            "vel_z": 0.0,
            "baro_alt": 0.0,
        },
        {
            "time_s": 0.02,
            "actuator_out_1": 1000.0,
            "actuator_out_2": 1000.0,
            "actuator_out_3": 1000.0,
            "actuator_out_4": 1000.0,
            "accel_z": -9.8,
            "vel_x": 0.0,
            "vel_y": 0.0,
            "vel_z": 0.0,
            "baro_alt": 0.0,
        },
    ]

    assert eval_telemetry_live._replay_points_fast(
        _DummyDetector(),
        records,
        every_n=1,
        batch_size=16,
    ) == []
    assert len(eval_telemetry_live._replay_points_fast(
        _DummyDetector(),
        records,
        every_n=1,
        batch_size=16,
        respect_input_contract=False,
    )) == 1


def test_main_routes_px4_backend_into_records_for_backend_fast_mode(
    tmp_path, monkeypatch
):
    csv_dir = tmp_path / "csv"
    csv_dir.mkdir()
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    out_csv = tmp_path / "out.csv"

    pd.DataFrame(
        [{"time_s": 0.0, "motor1": 0.1}, {"time_s": 0.02, "motor1": 0.2}]
    ).to_csv(csv_dir / "sample.csv", index=False)

    fake_fault_detector = types.ModuleType("fault_detector")

    class _DummyDetector:
        def __init__(self, model_dir, device, window_size, exclude_features=None):
            self.window_size = 2
            self.config = {"model": "dummy"}

    fake_fault_detector.FaultDetector = _DummyDetector

    fake_event_aggregator = types.ModuleType("event_aggregator")

    class _DummyAggregator:
        def __init__(self, **kwargs):
            pass

        def update(self, **kwargs):
            return []

        def flush(self):
            return []

    fake_event_aggregator.FaultEventAggregator = _DummyAggregator

    monkeypatch.setitem(sys.modules, "uav_fault.online.fault_detector", fake_fault_detector)
    monkeypatch.setitem(sys.modules, "uav_fault.online.event_aggregator", fake_event_aggregator)

    captured = {"backend": None, "mode": None}

    def _fake_records_for_backend(df, backend):
        captured["backend"] = backend
        return [{"time_s": 0.0}, {"time_s": 0.02}]

    def _fake_replay_fast(detector, records, every_n, batch_size, **kwargs):
        captured["mode"] = "fast"
        return []

    monkeypatch.setattr(
        eval_telemetry_live, "_records_for_backend", _fake_records_for_backend
    )
    monkeypatch.setattr(eval_telemetry_live, "_replay_points_fast", _fake_replay_fast)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "eval_telemetry_live.py",
            "--csv-dir",
            str(csv_dir),
            "--model-dir",
            str(model_dir),
            "--backend",
            "px4",
            "--replay-mode",
            "fast",
            "--out",
            str(out_csv),
        ],
    )

    rc = eval_telemetry_live.main()

    assert rc == 0
    assert captured["backend"] == "px4"
    assert captured["mode"] == "fast"
    assert out_csv.exists()
    saved = pd.read_csv(out_csv)
    assert saved['policy_inference_every'].tolist() == [10]
    assert saved['policy_onset_th'].tolist() == [0.6]
    assert saved['respect_input_contract'].tolist() == [True]
    assert saved['input_contract_check_mode'].tolist() == ['per_inference']
    assert saved['input_contract_check_every_frames'].tolist() == [10]


def test_main_routes_replay_backend_into_records_for_backend_legacy_mode(
    tmp_path, monkeypatch
):
    csv_dir = tmp_path / "csv"
    csv_dir.mkdir()
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    out_csv = tmp_path / "out.csv"

    pd.DataFrame([{"time_s": 0.0, "roll": 1.0}, {"time_s": 0.02, "roll": 2.0}]).to_csv(
        csv_dir / "sample.csv", index=False
    )

    fake_fault_detector = types.ModuleType("fault_detector")

    class _DummyDetector:
        def __init__(self, model_dir, device, window_size, exclude_features=None):
            self.window_size = 2
            self.config = {"model": "dummy"}

    fake_fault_detector.FaultDetector = _DummyDetector

    fake_event_aggregator = types.ModuleType("event_aggregator")

    class _DummyAggregator:
        def __init__(self, **kwargs):
            pass

        def update(self, **kwargs):
            return []

        def flush(self):
            return []

    fake_event_aggregator.FaultEventAggregator = _DummyAggregator

    monkeypatch.setitem(sys.modules, "uav_fault.online.fault_detector", fake_fault_detector)
    monkeypatch.setitem(sys.modules, "uav_fault.online.event_aggregator", fake_event_aggregator)

    captured = {"backend": None, "mode": None}

    def _fake_records_for_backend(df, backend):
        captured["backend"] = backend
        return [{"time_s": 0.0}, {"time_s": 0.02}]

    def _fake_replay_legacy(detector, records, every_n):
        captured["mode"] = "legacy"
        return []

    monkeypatch.setattr(
        eval_telemetry_live, "_records_for_backend", _fake_records_for_backend
    )
    monkeypatch.setattr(
        eval_telemetry_live, "_replay_points_legacy", _fake_replay_legacy
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "eval_telemetry_live.py",
            "--csv-dir",
            str(csv_dir),
            "--model-dir",
            str(model_dir),
            "--backend",
            "replay",
            "--replay-mode",
            "legacy",
            "--ignore-input-contract",
            "--out",
            str(out_csv),
        ],
    )

    rc = eval_telemetry_live.main()

    assert rc == 0
    assert captured["backend"] == "replay"
    assert captured["mode"] == "legacy"
    assert out_csv.exists()
    saved = pd.read_csv(out_csv)
    assert saved['respect_input_contract'].tolist() == [False]
    assert saved['input_contract_check_mode'].tolist() == ['disabled']
    assert saved['input_contract_check_every_frames'].isna().all()


def test_main_passes_exclude_features_to_fault_detector(tmp_path, monkeypatch):
    csv_dir = tmp_path / "csv"
    csv_dir.mkdir()
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    out_csv = tmp_path / "out.csv"

    pd.DataFrame(
        [{"time_s": 0.0, "motor1": 0.1}, {"time_s": 0.02, "motor1": 0.2}]
    ).to_csv(csv_dir / "sample.csv", index=False)

    fake_fault_detector = types.ModuleType("fault_detector")
    captured = {"exclude_features": None}

    class _DummyDetector:
        def __init__(self, model_dir, device, window_size, exclude_features=None):
            captured["exclude_features"] = exclude_features
            self.window_size = 2
            self.config = {"model": "dummy"}

    fake_fault_detector.FaultDetector = _DummyDetector

    fake_event_aggregator = types.ModuleType("event_aggregator")

    class _DummyAggregator:
        def __init__(self, **kwargs):
            pass

        def update(self, **kwargs):
            return []

        def flush(self):
            return []

    fake_event_aggregator.FaultEventAggregator = _DummyAggregator

    monkeypatch.setitem(sys.modules, "uav_fault.online.fault_detector", fake_fault_detector)
    monkeypatch.setitem(sys.modules, "uav_fault.online.event_aggregator", fake_event_aggregator)
    monkeypatch.setattr(
        eval_telemetry_live,
        "_records_for_backend",
        lambda df, backend: [{"time_s": 0.0}, {"time_s": 0.02}],
    )
    monkeypatch.setattr(
        eval_telemetry_live,
        "_replay_points_fast",
        lambda detector, records, every_n, batch_size, **kwargs: [],
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "eval_telemetry_live.py",
            "--csv-dir",
            str(csv_dir),
            "--model-dir",
            str(model_dir),
            "--backend",
            "px4",
            "--replay-mode",
            "fast",
            "--exclude-features",
            "actuator_out_1,baro_temp,mag_x",
            "--out",
            str(out_csv),
        ],
    )

    rc = eval_telemetry_live.main()

    assert rc == 0
    assert captured["exclude_features"] == ["actuator_out_1", "baro_temp", "mag_x"]
