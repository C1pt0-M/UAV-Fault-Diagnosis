"""Minimal checks for retrospective replay; no performance claims from fixtures."""
import importlib.util
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts/research"))
sys.path.insert(0, str(ROOT / "scripts/data"))
_SPEC = importlib.util.spec_from_file_location("normal_replay_runner", ROOT / "scripts/research/run_normal_replay_v1.py")
runner = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(runner)


def event(start, end, confirmed=None, emitted=None):
    return {"start_t": start, "end_t": end, "meta": {"timing": {"confirmed_t": confirmed or start + 0.4, "emitted_t": emitted or end}}}


def test_interval_extraction_and_relevant_delay_edges():
    assert runner.intervals(np.arange(7) * 0.02, np.array([1, 1, 0, 1, 0, 1, 1], dtype=bool)) == [(0.0, 0.02), (0.06, 0.06), (0.1, 0.12)]
    result = runner.event_metrics([event(1, 4), event(8, 9), event(13, 14)], [(2, 5), (7, 10), (11, 12)])
    assert result["hit"]
    assert result["intervals"][0]["pre_onset_alarm"]
    assert result["intervals"][0]["event_start_delay_s"] is None
    assert result["intervals"][1]["event_start_delay_s"] == 1
    assert not result["intervals"][2]["hit"]
    assert result["intervals"][2]["event_start_delay_s"] is None  # later unrelated alarm is NOT a detection
    assert result["non_overlapping_alarm_events"] == 1
    assert result["pre_first_onset_events"] == 1
    assert not runner.event_metrics([], [(1, 2)])["hit"]
    assert runner.distribution([])["median"] is None


def test_full_replay_preserves_per_window_altitude_subtraction_and_stride():
    class Model(torch.nn.Module):
        def forward(self, x):
            # Every window should begin with zero relative altitude; a whole-log
            # subtraction would produce different probabilities after window 1.
            score = x[:, 0, 0]
            return torch.stack([score, -score], dim=1)

    matrix = np.arange(1000 * 26, dtype=np.float64).reshape(1000, 26)
    columns = ["baro_alt_relative"] + [str(i) for i in range(25)]
    ends, probabilities, sample_ends, _, _ = runner.predict_case(Model(), matrix, columns, torch.zeros(1, 26, 1), torch.ones(1, 26, 1), torch.device("cpu"), sampled=True)
    np.testing.assert_array_equal(ends, np.arange(150, 1001, 10))
    np.testing.assert_allclose(probabilities, 0.5)
    assert len(sample_ends) == 64
    assert sample_ends[0] == 150 and sample_ends[-1] == 1000


def test_aggregator_uses_existing_rules_and_separates_confirmation():
    module = runner.load_module("fixture_event_aggregator", ROOT / "packages/uav_fault/uav_fault/online/event_aggregator.py")
    times = np.arange(12) * 0.2
    events = runner.aggregate(module.FaultEventAggregator, times, np.ones(12) * 0.9, 0.6)
    assert len(events) == 1
    assert events[0]["start_t"] == 0
    assert events[0]["meta"]["timing"]["confirmed_t"] == 0.4
    assert events[0]["meta"]["timing"]["emission_reason"] == "flush"
    assert not runner.aggregate(module.FaultEventAggregator, times, np.zeros(12), 0.6)
