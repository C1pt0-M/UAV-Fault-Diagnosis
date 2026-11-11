"""Differential checks against the frozen research implementation, without models."""
from __future__ import annotations

import ast
import hashlib
import itertools
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from uav_fault.online import event_aggregator
from uav_fault.online.detection_policy import OnlineDetectionPolicy
from uav_fault.online.event_aggregator import event_timing
from uav_fault.online.event_replay import aggregate_scores
from uav_fault.online.monitoring_session import OnlineMonitoringSession


ROOT = Path(__file__).resolve().parents[1]
FROZEN_SOURCE = "scripts/research/replay_m1_continuous.py"
PROFILE_NAMES = ["default", "gbt_source", "a1", "no_merge", "single_point", "equal_thresholds"]
CASE_NAMES = [
    "empty", "normal", "unavailable", "short_confirmation", "open_at_end",
    "cleared", "merged", "separated", "candidate_gap", "gap_then_tail",
    "two_segments", "threshold_edges", "irregular_clock",
]


@pytest.fixture(scope="module")
def frozen_aggregate():
    """Extract the pinned reference only in tests; never run neural imports."""
    source = ROOT / FROZEN_SOURCE
    protocol = json.loads((ROOT / "configs/experiments/M3_preflight_v1.json").read_text(encoding="utf-8"))
    assert hashlib.sha256(source.read_bytes()).hexdigest() == protocol["code_identities"][FROZEN_SOURCE]
    tree = ast.parse(source.read_text(encoding="utf-8"))
    function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "aggregate_scores")
    namespace = {
        "np": np, "SimpleNamespace": SimpleNamespace,
        "OnlineDetectionPolicy": OnlineDetectionPolicy,
        "OnlineMonitoringSession": OnlineMonitoringSession,
        "event_timing": event_timing,
    }
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(source), "exec"), namespace)
    return namespace["aggregate_scores"]


def policy_for(name):
    if name == "default":
        return None
    if name == "gbt_source":
        # Keep this synthetic policy independent of locally installed model
        # assets, which are outside the architecture change.
        return OnlineDetectionPolicy(onset_th=0.95)
    if name == "a1":
        return OnlineDetectionPolicy(onset_th=0.85)
    if name == "no_merge":
        return OnlineDetectionPolicy(merge_gap_s=0)
    if name == "single_point":
        return OnlineDetectionPolicy(
            onset_th=0.5, clear_th=0.3, k_on=1, k_off=1,
            min_event_s=0, merge_gap_s=0, inference_every=7,
        )
    return OnlineDetectionPolicy(onset_th=0.5, clear_th=0.5, k_on=2, k_off=2)


def score_case(name, policy):
    settings = policy or OnlineDetectionPolicy()
    high, low, gap = 1.0, 0.0, np.nan
    sequences = {
        "empty": [],
        "normal": [low] * 180,
        "unavailable": [gap, np.inf, -np.inf] * 20,
        "short_confirmation": [high] * 3 + [low] * 100,
        "open_at_end": [high] * 101,
        "cleared": [high] * 40 + [low] * 100,
        "merged": [high] * 40 + [low] * 3 + [high] * 40 + [low] * 100,
        "separated": [high] * 40 + [low] * 80 + [high] * 40 + [low] * 100,
        "candidate_gap": [high] * 2 + [gap] + [high] * 2 + [low] * 100,
        "gap_then_tail": [high] * 50 + [gap] * 100,
        "two_segments": [high] * 50 + [gap] * 4 + [high] * 60,
        "threshold_edges": (
            [np.nextafter(settings.onset_th, -np.inf)] * 5
            + [settings.onset_th] * 40 + [settings.clear_th] * 20
            + [np.nextafter(settings.clear_th, -np.inf)] * 100
        ),
        "irregular_clock": [high] * 12 + [low] * 5 + [high] * 10 + [gap] + [high] * 20,
    }
    scores = np.asarray(sequences[name], dtype=float)
    times = np.arange(len(scores), dtype=float) * 0.02
    if name == "irregular_clock":
        times = np.cumsum(np.resize([0.02, 0.1, 0.25, 1.0, 0.03], len(scores)))
    return times, scores


def fixed_clock_run(monkeypatch, function, times, probabilities, policy):
    ticks = itertools.count()
    monkeypatch.setattr(event_aggregator, "time", lambda: 1700000000.0 + next(ticks) / 1000)
    return function(times, probabilities, policy=policy)


def assert_equivalent(monkeypatch, reference, times, probabilities, policy):
    original_times, original_scores = times.copy(), probabilities.copy()
    expected = fixed_clock_run(monkeypatch, reference, times, probabilities, policy)
    actual = fixed_clock_run(monkeypatch, aggregate_scores, times, probabilities, policy)
    # Compare every field, including host-clock metadata and exact JSON number
    # representations. No tolerance, dropped timing fields or metadata masks.
    assert actual == expected
    assert json.dumps(actual, sort_keys=True, allow_nan=False) == json.dumps(
        expected, sort_keys=True, allow_nan=False,
    )
    np.testing.assert_array_equal(times, original_times)
    np.testing.assert_array_equal(probabilities, original_scores)


@pytest.mark.parametrize("profile", PROFILE_NAMES)
@pytest.mark.parametrize("case", CASE_NAMES)
def test_frozen_replay_matches_every_output_field(frozen_aggregate, monkeypatch, profile, case):
    policy = policy_for(profile)
    times, scores = score_case(case, policy)
    assert_equivalent(monkeypatch, frozen_aggregate, times, scores, policy)


@pytest.mark.parametrize("profile", PROFILE_NAMES)
def test_seeded_variable_clock_and_gap_traces_match(frozen_aggregate, monkeypatch, profile):
    rng = np.random.default_rng(20261111)
    policy = policy_for(profile)
    for _ in range(10):
        count = int(rng.integers(1, 400))
        times = 100 + np.cumsum(rng.choice([0.02, 0.04, 0.1, 0.8], count))
        scores = rng.choice([0.0, 0.4, 0.6, 0.85, 0.95, 1.0, np.nan], count)
        assert_equivalent(monkeypatch, frozen_aggregate, times, scores, policy)


@pytest.mark.parametrize("times,scores", [
    ([0.0, 0.02], [1.0]),
    ([0.0, 0.0], [1.0, 1.0]),
    ([0.02, 0.0], [1.0, 1.0]),
])
def test_existing_input_errors_match(frozen_aggregate, times, scores):
    for function in (frozen_aggregate, aggregate_scores):
        with pytest.raises(ValueError, match="^Ordered inference timestamps required$"):
            function(times, scores)


def test_inference_interval_does_not_resample_an_existing_score_sequence():
    result = aggregate_scores(
        np.arange(51) * 0.02, np.ones(51),
        policy=OnlineDetectionPolicy(inference_every=7),
    )
    assert result["confirmations"] == [{"candidate_start_t": 0.0, "confirmed_t": 0.04}]
    assert len(result["finalized_events"]) == 1
    assert result["policy"]["inference_every"] == 7


def test_gap_observation_time_and_aggregator_time_remain_distinct():
    result = aggregate_scores(np.arange(150) * 0.02, np.r_[np.ones(50), np.full(100, np.nan)])
    event, = result["finalized_events"]
    assert event["end_t"] == event["deployment_timing"]["emitted_t"] == 0.98
    assert event["emitted_t"] == 1.0
    assert event["emission_reason"] == "invalid_gap_flush"


def test_confirmation_survives_final_minimum_duration_filter():
    result = aggregate_scores(np.arange(103) * 0.02, np.r_[np.ones(3), np.zeros(100)])
    assert result["confirmations"] == [{"candidate_start_t": 0.0, "confirmed_t": 0.04}]
    assert result["finalized_events"] == []


def test_import_and_replay_without_neural_runtime_or_research_scripts():
    code = r'''
import importlib.abc
import sys
sys.path.insert(0, sys.argv[1])
class NoModels(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if (fullname.split('.')[0] in {'torch', 'tensorflow', 'jax', 'sklearn'}
                or fullname.startswith(('uav_fault.models', 'uav_fault.online.fault_detector',
                                        'scripts.research', 'replay_m1_continuous'))):
            raise AssertionError('Unexpected model/research import: ' + fullname)
sys.meta_path.insert(0, NoModels())
import numpy as np
from uav_fault.online.event_replay import aggregate_scores
from uav_fault.online.detection_policy import OnlineDetectionPolicy
result = aggregate_scores(np.arange(101) * .02, np.ones(101),
                          policy=OnlineDetectionPolicy(onset_th=.95))
assert len(result['finalized_events']) == 1
assert result['policy']['onset_th'] == .95
'''
    completed = subprocess.run(
        [sys.executable, "-I", "-B", "-c", code, str(ROOT / "packages/uav_fault")],
        capture_output=True, text=True, timeout=30,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
