"""Replay already-produced fault scores through the shared event policy.

This is the lightweight reuse path for new or versioned research callers.
The historical implementation in scripts/research/replay_m1_continuous.py
remains frozen. Importing this module does not load a detector or ML runtime.
"""
from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import numpy as np

from uav_fault.online.detection_policy import OnlineDetectionPolicy
from uav_fault.online.event_aggregator import event_timing
from uav_fault.online.monitoring_session import OnlineMonitoringSession


def aggregate_scores(
    times,
    probabilities,
    *,
    policy: OnlineDetectionPolicy | None = None,
) -> dict[str, Any]:
    """Return confirmations, finalized events and policy for a score sequence.

    Matching input arrays describe an ordered stream-clock timeline in seconds.
    Every supplied point is consumed; inference_every is recorded in the policy
    but does not resample scores that the caller has already produced.

    Nonfinite scores flush a preceding available segment and reset its event
    state. The first unavailable timestamp records gap emission; end-of-stream
    emission uses the final supplied timestamp. deployment_timing retains the
    aggregator's own stream and emitting-host times, which are distinct from
    that replay observation time. Historical validation and defaults are kept.
    """
    times = np.asarray(times, dtype=float)
    probabilities = np.asarray(probabilities, dtype=float)
    if times.shape != probabilities.shape or np.any(np.diff(times) <= 0):
        raise ValueError("Ordered inference timestamps required")

    policy = policy or OnlineDetectionPolicy()
    aggregator = policy.make_aggregator()
    session = OnlineMonitoringSession()
    runner = SimpleNamespace(aggregator=aggregator, flush=aggregator.flush)
    session.start_detection(runner)
    events = []
    confirmations = []
    previous_available = False

    def record(event, reason, emitted_t):
        timing = event_timing(event)
        events.append({
            "start_t": event.start_t,
            "end_t": event.end_t,
            "confirmed_t": timing.get("confirmed_t"),
            "emitted_t": float(emitted_t),
            "emission_reason": reason,
            "deployment_timing": timing,
        })

    for frame_idx, (timestamp, probability) in enumerate(zip(times, probabilities), 1):
        if not np.isfinite(probability):
            if previous_available:
                for event in aggregator.flush():
                    record(event, "invalid_gap_flush", timestamp)
                aggregator = policy.make_aggregator()
                runner = SimpleNamespace(aggregator=aggregator, flush=aggregator.flush)
                session = OnlineMonitoringSession()
                session.start_detection(runner)
            previous_available = False
            continue

        previous_available = True
        # Confirmation precedes final duration filtering. Keep the historical
        # observation of the aggregator's active state local to this module.
        before = getattr(aggregator, "_active", None)
        new_events = aggregator.update(
            t=float(timestamp), fault_prob=float(probability), frame_idx=frame_idx,
        )
        active = getattr(aggregator, "_active", None)
        if before is None and active is not None:
            confirmations.append({
                "candidate_start_t": active.start_t,
                "confirmed_t": active.confirmed_t,
            })
        session.ingest_external(
            result={
                "fault_prob": float(probability),
                "label": "fault" if probability >= 0.5 else "normal",
            },
            new_events=new_events,
            frame_idx=frame_idx,
            time_s=float(timestamp),
            onset_th=policy.onset_th,
        )
        for event in new_events:
            record(event, "online_update", timestamp)

    for event in aggregator.flush():
        record(event, "end_of_stream_flush", times[-1])
    return {
        "confirmations": confirmations,
        "finalized_events": events,
        "policy": policy.to_dict(),
    }
