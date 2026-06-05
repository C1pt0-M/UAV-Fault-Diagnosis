"""
Fault event aggregation from streaming prediction points.

We intentionally operate on low-frequency inference points (e.g. every N frames),
not on raw telemetry frames. This keeps persistence compact and "delivery-ready":
we store events (start/end) rather than full 50Hz telemetry in the DB.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class AggregatedEvent:
    start_t: float
    end_t: float
    start_frame_idx: int | None
    end_frame_idx: int | None
    n_points: int
    max_fault_prob: float
    mean_fault_prob: float
    meta: dict[str, Any]


@dataclass
class _EventAccum:
    start_t: float
    start_frame_idx: int | None
    end_t: float
    end_frame_idx: int | None
    n_points: int
    max_fault_prob: float
    sum_fault_prob: float

    def add_point(self, t: float, fault_prob: float, frame_idx: int | None):
        self.end_t = float(t)
        self.end_frame_idx = frame_idx
        self.n_points += 1
        if fault_prob > self.max_fault_prob:
            self.max_fault_prob = float(fault_prob)
        self.sum_fault_prob += float(fault_prob)

    def to_event(self, meta: dict[str, Any]) -> AggregatedEvent:
        mean = self.sum_fault_prob / max(1, self.n_points)
        return AggregatedEvent(
            start_t=float(self.start_t),
            end_t=float(self.end_t),
            start_frame_idx=self.start_frame_idx,
            end_frame_idx=self.end_frame_idx,
            n_points=int(self.n_points),
            max_fault_prob=float(self.max_fault_prob),
            mean_fault_prob=float(mean),
            meta=meta,
        )


class FaultEventAggregator:
    """Convert a stream of (t, fault_prob) points into fault events using hysteresis."""

    def __init__(
        self,
        *,
        onset_th: float = 0.6,
        clear_th: float = 0.4,
        k_on: int = 3,
        k_off: int = 3,
        min_event_s: float = 0.5,
        merge_gap_s: float = 0.5,
    ):
        if clear_th > onset_th:
            raise ValueError("clear_th must be <= onset_th for hysteresis to make sense")
        if k_on <= 0 or k_off <= 0:
            raise ValueError("k_on and k_off must be positive integers")

        self.onset_th = float(onset_th)
        self.clear_th = float(clear_th)
        self.k_on = int(k_on)
        self.k_off = int(k_off)
        self.min_event_s = float(min_event_s)
        self.merge_gap_s = float(merge_gap_s)

        # Candidate streak (to start an event).
        self._cand: list[tuple[float, float, int | None]] = []

        # Active event accumulation.
        self._active: _EventAccum | None = None

        # If merge_gap_s > 0, we keep the last closed event pending until the gap passes.
        self._pending_closed: _EventAccum | None = None
        self._off_count = 0

        self._last_t: float | None = None

    def _finalize_if_ready(self, t_now: float) -> list[AggregatedEvent]:
        """Emit pending event once we are sure it cannot be merged anymore."""
        if self._pending_closed is None:
            return []
        if self.merge_gap_s <= 0:
            ev = self._pending_closed
            self._pending_closed = None
            out = self._maybe_keep(ev)
            return [out] if out is not None else []

        # Wait until time has advanced enough.
        if (t_now - self._pending_closed.end_t) <= self.merge_gap_s:
            return []

        ev = self._pending_closed
        self._pending_closed = None
        out = self._maybe_keep(ev)
        return [out] if out is not None else []

    def _maybe_keep(self, acc: _EventAccum) -> AggregatedEvent | None:
        dur = float(acc.end_t) - float(acc.start_t)
        if dur < self.min_event_s:
            return None

        meta = {
            "onset_th": self.onset_th,
            "clear_th": self.clear_th,
            "k_on": self.k_on,
            "k_off": self.k_off,
            "min_event_s": self.min_event_s,
            "merge_gap_s": self.merge_gap_s,
        }
        return acc.to_event(meta)

    def update(
        self,
        *,
        t: float,
        fault_prob: float,
        frame_idx: int | None = None,
    ) -> list[AggregatedEvent]:
        """Consume one inference point and return newly finalized events (if any)."""
        t = float(t)
        fault_prob = float(fault_prob)

        finalized: list[AggregatedEvent] = []

        # If we have a pending event, it may become final as time advances.
        if self._pending_closed is not None:
            finalized.extend(self._finalize_if_ready(t))

        self._last_t = t

        # ── Not currently in an event: look for onset streak ──
        if self._active is None:
            if fault_prob >= self.onset_th:
                self._cand.append((t, fault_prob, frame_idx))
                if len(self._cand) >= self.k_on:
                    # Start a new event from the beginning of the streak.
                    start_t, start_p, start_frame = self._cand[0]

                    # Merge with a recently closed event if within gap.
                    if (
                        self._pending_closed is not None
                        and (start_t - self._pending_closed.end_t) <= self.merge_gap_s
                    ):
                        acc = self._pending_closed
                        self._pending_closed = None
                        self._active = acc
                    else:
                        self._active = _EventAccum(
                            start_t=float(start_t),
                            start_frame_idx=start_frame,
                            end_t=float(start_t),
                            end_frame_idx=start_frame,
                            n_points=0,
                            max_fault_prob=0.0,
                            sum_fault_prob=0.0,
                        )

                    # Add all candidate points into the active event.
                    assert self._active is not None
                    for tt, pp, ff in self._cand:
                        self._active.add_point(tt, pp, ff)
                    self._cand = []
                    self._off_count = 0
            else:
                self._cand = []
            return finalized

        # ── In an event: extend while above clear threshold, close on k_off ──
        if fault_prob >= self.clear_th:
            self._active.add_point(t, fault_prob, frame_idx)
            self._off_count = 0
            return finalized

        # Below clear threshold: count towards ending the event.
        self._off_count += 1
        if self._off_count < self.k_off:
            return finalized

        # Close event at the last "kept" point (end_t is already last >= clear_th point).
        closed = self._active
        self._active = None
        self._off_count = 0
        self._cand = []  # reset candidate streak after a closure

        # Keep it pending for merge, or emit immediately.
        if self.merge_gap_s > 0:
            self._pending_closed = closed
        else:
            ev = self._maybe_keep(closed)
            if ev is not None:
                finalized.append(ev)

        return finalized

    def flush(self) -> list[AggregatedEvent]:
        """Finalize any open/pending event (end-of-stream)."""
        out: list[AggregatedEvent] = []

        # Close active event if any.
        if self._active is not None:
            closed = self._active
            self._active = None
            self._pending_closed = closed

        # Emit pending event regardless of merge gap.
        if self._pending_closed is not None:
            ev = self._maybe_keep(self._pending_closed)
            if ev is not None:
                out.append(ev)
            self._pending_closed = None

        # Reset candidates.
        self._cand = []
        self._off_count = 0
        return out
