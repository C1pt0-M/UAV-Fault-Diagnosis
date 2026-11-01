"""Shared scheduling and event policy; transports still own queues and input checks."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math


def input_contract_check_due(
    *, frame_idx: int, window_length: int, min_window_size: int,
    every_frames: int = 1, has_previous: bool = False, force: bool = False,
) -> bool:
    """Share cadence decisions without owning or weakening the input contract check."""
    if isinstance(every_frames, bool) or not isinstance(every_frames, int) or every_frames <= 0:
        raise ValueError("every_frames must be a positive integer")
    return window_length >= min_window_size and (
        force or not has_previous or frame_idx % every_frames == 0
    )


@dataclass(frozen=True)
class OnlineDetectionPolicy:
    inference_every: int = 1
    onset_th: float = 0.6
    clear_th: float = 0.4
    k_on: int = 3
    k_off: int = 3
    min_event_s: float = 0.5
    merge_gap_s: float = 0.5

    def __post_init__(self) -> None:
        for name in ("inference_every", "k_on", "k_off"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        for name in ("onset_th", "clear_th", "min_event_s", "merge_gap_s"):
            value = getattr(self, name)
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
        if not 0 <= self.clear_th <= self.onset_th <= 1:
            raise ValueError("thresholds must satisfy 0 <= clear_th <= onset_th <= 1")

    def should_infer(self, *, frame_idx: int, window_length: int, window_size: int) -> bool:
        """frame_idx is the 1-based count since the transport started, not since arming."""
        return (
            frame_idx > 0
            and window_length >= window_size
            and frame_idx % self.inference_every == 0
        )

    def window_ends(self, n_rows: int, window_size: int) -> range:
        """1-based exclusive ends, with the same phase as should_infer."""
        first = ((window_size + self.inference_every - 1) // self.inference_every) * self.inference_every
        return range(first, n_rows + 1, self.inference_every)

    def make_aggregator(self):
        # Local import keeps model and torch imports out of transport startup.
        from uav_fault.online.event_aggregator import FaultEventAggregator

        return FaultEventAggregator(
            onset_th=self.onset_th,
            clear_th=self.clear_th,
            k_on=self.k_on,
            k_off=self.k_off,
            min_event_s=self.min_event_s,
            merge_gap_s=self.merge_gap_s,
        )

    def to_dict(self) -> dict:
        return asdict(self)
