from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[4]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from online.event_aggregator import FaultEventAggregator


def _load_fault_detector():
    from online.fault_detector import FaultDetector

    return FaultDetector


class DetectionRunner:
    def __init__(self, model_path: str, *, device: str = "cpu"):
        fault_detector_cls = _load_fault_detector()
        self.detector = fault_detector_cls(model_dir=model_path, device=device)
        self.aggregator = FaultEventAggregator(
            onset_th=0.6,
            clear_th=0.4,
            k_on=3,
            k_off=3,
            min_event_s=0.5,
            merge_gap_s=0.5,
        )

    @property
    def window_size(self) -> int:
        return int(self.detector.window_size)

    def predict(self, window):
        return self.detector.predict(window)

    def update(self, *, frame, window, frame_idx: int):
        result = self.predict(window)
        if result is None:
            return result, []
        t = frame.get("time_s", frame_idx / 50.0)
        events = self.aggregator.update(
            t=t,
            fault_prob=result["fault_prob"],
            frame_idx=frame_idx,
        )
        return result, events

    def flush(self):
        return self.aggregator.flush()
