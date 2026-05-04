from __future__ import annotations

import math
from collections import deque
from statistics import fmean, pstdev
from typing import Any


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float(default)
    if not math.isfinite(out):
        return float(default)
    return float(out)


class RealtimeFeatureWatcher:
    """基于稳定可得实时特征的轻量异常评分器。"""

    def __init__(
        self,
        *,
        warmup_frames: int = 150,
        baseline_window: int = 600,
        ema_alpha: float = 0.35,
    ) -> None:
        self.warmup_frames = int(warmup_frames)
        self.baseline_window = int(max(baseline_window, warmup_frames))
        self.ema_alpha = float(ema_alpha)
        self._history = {
            "attitude": deque(maxlen=self.baseline_window),
            "angular_rate": deque(maxlen=self.baseline_window),
            "speed": deque(maxlen=self.baseline_window),
            "acceleration": deque(maxlen=self.baseline_window),
        }
        self._ema_score = 0.0
        self._count = 0

    def _extract_metrics(self, row: dict[str, Any]) -> dict[str, float]:
        roll = _safe_float(row.get("roll"))
        pitch = _safe_float(row.get("pitch"))
        rollspeed = _safe_float(row.get("rollspeed"))
        pitchspeed = _safe_float(row.get("pitchspeed"))
        yawspeed = _safe_float(row.get("yawspeed"))
        vx = _safe_float(row.get("vx"))
        vy = _safe_float(row.get("vy"))
        vz = _safe_float(row.get("vz"))
        ax = _safe_float(row.get("accel_x"))
        ay = _safe_float(row.get("accel_y"))
        az = _safe_float(row.get("accel_z"))
        return {
            "attitude": math.sqrt(roll * roll + pitch * pitch),
            "angular_rate": math.sqrt(
                rollspeed * rollspeed
                + pitchspeed * pitchspeed
                + yawspeed * yawspeed
            ),
            "speed": math.sqrt(vx * vx + vy * vy + vz * vz),
            "acceleration": math.sqrt(ax * ax + ay * ay + az * az),
        }

    @staticmethod
    def _score_metric(value: float, baseline: deque[float]) -> float:
        if len(baseline) < 5:
            return 0.0
        mean = fmean(baseline)
        std = pstdev(baseline) if len(baseline) > 1 else 0.0
        std = max(std, 1e-3)
        z = abs(value - mean) / std
        return max(0.0, min(z / 6.0, 1.0))

    def update(self, row: dict[str, Any]) -> dict[str, Any]:
        metrics = self._extract_metrics(row)
        self._count += 1

        baseline_ready = self._count > self.warmup_frames
        if not baseline_ready:
            for key, value in metrics.items():
                self._history[key].append(value)
            return {
                "baseline_ready": False,
                "fault_prob": 0.0,
                "risk_score": 0.0,
                "component_scores": {key: 0.0 for key in metrics},
            }

        component_scores = {
            key: self._score_metric(value, self._history[key])
            for key, value in metrics.items()
        }
        raw_score = (
            component_scores["attitude"] * 0.25
            + component_scores["angular_rate"] * 0.3
            + component_scores["speed"] * 0.15
            + component_scores["acceleration"] * 0.3
        )
        self._ema_score = self.ema_alpha * raw_score + (1.0 - self.ema_alpha) * self._ema_score

        for key, value in metrics.items():
            self._history[key].append(value)

        return {
            "baseline_ready": True,
            "fault_prob": round(raw_score, 4),
            "risk_score": round(self._ema_score, 4),
            "component_scores": {
                key: round(value, 4) for key, value in component_scores.items()
            },
        }
