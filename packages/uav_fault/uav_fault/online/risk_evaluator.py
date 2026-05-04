from __future__ import annotations

from collections import deque


class RiskEvaluator:
    def __init__(
        self,
        *,
        alpha: float = 0.35,
        short_window: int = 3,
        long_window: int = 8,
        watch_th: float = 0.35,
        warning_th: float = 0.55,
        critical_th: float = 0.75,
        trend_delta: float = 0.05,
    ):
        self.alpha = alpha
        self.short_window = short_window
        self.long_window = long_window
        self.watch_th = watch_th
        self.warning_th = warning_th
        self.critical_th = critical_th
        self.trend_delta = trend_delta
        self.reset()

    def reset(self) -> None:
        self._ema: float | None = None
        self._history = deque(maxlen=max(self.long_window, self.short_window, 1))

    def update(
        self,
        *,
        fault_prob: float,
        has_preview: bool = False,
        has_confirmed: bool = False,
    ) -> dict:
        prob = max(0.0, min(1.0, float(fault_prob or 0.0)))
        if self._ema is None:
            self._ema = prob
        else:
            self._ema = self.alpha * prob + (1.0 - self.alpha) * self._ema
        self._history.append(prob)

        trend = self._compute_trend()
        stage = self._compute_stage(
            has_preview=has_preview, has_confirmed=has_confirmed
        )
        reason = self._build_reason(stage=stage, trend=trend)

        return {
            "current_fault_prob": round(prob, 3),
            "risk_score": round(float(self._ema or 0.0), 3),
            "risk_trend": trend,
            "warning_stage": stage,
            "warning_reason": reason,
        }

    def _compute_trend(self) -> str:
        if len(self._history) < 2:
            return "stable"
        short = list(self._history)[-self.short_window :]
        long = list(self._history)[-self.long_window :]
        short_mean = sum(short) / len(short)
        long_mean = sum(long) / len(long)
        delta = short_mean - long_mean
        if delta > self.trend_delta:
            return "rising"
        if delta < -self.trend_delta:
            return "falling"
        return "stable"

    def _compute_stage(self, *, has_preview: bool, has_confirmed: bool) -> str:
        score = float(self._ema or 0.0)
        if has_confirmed or score >= self.critical_th:
            return "critical"
        if has_preview or score >= self.warning_th:
            return "warning"
        if score >= self.watch_th:
            return "watch"
        return "normal"

    def _build_reason(self, *, stage: str, trend: str) -> str:
        score = float(self._ema or 0.0)
        if stage == "critical":
            return "已形成稳定故障事件，建议立即处置"
        if stage == "warning":
            if trend == "rising":
                return "风险连续上升，异常概率已超过预警阈值"
            return "异常概率已进入预警区间，建议尽快复核"
        if stage == "watch":
            if trend == "rising":
                return "风险连续上升，进入观察阶段"
            return "检测到轻微异常波动，建议持续观察"
        if score <= 0.05:
            return "当前未发现风险演化信号"
        return "风险处于低位，当前链路稳定"
