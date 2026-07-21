from __future__ import annotations

from dataclasses import dataclass
from threading import RLock
from typing import Any

from uav_fault.online.decision_engine import TemporalDecisionEngine
from uav_fault.online.decision_models import DecisionSignal
from uav_fault.online.risk_evaluator import RiskEvaluator


def _load_fault_detector():
    from uav_fault.online.fault_detector import FaultDetector

    return FaultDetector


class OnlineDetectionRunner:
    """Run canonical telemetry windows through detection and event aggregation."""

    def __init__(
        self,
        model_path: str,
        *,
        device: str = "cpu",
        window_size: int | None = None,
    ) -> None:
        detector_cls = _load_fault_detector()
        detector_kwargs = {"model_dir": model_path, "device": device}
        if window_size is not None:
            detector_kwargs["window_size"] = int(window_size)
        self.detector = detector_cls(**detector_kwargs)
        from uav_fault.online.event_aggregator import FaultEventAggregator

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
            return None, []
        events = self.aggregator.update(
            t=frame.get("time_s", frame_idx / 50.0),
            fault_prob=result["fault_prob"],
            frame_idx=frame_idx,
        )
        return result, events

    def flush(self):
        return self.aggregator.flush()


@dataclass(frozen=True)
class MonitoringSnapshot:
    generation: int
    state: str
    current_fault_prob: float
    risk_score: float
    risk_trend: str
    warning_stage: str
    warning_reason: str
    preview_active: bool
    preview_started_at: float | None
    last_preview_lead_time_s: float | None
    model_summary: dict[str, Any] | None


@dataclass(frozen=True)
class MonitoringUpdate:
    result: dict[str, Any] | None
    new_events: tuple[Any, ...]
    snapshot: MonitoringSnapshot
    stale: bool = False


class OnlineMonitoringSession:
    """Own one collection session's online detection and risk lifecycle."""

    def __init__(self) -> None:
        self._lock = RLock()
        self._generation = 0
        self._state = "idle"
        self._runner = None
        self._model_summary: dict[str, Any] | None = None
        self._risk_evaluator = RiskEvaluator()
        self._decision_engine = TemporalDecisionEngine()
        self._current_fault_prob = 0.0
        self._risk_score = 0.0
        self._risk_trend = "stable"
        self._warning_stage = "normal"
        self._warning_reason = "当前未发现风险演化信号"
        self._preview_started_at: float | None = None
        self._last_preview_lead_time_s: float | None = None

    @property
    def generation(self) -> int:
        with self._lock:
            return self._generation

    @property
    def runner(self):
        with self._lock:
            return self._runner

    def reset_collection(self) -> MonitoringSnapshot:
        with self._lock:
            self._generation += 1
            self._state = "idle"
            self._runner = None
            self._model_summary = None
            self._risk_evaluator.reset()
            self._decision_engine.reset()
            self._current_fault_prob = 0.0
            self._risk_score = 0.0
            self._risk_trend = "stable"
            self._warning_stage = "normal"
            self._warning_reason = "当前未发现风险演化信号"
            self._preview_started_at = None
            self._last_preview_lead_time_s = None
            return self._snapshot_locked()

    def start_detection(
        self, runner, *, model_summary: dict[str, Any] | None = None
    ) -> MonitoringSnapshot:
        with self._lock:
            if self._state == "running":
                raise RuntimeError("online monitoring detection is already running")
            self._generation += 1
            self._runner = runner
            self._model_summary = dict(model_summary) if model_summary else None
            self._state = "running"
            return self._snapshot_locked()

    def ingest(
        self,
        *,
        frame: dict[str, Any],
        window: list[dict[str, Any]],
        frame_idx: int,
        generation: int | None = None,
    ) -> MonitoringUpdate:
        with self._lock:
            current_generation = self._generation
            runner = self._runner
            if (
                self._state != "running"
                or runner is None
                or (generation is not None and generation != current_generation)
            ):
                return self._stale_update_locked()

        result, new_events = runner.update(
            frame=frame,
            window=window,
            frame_idx=frame_idx,
        )
        onset_th = float(
            getattr(getattr(runner, "aggregator", None), "onset_th", 0.6)
        )
        return self.ingest_external(
            result=result,
            new_events=new_events,
            frame_idx=frame_idx,
            time_s=frame.get("time_s", frame_idx / 50.0),
            onset_th=onset_th,
            generation=current_generation,
        )

    def ingest_external(
        self,
        *,
        result: dict[str, Any] | None,
        new_events: list[Any] | tuple[Any, ...],
        frame_idx: int,
        time_s: float,
        onset_th: float = 0.6,
        generation: int | None = None,
    ) -> MonitoringUpdate:
        with self._lock:
            if (
                self._state != "running"
                or (generation is not None and generation != self._generation)
            ):
                return self._stale_update_locked()
            if result is None:
                return MonitoringUpdate(None, (), self._snapshot_locked())

            events = tuple(new_events or ())
            fault_prob = float(result.get("fault_prob", 0.0))
            event_time = float(time_s)
            if events:
                if self._preview_started_at is not None:
                    self._last_preview_lead_time_s = round(
                        max(0.0, event_time - self._preview_started_at), 1
                    )
                self._preview_started_at = None
            elif fault_prob >= float(onset_th):
                if self._preview_started_at is None:
                    self._preview_started_at = event_time
            else:
                self._preview_started_at = None

            risk_state = self._risk_evaluator.update(
                fault_prob=fault_prob,
                has_preview=self._preview_started_at is not None,
                has_confirmed=bool(events),
            )
            self._decision_engine.ingest_signal(
                DecisionSignal(
                    ts=event_time,
                    frame_idx=int(frame_idx),
                    fault_prob=fault_prob,
                    label=str(result.get("label") or "fault"),
                    confidence=result.get("confidence"),
                    risk_score=risk_state["risk_score"],
                    risk_trend=risk_state["risk_trend"],
                    warning_stage=risk_state["warning_stage"],
                ),
                onset_th=float(onset_th),
                online_model=self._model_summary,
            )
            for event in events:
                self._decision_engine.confirm_event(event, ts=event_time)
                event.lead_time_s = self._last_preview_lead_time_s

            self._current_fault_prob = risk_state["current_fault_prob"]
            self._risk_score = risk_state["risk_score"]
            self._risk_trend = risk_state["risk_trend"]
            self._warning_stage = risk_state["warning_stage"]
            self._warning_reason = risk_state["warning_reason"]
            return MonitoringUpdate(
                dict(result), events, self._snapshot_locked()
            )

    def stop_detection(self) -> MonitoringUpdate:
        with self._lock:
            if self._state != "running":
                raise RuntimeError("online monitoring detection is not running")
            runner = self._runner
            self._generation += 1
            self._state = "stopping"

        events = tuple(runner.flush()) if runner is not None else ()
        with self._lock:
            if events:
                event_time = max(float(event.end_t) for event in events)
                if self._preview_started_at is not None:
                    self._last_preview_lead_time_s = round(
                        max(0.0, event_time - self._preview_started_at), 1
                    )
                self._preview_started_at = None
                for event in events:
                    self._decision_engine.confirm_event(
                        event, ts=float(event.end_t)
                    )
                    event.lead_time_s = self._last_preview_lead_time_s
                self._warning_stage = "critical"
                self._warning_reason = "已形成稳定故障事件，建议立即处置"
            self._runner = None
            self._state = "stopped"
            return MonitoringUpdate(None, events, self._snapshot_locked())

    def snapshot(self) -> MonitoringSnapshot:
        with self._lock:
            return self._snapshot_locked()

    def _stale_update_locked(self) -> MonitoringUpdate:
        return MonitoringUpdate(None, (), self._snapshot_locked(), stale=True)

    def _snapshot_locked(self) -> MonitoringSnapshot:
        return MonitoringSnapshot(
            generation=self._generation,
            state=self._state,
            current_fault_prob=self._current_fault_prob,
            risk_score=self._risk_score,
            risk_trend=self._risk_trend,
            warning_stage=self._warning_stage,
            warning_reason=self._warning_reason,
            preview_active=self._preview_started_at is not None,
            preview_started_at=self._preview_started_at,
            last_preview_lead_time_s=self._last_preview_lead_time_s,
            model_summary=(
                dict(self._model_summary) if self._model_summary is not None else None
            ),
        )
