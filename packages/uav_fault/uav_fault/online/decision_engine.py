from __future__ import annotations

from typing import Any
from uuid import uuid4

from uav_fault.online.decision_models import DecisionOutcome, DecisionSignal, DecisionState


class TemporalDecisionEngine:
    def __init__(self):
        self._state = DecisionState()

    def reset(self) -> None:
        self._state = DecisionState()

    def current_state(self) -> DecisionState:
        return self._state

    def ingest_signal(
        self,
        signal: DecisionSignal,
        *,
        onset_th: float,
        online_model: dict[str, Any] | None,
    ) -> DecisionOutcome:
        if signal.fault_prob >= float(onset_th):
            if self._state.episode_id is None:
                self._state.episode_id = f"ep_{uuid4().hex[:12]}"
            self._state.state = "PREVIEW"
            if self._state.preview_started_at is None:
                self._state.preview_started_at = signal.ts
            self._state.preview_alert = {
                "source": "realtime_preview",
                "label": signal.label,
                "confidence": signal.confidence,
                "probability": signal.confidence,
                "model": online_model,
            }
        else:
            self._state.state = "WATCH" if signal.warning_stage == "watch" else "IDLE"
            self._state.preview_started_at = None
            self._state.preview_alert = None

        self._state.current_fault_prob = signal.fault_prob
        self._state.risk_score = signal.risk_score
        self._state.risk_trend = signal.risk_trend
        self._state.warning_stage = signal.warning_stage
        self._state.warning_reason = self._reason_for_stage(
            signal.warning_stage, signal.risk_trend
        )
        self._state.top_recommendations = self._recommend_actions_for_stage(
            signal.warning_stage
        )

        return self._to_outcome()

    def confirm_event(self, event: Any, *, ts: float | None = None) -> DecisionOutcome:
        lead_time_s = None
        if self._state.preview_started_at is not None and ts is not None:
            lead_time_s = round(
                max(0.0, float(ts) - float(self._state.preview_started_at)), 1
            )
        self._state.last_preview_lead_time_s = lead_time_s
        self._state.state = "CONFIRMED_EVENT"
        self._state.warning_stage = "critical"
        self._state.warning_reason = "已形成稳定故障事件，建议立即处置"
        self._state.preview_started_at = None
        self._state.preview_alert = None
        self._state.top_recommendations = self._recommend_actions_for_stage("critical")
        return self._to_outcome()

    def attach_offline_diagnosis(self, diagnosis: dict[str, Any]) -> DecisionOutcome:
        if self._state.episode_id is None:
            self._state.episode_id = f"ep_{uuid4().hex[:12]}"
        raw_severity = diagnosis.get("severity")
        severity = raw_severity if isinstance(raw_severity, dict) else {}
        self._state.state = "DECIDED"
        self._state.fault_label = (
            diagnosis.get("fault_label")
            or diagnosis.get("label")
            or self._state.fault_label
        )
        self._state.severity = severity.get("level") or self._state.severity
        self._state.offline_evidence = {
            "severity": severity,
            "evidence": diagnosis.get("evidence") or [],
        }
        recs = diagnosis.get("recommendations")
        if isinstance(recs, list) and recs:
            self._state.top_recommendations = recs
        return self._to_outcome()

    def _to_outcome(self) -> DecisionOutcome:
        evidence_summary = None
        if self._state.offline_evidence:
            severity = self._state.offline_evidence.get("severity") or {}
            evidence = self._state.offline_evidence.get("evidence") or []
            evidence_summary = {
                "source": "offline",
                "fault_label": self._state.fault_label,
                "severity": {
                    "level": severity.get("level"),
                    "score": severity.get("score"),
                    "method": severity.get("method"),
                },
                "top_evidence": list(evidence[:3])
                if isinstance(evidence, list)
                else [],
            }
        return DecisionOutcome(
            episode_id=self._state.episode_id,
            state=self._state.state,
            warning_stage=self._state.warning_stage,
            risk_score=self._state.risk_score,
            risk_trend=self._state.risk_trend,
            warning_reason=self._state.warning_reason,
            lead_time_s=self._state.last_preview_lead_time_s,
            fault_label=self._state.fault_label,
            severity=self._state.severity,
            evidence_summary=evidence_summary,
            recommendations=self._state.top_recommendations,
        )

    def _recommend_actions_for_stage(self, stage: str) -> list[dict[str, Any]]:
        if stage == "critical":
            return [
                {
                    "title": "立即中止当前任务并执行人工复核",
                    "priority": "P0",
                    "reason_codes": ["mission_abort"],
                }
            ]
        if stage == "warning":
            return [
                {
                    "title": "建议结束当前架次后执行离线诊断",
                    "priority": "P1",
                    "reason_codes": ["offline_followup"],
                }
            ]
        if stage == "watch":
            return [
                {
                    "title": "持续观察后续风险窗口变化",
                    "priority": "P2",
                    "reason_codes": ["realtime_observe"],
                }
            ]
        return [
            {
                "title": "继续监测当前飞行状态",
                "priority": "P3",
                "reason_codes": ["stable_monitor"],
            }
        ]

    def _reason_for_stage(self, stage: str, trend: str) -> str:
        if stage == "critical":
            return "已形成稳定故障事件，建议立即处置"
        if stage == "warning":
            return (
                "风险连续上升，异常概率已超过预警阈值"
                if trend == "rising"
                else "异常概率已进入预警区间，建议尽快复核"
            )
        if stage == "watch":
            return (
                "风险连续上升，进入观察阶段"
                if trend == "rising"
                else "检测到轻微异常波动，建议持续观察"
            )
        return "当前未发现风险演化信号"
