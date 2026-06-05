from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class DecisionSignal:
    ts: float
    frame_idx: int
    fault_prob: float
    label: str
    confidence: float | None
    risk_score: float
    risk_trend: str
    warning_stage: str


@dataclass
class DecisionState:
    episode_id: str | None = None
    state: str = "IDLE"
    current_fault_prob: float = 0.0
    risk_score: float = 0.0
    risk_trend: str = "stable"
    warning_stage: str = "normal"
    warning_reason: str = "当前未发现风险演化信号"
    preview_started_at: float | None = None
    preview_alert: dict[str, Any] | None = None
    top_recommendations: list[dict[str, Any]] = field(default_factory=list)
    last_preview_lead_time_s: float | None = None
    fault_label: str | None = None
    severity: str | None = None
    offline_evidence: dict[str, Any] = field(default_factory=dict)


@dataclass
class DecisionOutcome:
    episode_id: str | None
    state: str
    warning_stage: str
    risk_score: float
    risk_trend: str
    warning_reason: str
    lead_time_s: float | None
    fault_label: str | None
    severity: str | None
    evidence_summary: dict[str, Any] | None
    recommendations: list[dict[str, Any]]
