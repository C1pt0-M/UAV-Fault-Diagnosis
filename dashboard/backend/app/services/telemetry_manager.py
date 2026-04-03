"""遥测进程管理服务。

管理 PX4 SITL → Collector → Receiver → Detector 的完整生命周期。
采用单例模式，同一时间只允许一个采集会话。
"""

import copy
import os
import random
import shutil
import signal
import socket
import subprocess
import sys
import threading
import time
from collections import deque
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[4]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from app.services.diagnosis_contract import (
    build_diagnosis_payload,
    build_time_range,
    canonical_severity_from_probability,
    legacy_severity_from_canonical,
    normalize_recommendations,
)
from app.services.model_registry import resolve_registered_model
from app.services.telemetry_backends import PX4ScenarioBackend, ReplayScenarioBackend
from app.utils.path_store import resolve_local_path, serialize_local_path
from telemetry.receiver import TelemetryReceiver
from telemetry.replay_metrics import gt_event_metrics
from telemetry.schema_adapter import (
    CANONICAL_SCHEMA_VERSION,
    adapt_px4_live_frame,
    canonicalize_rfly_frame,
)
from online.decision_engine import TemporalDecisionEngine
from online.decision_models import DecisionSignal
from online.risk_evaluator import RiskEvaluator

DetectionRunner = None


def _generate_id(prefix: str) -> str:
    return f"{prefix}-{random.randint(100000, 999999)}"


def _build_realtime_recommendations(severity: str, probability: float) -> list[dict]:
    rec_map = {
        "light": [
            {
                "title": "持续观察后续窗口，确认异常是否持续",
                "priority": "P2",
                "reason_codes": ["realtime_observe"],
            },
            {
                "title": "记录当前飞行姿态与动力输出，准备离线复核",
                "priority": "P2",
                "reason_codes": ["realtime_capture"],
            },
        ],
        "medium": [
            {
                "title": "建议结束当前架次后执行离线诊断",
                "priority": "P1",
                "reason_codes": ["offline_followup"],
            },
            {
                "title": "检查动力链与关键传感器状态",
                "priority": "P1",
                "reason_codes": ["inspect_subsystems"],
            },
        ],
        "heavy": [
            {
                "title": "立即中止当前任务并执行人工复核",
                "priority": "P0",
                "reason_codes": ["mission_abort"],
            },
            {
                "title": "重点检查推进系统、螺旋桨与 IMU 传感链路",
                "priority": "P0",
                "reason_codes": ["critical_inspection"],
            },
        ],
    }
    recommendations = rec_map.get(severity, rec_map["medium"])
    if probability is not None and probability >= 0.95 and severity == "heavy":
        recommendations = recommendations + [
            {
                "title": "保留当前遥测会话作为高优先级故障样本",
                "priority": "P1",
                "reason_codes": ["preserve_session"],
            }
        ]
    return normalize_recommendations(recommendations)


class TelemetryPersistence:
    def __init__(self, manager):
        self.manager = manager

    def create_session_record(
        self,
        *,
        session_id,
        drone_id,
        session_type,
        csv_path,
        backend_name,
        schema_version,
        source_path=None,
        meta=None,
    ) -> None:
        if not self.manager.app:
            return
        try:
            with self.manager.app.app_context():
                from app import db
                from app.models.telemetry_session import TelemetrySession

                session = TelemetrySession(
                    id=session_id,
                    drone_id=drone_id,
                    session_type=session_type,
                    scenario_backend=backend_name,
                    schema_version=schema_version,
                    source_path=serialize_local_path(source_path, self.manager.app.config),
                    start_time=datetime.now(),
                    csv_path=serialize_local_path(csv_path, self.manager.app.config),
                    status="running",
                )
                session.set_meta(meta or {})
                db.session.add(session)
                db.session.commit()
        except Exception as e:
            print(f"[telemetry_manager] 创建会话记录失败: {e}")

    def delete_session_record(self, *, session_id) -> None:
        if not self.manager.app:
            return
        try:
            with self.manager.app.app_context():
                from app import db
                from app.models.telemetry_session import TelemetrySession

                session = db.session.get(TelemetrySession, session_id)
                if session:
                    db.session.delete(session)
                    db.session.commit()
        except Exception as e:
            print(f"[telemetry_manager] 删除会话记录失败: {e}")

    def finalize_session(
        self, *, session_id, duration, frame_count, fault_count, replay_metrics=None
    ) -> None:
        if not self.manager.app:
            return
        try:
            with self.manager.app.app_context():
                from app import db
                from app.models.telemetry_session import TelemetrySession

                session = db.session.get(TelemetrySession, session_id)
                if session:
                    meta = session.get_meta()
                    if replay_metrics:
                        meta["replay_metrics"] = replay_metrics
                        session.set_meta(meta)
                    session.end_time = datetime.now()
                    session.duration_seconds = round(duration, 1)
                    session.frame_count = frame_count
                    session.fault_events_count = fault_count
                    session.status = "completed"
                    db.session.commit()
        except Exception as e:
            print(f"[telemetry_manager] 更新会话记录失败: {e}")

    def update_session_detection(self, *, session_id, has_detection) -> None:
        if not self.manager.app:
            return
        try:
            with self.manager.app.app_context():
                from app import db
                from app.models.telemetry_session import TelemetrySession

                session = db.session.get(TelemetrySession, session_id)
                if session:
                    session.has_detection = has_detection
                    db.session.commit()
        except Exception as e:
            print(f"[telemetry_manager] 更新检测状态失败: {e}")

    def write_log(self, *, drone_id, log_type, content) -> None:
        if not self.manager.app:
            return
        try:
            with self.manager.app.app_context():
                from app import db
                from app.models.log import Log

                log = Log(
                    id=_generate_id("LOG"),
                    drone_id=drone_id,
                    type=log_type,
                    content=content,
                )
                db.session.add(log)
                db.session.commit()
        except Exception as e:
            print(f"[telemetry_manager] 写入日志失败: {e}")

    def record_fault_event(
        self, *, session_id, event, diagnosis, legacy_severity
    ) -> None:
        if not self.manager.app:
            return
        try:
            with self.manager.app.app_context():
                from app import db
                from app.models.log import Log
                from app.models.telemetry_session import TelemetryFaultEvent

                fault_event = TelemetryFaultEvent(
                    id=_generate_id("FE"),
                    session_id=session_id,
                    start_time=event.start_t,
                    end_time=event.end_t,
                    duration=event.end_t - event.start_t,
                    max_fault_prob=event.max_fault_prob,
                    mean_fault_prob=event.mean_fault_prob,
                    data_points=event.n_points,
                    severity=legacy_severity,
                )
                fault_event.set_diagnosis(diagnosis)
                db.session.add(fault_event)
                db.session.add(
                    Log(
                        id=_generate_id("LOG"),
                        type="realtime_detection",
                        content=f"检出故障事件 {event.start_t:.1f}s-{event.end_t:.1f}s，最大概率 {event.max_fault_prob:.1%}",
                    )
                )
                db.session.commit()
        except Exception as e:
            print(f"[telemetry_manager] 写入故障事件失败: {e}")


class TelemetryProcessManager:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self, app=None):
        if self._initialized:
            return
        self._initialized = True

        self.app = app
        # 进程/线程句柄
        self._px4_proc: subprocess.Popen | None = None
        self._collector_proc: subprocess.Popen | None = None
        self._flight_proc: subprocess.Popen | None = None
        self._receiver: TelemetryReceiver | None = None
        self._receiver_thread: threading.Thread | None = None
        # 检测
        self._detector = None
        self._aggregator = None
        self._detecting = False
        self._detection_callback = None
        # 状态
        self._session_id: str | None = None
        self._start_time: float | None = None
        self._planned_duration_s: float | None = None
        self._frame_count = 0
        self._fault_events: list = []
        self._preview_alert = None
        self._lock = threading.Lock()
        # 盲测模式
        self._blind_test_active = False
        self._blind_test_ground_truth: dict | None = None
        # 实时数据缓冲
        self._realtime_buffer: deque = deque(maxlen=500)
        # 路径
        self._active_model: dict | None = None
        self._telemetry_python_bin: str | None = None
        self._px4_root = os.path.expanduser(os.environ.get("PX4_ROOT", "~/PX4-Autopilot"))
        self._src_dir = SRC_DIR
        self._backend = None
        self._backend_name = None
        self._backend_options: dict = {}
        self._schema_version = None
        self._feature_coverage = None
        self._last_frame_time_s = None
        self._last_frame_wall_time = None
        self._input_contract_status = None
        self._startup_phase = None
        self._timeout_reason = None
        self._replay_metrics: dict | None = None
        self._current_fault_prob: float = 0.0
        self._risk_score: float = 0.0
        self._risk_trend: str = "stable"
        self._warning_stage: str = "normal"
        self._warning_reason: str = "当前未发现风险演化信号"
        self._preview_started_at: float | None = None
        self._top_recommendations: list[dict] = []
        self._last_preview_lead_time_s: float | None = None
        self._risk_evaluator = RiskEvaluator()
        self._decision_engine = TemporalDecisionEngine()
        self._stop_in_progress = False
        self._last_stop_result: dict | None = None
        self._last_stop_error: Exception | None = None
        self._stop_condition = threading.Condition(self._lock)
        self._persistence = TelemetryPersistence(self)
        self._sessions_dir = PROJECT_ROOT / "data" / "telemetry_sessions"
        self._sessions_dir.mkdir(parents=True, exist_ok=True)

    def init_app(self, app):
        self.app = app
        self._sessions_dir = Path(app.config["RUNTIME_DIR"]) / "telemetry_sessions"
        self._sessions_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _get_detection_runner_cls():
        global DetectionRunner
        if DetectionRunner is None:
            from app.services.detection_runner import DetectionRunner as _DetectionRunner

            DetectionRunner = _DetectionRunner
        return DetectionRunner

    @staticmethod
    def _popen_session_kwargs() -> dict:
        if os.name == "nt":
            creationflags = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
            return {"creationflags": creationflags} if creationflags else {}
        return {"start_new_session": True}

    def _get_px4_root(self) -> str:
        if self.app:
            configured = str(self.app.config.get("PX4_ROOT", "") or "").strip()
            if configured:
                return os.path.expanduser(configured)
        return self._px4_root

    def _reset_runtime_state(self) -> None:
        self._session_id = None
        self._start_time = None
        self._planned_duration_s = None
        self._frame_count = 0
        self._fault_events = []
        self._preview_alert = None
        self._detecting = False
        self._active_model = None
        self._backend = None
        self._backend_name = None
        self._backend_options = {}
        self._schema_version = None
        self._feature_coverage = None
        self._last_frame_time_s = None
        self._last_frame_wall_time = None
        self._input_contract_status = None
        self._startup_phase = None
        self._timeout_reason = None
        self._replay_metrics = None
        self._current_fault_prob = 0.0
        self._risk_score = 0.0
        self._risk_trend = "stable"
        self._warning_stage = "normal"
        self._warning_reason = "当前未发现风险演化信号"
        self._preview_started_at = None
        self._top_recommendations = []
        self._last_preview_lead_time_s = None
        self._risk_evaluator.reset()
        self._decision_engine.reset()

    def _activate_runtime_state(
        self,
        *,
        session_id: str,
        sim_time: float,
        backend_name: str,
        backend_options: dict,
        backend_instance,
        session_type: str,
    ) -> None:
        self._session_id = session_id
        self._start_time = time.time()
        self._planned_duration_s = sim_time
        self._frame_count = 0
        self._fault_events = []
        self._preview_alert = None
        self._detecting = False
        self._active_model = None
        self._backend_name = backend_name
        self._backend_options = backend_options
        self._schema_version = CANONICAL_SCHEMA_VERSION
        self._feature_coverage = None
        self._last_frame_time_s = None
        self._last_frame_wall_time = None
        self._input_contract_status = None
        self._startup_phase = "starting"
        self._timeout_reason = None
        self._replay_metrics = None
        self._current_fault_prob = 0.0
        self._risk_score = 0.0
        self._risk_trend = "stable"
        self._warning_stage = "normal"
        self._warning_reason = "当前未发现风险演化信号"
        self._preview_started_at = None
        self._top_recommendations = []
        self._last_preview_lead_time_s = None
        self._risk_evaluator.reset()
        self._decision_engine.reset()
        self._stop_in_progress = False
        self._last_stop_result = None
        self._last_stop_error = None
        self._backend = backend_instance
        if session_type != "blind_test":
            self._blind_test_active = False
            self._blind_test_ground_truth = None

    def _python_supports_telemetry(self, python_bin: str) -> bool:
        try:
            completed = subprocess.run(
                [
                    python_bin,
                    "-c",
                    "import importlib.util;import sys;"
                    "mods=('mavsdk','pymavlink');"
                    "sys.exit(0 if all(importlib.util.find_spec(m) for m in mods) else 1)",
                ],
                capture_output=True,
                text=True,
                timeout=10,
                check=False,
            )
            return completed.returncode == 0
        except Exception:
            return False

    def _resolve_telemetry_python_bin(self) -> str:
        if self._telemetry_python_bin and self._python_supports_telemetry(
            self._telemetry_python_bin
        ):
            return self._telemetry_python_bin

        if self.app:
            cfg = self.app.config
            candidates = [
                cfg.get("TELEMETRY_PYTHON_BIN", ""),
                cfg.get("ML_PYTHON_BIN", ""),
            ]
        else:
            candidates = []
        candidates.extend(
            [
                sys.executable,
                shutil.which("python3"),
                shutil.which("python"),
            ]
        )

        seen: set[str] = set()
        for raw in candidates:
            python_bin = str(raw or "").strip()
            if not python_bin or python_bin in seen:
                continue
            seen.add(python_bin)
            if not os.path.exists(python_bin):
                continue
            if self._python_supports_telemetry(python_bin):
                self._telemetry_python_bin = python_bin
                return python_bin

        raise RuntimeError(
            "未找到可用的遥测运行环境：需要同时安装 mavsdk 和 pymavlink。"
            "请配置 TELEMETRY_PYTHON_BIN 指向可用解释器。"
        )

    # ── 状态查询 ──────────────────────────────────────────────

    @property
    def is_collecting(self) -> bool:
        if self._backend_name == "replay" and self._backend is not None:
            thread = getattr(self._backend, "_thread", None)
            return thread is not None and thread.is_alive()
        return self._receiver_thread is not None and self._receiver_thread.is_alive()

    @property
    def is_detecting(self) -> bool:
        return self._detecting and self.is_collecting

    def _build_realtime_diagnosis(
        self,
        *,
        probability: float,
        start_time: float,
        end_time: float,
        lead_time_s: float | None = None,
    ) -> dict:
        severity = canonical_severity_from_probability(probability)
        return build_diagnosis_payload(
            source="realtime",
            label="fault",
            confidence=probability,
            severity=severity,
            recommendations=_build_realtime_recommendations(severity, probability),
            model=self._get_online_model_summary(),
            time_range=build_time_range(start_time=start_time, end_time=end_time),
            lead_time_s=lead_time_s,
            evidence_summary=self._build_online_evidence_summary(
                label="fault",
                severity=severity,
                probability=probability,
                start_time=start_time,
                end_time=end_time,
                lead_time_s=lead_time_s,
            ),
        )

    def _build_online_evidence_summary(
        self,
        *,
        label: str,
        severity: str,
        probability: float | None,
        start_time: float | None,
        end_time: float | None,
        lead_time_s: float | None = None,
    ) -> dict:
        return {
            "source": "online",
            "fault_label": label,
            "severity": {
                "level": severity,
                "score": None if probability is None else round(float(probability), 3),
                "method": "online_probability_threshold",
            },
            "top_evidence": [
                {
                    "kind": "signal",
                    "name": "fault_probability",
                    "value": None
                    if probability is None
                    else round(float(probability), 3),
                    "status": "observed",
                },
                {
                    "kind": "risk",
                    "name": "warning_stage",
                    "value": self._warning_stage,
                    "status": "derived",
                },
                {
                    "kind": "window",
                    "name": "lead_time_s",
                    "value": None
                    if lead_time_s is None
                    else round(float(lead_time_s), 2),
                    "status": "derived",
                },
            ],
        }

    def _recommend_actions_for_stage(self, stage: str) -> list[dict]:
        if stage == "critical":
            return normalize_recommendations(
                [
                    {"title": "立即中止当前任务并执行人工复核", "priority": "P0"},
                    {"title": "重点检查推进系统、螺旋桨与 IMU 链路", "priority": "P0"},
                ]
            )
        if stage == "warning":
            return normalize_recommendations(
                [
                    {"title": "建议结束当前架次后执行离线诊断", "priority": "P1"},
                    {"title": "检查动力链与姿态稳定性", "priority": "P1"},
                ]
            )
        if stage == "watch":
            return normalize_recommendations(
                [
                    {"title": "持续观察后续风险窗口变化", "priority": "P2"},
                    {"title": "记录当前姿态与动力输出供后续复核", "priority": "P2"},
                ]
            )
        return normalize_recommendations(
            [{"title": "继续监测当前飞行状态", "priority": "P3"}]
        )

    def _session_business_status(self) -> str:
        if self.is_detecting:
            return "detecting"
        if self.is_collecting:
            return "collecting"
        return "idle"

    def _backend_process_status(self) -> dict:
        if self._backend is not None:
            try:
                return dict(self._backend.status() or {})
            except Exception:
                return {
                    "px4_sitl": self._proc_status(self._px4_proc),
                    "collector": self._proc_status(self._collector_proc),
                    "receiver": "running" if self.is_collecting else None,
                }
        return {
            "px4_sitl": self._proc_status(self._px4_proc),
            "collector": self._proc_status(self._collector_proc),
            "receiver": "running" if self.is_collecting else None,
        }

    def _transport_status(self) -> str:
        processes = self._backend_process_status()
        if self._timeout_reason:
            return "degraded"
        if self._backend_name == "replay" and self.is_collecting:
            return "connected"
        if self.is_collecting and processes.get("collector") == "running":
            return "connected"
        if self.is_collecting:
            return "degraded"
        return "idle"

    def _collect_replay_metrics(self, session_id: str | None) -> dict | None:
        if self._backend_name != "replay" or not self.app or not session_id:
            return None
        try:
            with self.app.app_context():
                from app import db
                from app.models.telemetry_session import TelemetrySession

                session = db.session.get(TelemetrySession, session_id)
                resolved_csv_path = (
                    resolve_local_path(session.csv_path, self.app.config)
                    if session is not None and session.csv_path
                    else ''
                )
                if (
                    session is None
                    or not resolved_csv_path
                    or not os.path.exists(resolved_csv_path)
                ):
                    return None
                import pandas as pd

                df = pd.read_csv(resolved_csv_path)
        except Exception as e:
            print(f"[telemetry_manager] 计算 replay metrics 失败: {e}")
            return None
        try:
            metrics = gt_event_metrics(df, list(self._fault_events))
        except Exception as e:
            print(f"[telemetry_manager] 汇总 replay metrics 失败: {e}")
            return None
        return {k: v for k, v in metrics.items() if v is not None}

    def _snapshot_runtime_value(self, value):
        return copy.deepcopy(value) if value is not None else None

    def get_status(self) -> dict:
        with self._lock:
            uptime = time.time() - self._start_time if self._start_time else 0
            fault_events = [self._event_to_dict(e) for e in self._fault_events[-10:]]
            preview_alert = self._snapshot_runtime_value(self._preview_alert)
            processes = dict(self._backend_process_status())
            processes["detector"] = "running" if self.is_detecting else None
            feature_coverage = self._snapshot_runtime_value(self._feature_coverage)
            last_frame_time_s = self._last_frame_time_s
            last_frame_wall_time = self._last_frame_wall_time
            input_contract_status = self._snapshot_runtime_value(
                self._input_contract_status
            )
            startup_phase = self._startup_phase
            timeout_reason = self._timeout_reason
            replay_metrics = self._snapshot_runtime_value(self._replay_metrics)
            preview_started_at = self._preview_started_at
            current_fault_prob = self._current_fault_prob
            risk_score = self._risk_score
            risk_trend = self._risk_trend
            warning_stage = self._warning_stage
            warning_reason = self._warning_reason
            top_recommendations = (
                self._snapshot_runtime_value(self._top_recommendations) or []
            )
            preview_lead_time_s = self._last_preview_lead_time_s
            if (
                preview_lead_time_s is None
                and preview_started_at is not None
                and last_frame_time_s is not None
            ):
                preview_lead_time_s = round(
                    max(0.0, last_frame_time_s - preview_started_at), 1
                )
            return {
                "is_collecting": self.is_collecting,
                "is_detecting": self.is_detecting,
                "is_blind_test": self._blind_test_active,
                "current_session_id": self._session_id,
                "uptime_seconds": round(uptime, 1),
                "planned_duration_s": self._planned_duration_s,
                "frame_count": self._frame_count,
                "fault_events_count": len(self._fault_events)
                + (1 if preview_alert is not None and not fault_events else 0),
                "fault_events": fault_events,
                "latest_diagnoses": [item["diagnosis"] for item in fault_events]
                if fault_events
                else ([preview_alert] if preview_alert is not None else []),
                "online_model": self._get_online_model_summary(),
                "transport_status": self._transport_status(),
                "business_status": self._session_business_status(),
                "scenario_backend": self._backend_name,
                "schema_version": self._schema_version,
                "feature_coverage": feature_coverage,
                "last_frame_time_s": last_frame_time_s,
                "last_frame_wall_time": last_frame_wall_time,
                "input_contract_status": input_contract_status,
                "startup_phase": startup_phase,
                "timeout_reason": timeout_reason,
                "replay_metrics": replay_metrics,
                "processes": processes,
                "current_fault_prob": round(float(current_fault_prob or 0.0), 3),
                "risk_score": round(float(risk_score or 0.0), 3),
                "risk_trend": risk_trend,
                "warning_stage": warning_stage,
                "warning_reason": warning_reason,
                "preview_lead_time_s": preview_lead_time_s,
                "top_recommendations": top_recommendations,
            }

    def get_realtime_data(self) -> dict:
        """返回实时缓冲区中的时序数据（下采样到 ~200 点）。"""
        with self._lock:
            buf = list(self._realtime_buffer)

        if not buf:
            return {
                k: []
                for k in [
                    "times",
                    "motor1",
                    "motor2",
                    "motor3",
                    "motor4",
                    "roll",
                    "pitch",
                    "yaw",
                    "pos_z",
                ]
            }

        # 下采样到 ~200 点
        step = max(1, len(buf) // 200)
        sampled = buf[::step]

        return {
            "times": [round(f["time_s"], 3) for f in sampled],
            "motor1": [round(f["motor1"], 4) for f in sampled],
            "motor2": [round(f["motor2"], 4) for f in sampled],
            "motor3": [round(f["motor3"], 4) for f in sampled],
            "motor4": [round(f["motor4"], 4) for f in sampled],
            "roll": [round(f["roll"], 3) for f in sampled],
            "pitch": [round(f["pitch"], 3) for f in sampled],
            "yaw": [round(f["yaw"], 3) for f in sampled],
            "pos_z": [round(f["pos_z"], 3) for f in sampled],
        }

    def _model_summary(self, meta: dict | None) -> dict | None:
        if meta is None:
            return None
        metrics = meta.get("metrics") or {}
        return {
            "name": meta.get("name"),
            "task": meta.get("task"),
            "architecture": meta.get("architecture"),
            "num_classes": meta.get("num_classes"),
            "label_names": meta.get("label_names") or [],
            "window_size": meta.get("window_size"),
            "feature_count": meta.get("feature_count"),
            "metrics": {
                "macro_f1": metrics.get("macro_f1"),
                "accuracy": metrics.get("accuracy"),
                "precision": metrics.get("precision"),
                "recall": metrics.get("recall"),
                "f1_fault": metrics.get("f1_fault"),
                "best_val_f1": metrics.get("best_val_f1"),
            },
        }

    def _resolve_online_model_meta(
        self, requested_model: str | None = None
    ) -> dict | None:
        if not self.app:
            return None
        return resolve_registered_model(
            self.app.config,
            requested_name=requested_model,
            task="online_binary",
            capability="realtime_detection",
        )

    def _get_online_model_summary(self) -> dict | None:
        active = self._active_model
        if active is not None:
            return dict(active)
        return self._model_summary(self._resolve_online_model_meta())

    # ── 盲测模式 ──────────────────────────────────────────────

    def start_blind_test(
        self, drone_id: str | None = None, sim_time: float = 120.0
    ) -> dict:
        """启动盲测：随机决定故障配置，不泄露 ground truth。"""
        if self.is_collecting:
            raise RuntimeError("采集已在运行中，请先停止当前会话")

        # 随机生成故障配置（none 20%, motor_failure 25%, thrust_loss 25%, prop_imbalance 15%, battery_failure 15%）
        rng = random.Random()
        roll = rng.random()
        if roll < 0.20:
            fault_type = "none"
            fault_config = {
                "fault_type": "none",
                "description": "正常飞行（无故障注入）",
            }
        elif roll < 0.45:
            fault_type = "motor_failure"
            motor_index = rng.randint(1, 4)
            effectiveness = round(rng.uniform(0.0, 0.4), 2)
            fault_config = {
                "fault_type": "motor_failure",
                "motor_index": motor_index,
                "effectiveness": effectiveness,
                "description": f"电机{motor_index}失效，效率={effectiveness}",
            }
        elif roll < 0.70:
            fault_type = "thrust_loss"
            scale = round(rng.uniform(0.4, 0.65), 2)
            fault_config = {
                "fault_type": "thrust_loss",
                "scale": scale,
                "description": f"推力损失，MPC_THR_MAX={scale}",
            }
        elif roll < 0.85:
            fault_type = "prop_imbalance"
            motor_index = rng.randint(1, 4)
            amp = round(rng.uniform(0.05, 0.15), 2)
            freq = round(rng.uniform(8.0, 15.0), 1)
            fault_config = {
                "fault_type": "prop_imbalance",
                "motor_index": motor_index,
                "vibration_amp": amp,
                "vibration_freq": freq,
                "description": f"螺旋桨{motor_index}不平衡，振幅={amp}, 频率={freq}Hz",
            }
        else:
            fault_type = "battery_failure"
            drain = round(rng.uniform(15.0, 30.0), 1)
            fault_config = {
                "fault_type": "battery_failure",
                "battery_drain": drain,
                "description": f"电池故障，放电时间={drain}s（正常60s）",
            }

        fault_delay = round(rng.uniform(15, 30), 1)
        fault_config["fault_delay_s"] = fault_delay

        with self._lock:
            self._blind_test_active = True
            self._blind_test_ground_truth = fault_config

        print(
            f"[telemetry_manager] 盲测模式：{fault_config['description']}，延迟 {fault_delay}s"
        )

        try:
            result = self.start_collection(
                drone_id=drone_id, session_type="blind_test", sim_time=sim_time
            )
        except Exception:
            with self._lock:
                self._blind_test_active = False
                self._blind_test_ground_truth = None
            raise
        return {"session_id": result["session_id"], "status": "blind_test_running"}

    def get_blind_test_reveal(self) -> dict:
        """揭晓盲测结果，仅停止后可调用。"""
        if not self._blind_test_active:
            raise RuntimeError("当前没有盲测会话")
        if self.is_collecting:
            raise RuntimeError("盲测仍在进行中，请先停止采集")

        gt = self._blind_test_ground_truth or {}
        gt_fault = gt.get("fault_type", "none")
        gt_delay = gt.get("fault_delay_s", 0)

        # 分析检测结果
        detected = len(self._fault_events) > 0
        first_detection_time = None
        if self._fault_events:
            first_detection_time = self._fault_events[0].start_t

        detection_result = {
            "detected": detected,
            "fault_events": [self._event_to_dict(e) for e in self._fault_events],
            "fault_events_count": len(self._fault_events),
            "first_detection_time": round(first_detection_time, 2)
            if first_detection_time
            else None,
        }

        # 判定
        if gt_fault == "none":
            verdict = "correct_normal" if not detected else "false_alarm"
        else:
            verdict = "hit" if detected else "miss"

        detection_latency = None
        if verdict == "hit" and first_detection_time is not None:
            detection_latency = round(first_detection_time - gt_delay, 2)

        evaluation = {
            "verdict": verdict,
            "detection_latency_s": detection_latency,
        }

        # 重置盲测状态
        with self._lock:
            self._blind_test_active = False
            self._blind_test_ground_truth = None

        return {
            "ground_truth": gt,
            "detection_result": detection_result,
            "evaluation": evaluation,
        }

    # ── 启动采集 ──────────────────────────────────────────────

    def start_collection(
        self,
        drone_id: str | None = None,
        session_type: str = "simulation",
        sim_time: float = 120.0,
        backend: str = "px4",
        replay_path: str | None = None,
    ) -> dict:
        if self.is_collecting:
            raise RuntimeError("采集已在运行中，请先停止当前会话")
        if backend == "px4":
            telemetry_python_bin = self._resolve_telemetry_python_bin()
            print(f"[telemetry_manager] 遥测子进程解释器: {telemetry_python_bin}")
        sim_time = float(sim_time)
        if sim_time <= 0:
            raise RuntimeError("采集时长必须大于 0 秒")

        session_id = _generate_id("TS")
        csv_path = str(self._sessions_dir / f"{session_id}.csv")
        backend_name = (backend or "px4").strip().lower()
        backend_options = {"replay_path": replay_path}
        backend_cls = {"px4": PX4ScenarioBackend, "replay": ReplayScenarioBackend}.get(
            backend_name
        )
        if backend_cls is None:
            raise RuntimeError(f"不支持的 backend: {backend_name}")
        backend_instance = backend_cls(
            self,
            session_id=session_id,
            csv_path=csv_path,
            sim_time=sim_time,
            options=backend_options,
        )
        try:
            backend_instance.prepare()
        except Exception:
            with self._lock:
                self._reset_runtime_state()
                self._realtime_buffer.clear()
            raise

        with self._lock:
            self._activate_runtime_state(
                session_id=session_id,
                sim_time=sim_time,
                backend_name=backend_name,
                backend_options=backend_options,
                backend_instance=backend_instance,
                session_type=session_type,
            )

        self._create_session_record(
            session_id,
            drone_id,
            session_type,
            csv_path,
            backend_name,
            source_path=replay_path,
            meta={"backend": backend_name},
        )
        self._write_log(drone_id, "telemetry", f"开始遥测数据采集，会话 {session_id}")
        self._realtime_buffer.clear()
        try:
            self._backend.start()
        except Exception:
            self._delete_session_record(session_id)
            with self._lock:
                self._reset_runtime_state()
            raise
        return {
            "session_id": session_id,
            "status": "running",
            "csv_path": serialize_local_path(csv_path, self.app.config),
            "planned_duration_s": sim_time,
            "scenario_backend": backend_name,
            "replay_path": serialize_local_path(replay_path, self.app.config),
        }

    def stop_collection(self) -> dict:
        with self._stop_condition:
            if self._stop_in_progress:
                while self._stop_in_progress:
                    self._stop_condition.wait()
                if self._last_stop_error is not None:
                    raise self._last_stop_error
                if self._last_stop_result is not None:
                    return dict(self._last_stop_result)
            if (
                not self.is_collecting
                and self._backend is None
                and self._px4_proc is None
            ):
                raise RuntimeError("当前没有运行中的采集会话")
            self._stop_in_progress = True
            self._last_stop_result = None
            self._last_stop_error = None
        session_id = self._session_id
        try:
            if self._detecting:
                self.stop_detection()
            stop_error = None
            if self._backend is not None:
                try:
                    self._backend.stop()
                except Exception as e:
                    stop_error = e
            self._replay_metrics = self._collect_replay_metrics(session_id)
            duration = time.time() - self._start_time if self._start_time else 0
            result = {
                "session_id": session_id,
                "duration_seconds": round(duration, 1),
                "frame_count": self._frame_count,
                "fault_events_count": len(self._fault_events),
                "scenario_backend": self._backend_name,
                "replay_metrics": self._replay_metrics,
            }
            self._finalize_session(
                session_id,
                duration,
                self._frame_count,
                len(self._fault_events),
                self._replay_metrics,
            )
            self._write_log(
                None,
                "telemetry",
                f"遥测采集完成，会话 {session_id}，共 {self._frame_count} 帧，时长 {duration:.1f}s",
            )
            with self._stop_condition:
                self._last_stop_result = dict(result)
            with self._lock:
                self._reset_runtime_state()
                self._realtime_buffer.clear()
            if stop_error is not None:
                raise stop_error
            return result
        except Exception as e:
            with self._stop_condition:
                self._last_stop_error = e
            raise
        finally:
            with self._stop_condition:
                self._stop_in_progress = False
                self._stop_condition.notify_all()

    def start_detection(self, requested_model: str | None = None) -> dict:
        if not self.is_collecting:
            raise RuntimeError("请先启动数据采集")
        if self._detecting:
            raise RuntimeError("实时检测已在运行中")
        requested_model = (
            str(requested_model).strip() or None
            if requested_model is not None
            else None
        )
        model_meta = self._resolve_online_model_meta(requested_model=requested_model)
        if model_meta is None:
            if requested_model:
                raise RuntimeError(f"未找到可用在线模型: {requested_model}")
            raise RuntimeError(
                "未找到可用在线模型，请先配置 DEFAULT_ONLINE_MODEL 或补齐模型目录"
            )
        try:
            runner_cls = self._get_detection_runner_cls()
            self._detector = runner_cls(model_meta["path"], device="cpu")
            self._detecting = True
            self._startup_phase = "detecting"
            self._active_model = self._model_summary(model_meta)
            self._update_session_detection(self._session_id, True)
            self._write_log(
                None,
                "realtime_detection",
                f"启动实时故障检测，关联会话 {self._session_id}",
            )
            return {"detector_status": "running", "model": self._active_model}
        except Exception as exc:
            self._detector = None
            self._detecting = False
            self._active_model = None
            if self._startup_phase == "detecting":
                self._startup_phase = "streaming"
            raise RuntimeError(
                f"在线检测启动失败: {model_meta.get('name') or model_meta.get('path')} - {exc}"
            ) from exc

    def stop_detection(self) -> dict:
        if not self._detecting:
            raise RuntimeError("实时检测未在运行")
        self._detecting = False
        if self._detector:
            for event in self._detector.flush():
                self._record_fault_event(event)
        self._replay_metrics = self._collect_replay_metrics(self._session_id)
        fault_events = [self._event_to_dict(e) for e in self._fault_events[-10:]]
        count = len(self._fault_events)
        self._write_log(
            None, "realtime_detection", f"停止实时检测，共检出 {count} 个故障事件"
        )
        self._detector = None
        return {
            "detector_status": "stopped",
            "fault_events_count": count,
            "latest_diagnoses": [item["diagnosis"] for item in fault_events],
            "model": self._active_model,
            "replay_metrics": self._replay_metrics,
        }

    def _on_backend_frame(
        self, frame: dict, window: deque, source_backend: str | None = None
    ) -> None:
        backend_name = source_backend or self._backend_name
        canonical = frame
        detect_window = list(window)
        coverage = None
        if backend_name == "px4":
            canonical, coverage = adapt_px4_live_frame(frame)
            detect_window = [adapt_px4_live_frame(item)[0] for item in window]
        elif backend_name == "replay":
            canonical, coverage = canonicalize_rfly_frame(frame)
            detect_window = [canonicalize_rfly_frame(item)[0] for item in window]
        with self._lock:
            self._frame_count += 1
            self._last_frame_time_s = canonical.get("time_s", self._frame_count / 50.0)
            self._last_frame_wall_time = time.time()
            self._startup_phase = "streaming"
            if coverage is not None:
                self._feature_coverage = {
                    "mapped_count": len(coverage.get("mapped", [])),
                    "derived_count": len(coverage.get("derived", [])),
                    "default_count": len(coverage.get("default", [])),
                    "mapped": coverage.get("mapped", []),
                    "derived": coverage.get("derived", []),
                    "default": coverage.get("default", []),
                }
                default_count = len(coverage.get("default", []))
                self._input_contract_status = {
                    "status": "degraded" if default_count > 0 else "healthy",
                    "reason": "missing_features" if default_count > 0 else "ok",
                }
            self._realtime_buffer.append(
                {
                    "time_s": canonical.get("time_s", self._frame_count / 50.0),
                    "motor1": canonical.get("actuator_out_1", frame.get("motor1", 0)),
                    "motor2": canonical.get("actuator_out_2", frame.get("motor2", 0)),
                    "motor3": canonical.get("actuator_out_3", frame.get("motor3", 0)),
                    "motor4": canonical.get("actuator_out_4", frame.get("motor4", 0)),
                    "roll": canonical.get("roll", frame.get("roll", 0)),
                    "pitch": canonical.get("pitch", frame.get("pitch", 0)),
                    "yaw": canonical.get("yaw", frame.get("yaw", 0)),
                    "pos_z": canonical.get(
                        "pos_z",
                        canonical.get(
                            "baro_alt",
                            frame.get("baro_alt_meter", frame.get("pos_z", 0)),
                        ),
                    ),
                }
            )
        if not self._detecting or self._detector is None:
            return
        if len(detect_window) < self._detector.window_size:
            return
        result, new_events = self._detector.update(
            frame=canonical, window=detect_window, frame_idx=self._frame_count
        )
        if result is None:
            return
        fault_prob = float(result.get("fault_prob", 0.0))
        onset_th = float(
            getattr(getattr(self._detector, "aggregator", None), "onset_th", 0.6)
        )
        if new_events:
            if (
                self._preview_started_at is not None
                and canonical.get("time_s") is not None
            ):
                self._last_preview_lead_time_s = round(
                    max(
                        0.0,
                        float(canonical.get("time_s"))
                        - float(self._preview_started_at),
                    ),
                    1,
                )
            self._preview_alert = None
            self._preview_started_at = None
        elif fault_prob >= onset_th:
            if self._preview_started_at is None:
                self._preview_started_at = canonical.get("time_s")
            self._preview_alert = build_diagnosis_payload(
                source="realtime_preview",
                label=str(result.get("label") or "fault"),
                confidence=result.get("confidence"),
                severity=canonical_severity_from_probability(fault_prob),
                recommendations=[],
                model=self._get_online_model_summary(),
                time_range=build_time_range(
                    start_time=canonical.get("time_s"),
                    end_time=canonical.get("time_s"),
                ),
                evidence_summary=self._build_online_evidence_summary(
                    label=str(result.get("label") or "fault"),
                    severity=canonical_severity_from_probability(fault_prob),
                    probability=fault_prob,
                    start_time=canonical.get("time_s"),
                    end_time=canonical.get("time_s"),
                    lead_time_s=None,
                ),
            )
        else:
            self._preview_alert = None
            self._preview_started_at = None
        risk_state = self._risk_evaluator.update(
            fault_prob=fault_prob,
            has_preview=self._preview_alert is not None,
            has_confirmed=bool(new_events),
        )
        decision_signal = DecisionSignal(
            ts=float(canonical.get("time_s") or self._frame_count / 50.0),
            frame_idx=int(self._frame_count),
            fault_prob=fault_prob,
            label=str(result.get("label") or "fault"),
            confidence=result.get("confidence"),
            risk_score=risk_state["risk_score"],
            risk_trend=risk_state["risk_trend"],
            warning_stage=risk_state["warning_stage"],
        )
        self._decision_engine.ingest_signal(
            decision_signal,
            onset_th=onset_th,
            online_model=self._get_online_model_summary(),
        )
        self._current_fault_prob = risk_state["current_fault_prob"]
        self._risk_score = risk_state["risk_score"]
        self._risk_trend = risk_state["risk_trend"]
        self._warning_stage = risk_state["warning_stage"]
        self._warning_reason = risk_state["warning_reason"]
        self._top_recommendations = self._recommend_actions_for_stage(
            self._warning_stage
        )
        for event in new_events:
            self._decision_engine.confirm_event(
                event,
                ts=float(canonical.get("time_s") or getattr(event, "end_t", 0.0)),
            )
            event.lead_time_s = self._last_preview_lead_time_s
            self._record_fault_event(event)

    def _record_fault_event(self, event) -> None:
        with self._lock:
            self._fault_events.append(event)
        diagnosis = self._build_realtime_diagnosis(
            probability=event.max_fault_prob,
            start_time=event.start_t,
            end_time=event.end_t,
            lead_time_s=self._last_preview_lead_time_s,
        )
        self._persistence.record_fault_event(
            session_id=self._session_id,
            event=event,
            diagnosis=diagnosis,
            legacy_severity=legacy_severity_from_canonical(diagnosis["severity"]),
        )

    def _receiver_worker(self) -> None:
        try:
            self._receiver.start()
        except Exception as e:
            self._startup_phase = "receiver_timeout"
            self._timeout_reason = "receiver_timeout"
            print(f"[telemetry_manager] Receiver 异常退出: {e}")
        finally:
            if self._session_id and self._start_time:
                threading.Thread(target=self._auto_finalize, daemon=True).start()

    def _auto_finalize(self) -> None:
        time.sleep(1)
        if self._session_id and not self.is_collecting:
            try:
                self.stop_collection()
            except Exception:
                pass

    def _start_px4(self) -> None:
        try:
            result = subprocess.run(
                ["pgrep", "-f", "px4_sitl_default/bin/px4"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode == 0:
                subprocess.run(
                    ["pkill", "-9", "-f", "px4_sitl_default/bin/px4"],
                    timeout=5,
                    capture_output=True,
                )
                subprocess.run(
                    ["pkill", "-9", "-f", "gz sim"], timeout=5, capture_output=True
                )
                time.sleep(3)
        except Exception:
            pass
        env = os.environ.copy()
        env["HEADLESS"] = "1"
        px4_sys_autostart = "4001"
        px4_sim_target = "gz_x500_lawn"
        px4_root = self._get_px4_root()
        if self.app:
            px4_sys_autostart = str(
                self.app.config.get("PX4_SYS_AUTOSTART", px4_sys_autostart)
            )
            px4_sim_target = str(self.app.config.get("PX4_SIM_TARGET", px4_sim_target))
        env["PX4_SYS_AUTOSTART"] = px4_sys_autostart
        px4_make = f"make -C {px4_root} px4_sitl {px4_sim_target}"
        self._px4_proc = subprocess.Popen(
            ["bash", "-c", px4_make],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            env=env,
            **self._popen_session_kwargs(),
        )
        print(f"[telemetry_manager] PX4 SITL 启动中 (PID={self._px4_proc.pid})")

    def _wait_for_px4(self, timeout: int = 90) -> bool:
        deadline = time.time() + timeout
        while time.time() < deadline:
            if self._px4_proc is not None and self._px4_proc.poll() is not None:
                print("[telemetry_manager] PX4 进程已退出")
                return False
            try:
                with socket.create_connection(("127.0.0.1", 18570), timeout=2):
                    return True
            except (ConnectionRefusedError, OSError):
                time.sleep(2)
        return False

    def _spawn_process(self, cmd: list[str], *, env: dict | None = None):
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            cwd=str(self._src_dir),
            env=env,
            **self._popen_session_kwargs(),
        )
        print(f"[telemetry_manager] 子进程已启动 PID={proc.pid}: {' '.join(cmd[:3])}")
        return proc

    def _stop_process(self, proc: subprocess.Popen | None, name: str) -> None:
        if proc is None or proc.poll() is not None:
            return
        try:
            if os.name == "nt":
                proc.terminate()
                proc.wait(timeout=10)
            else:
                os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
                proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            if os.name == "nt":
                proc.kill()
                proc.wait(timeout=5)
            else:
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
                proc.wait(timeout=5)
        except Exception as e:
            print(f"[telemetry_manager] 停止 {name} 失败: {e}")

    def _proc_status(self, proc: subprocess.Popen | None) -> str | None:
        if proc is None:
            return None
        return "running" if proc.poll() is None else "exited"

    @staticmethod
    def _event_to_dict(event) -> dict:
        diagnosis = event.get_diagnosis() if hasattr(event, "get_diagnosis") else None
        if not isinstance(diagnosis, dict) or not diagnosis:
            diagnosis = build_diagnosis_payload(
                source="realtime",
                label="fault",
                confidence=getattr(event, "max_fault_prob", None),
                severity=canonical_severity_from_probability(
                    getattr(event, "max_fault_prob", None)
                ),
                recommendations=[],
                model=None,
                time_range=build_time_range(
                    start_time=getattr(
                        event, "start_t", getattr(event, "start_time", None)
                    ),
                    end_time=getattr(event, "end_t", getattr(event, "end_time", None)),
                    duration=getattr(event, "duration", None),
                ),
                lead_time_s=getattr(event, "lead_time_s", None),
                evidence_summary={
                    "source": "online",
                    "fault_label": "fault",
                    "severity": {
                        "level": canonical_severity_from_probability(
                            getattr(event, "max_fault_prob", None)
                        ),
                        "score": getattr(event, "max_fault_prob", None),
                        "method": "online_probability_threshold",
                    },
                    "top_evidence": [
                        {
                            "kind": "signal",
                            "name": "fault_probability",
                            "value": getattr(event, "max_fault_prob", None),
                            "status": "observed",
                        },
                        {
                            "kind": "window",
                            "name": "lead_time_s",
                            "value": getattr(event, "lead_time_s", None),
                            "status": "derived",
                        },
                    ],
                },
            )
        time_range = diagnosis.get("time_range") or {}
        severity = diagnosis.get("severity") or canonical_severity_from_probability(
            getattr(event, "max_fault_prob", None)
        )
        return {
            "id": getattr(event, "id", None),
            "start_time": time_range.get("start_time"),
            "end_time": time_range.get("end_time"),
            "duration": time_range.get("duration"),
            "max_fault_prob": round(float(getattr(event, "max_fault_prob", 0.0)), 3),
            "mean_fault_prob": round(float(getattr(event, "mean_fault_prob", 0.0)), 3),
            "data_points": getattr(
                event, "n_points", getattr(event, "data_points", None)
            ),
            "severity": severity,
            "severity_legacy": legacy_severity_from_canonical(severity),
            "diagnosis": diagnosis,
            "created_at": getattr(event, "created_at", None).strftime(
                "%Y-%m-%d %H:%M:%S"
            )
            if getattr(event, "created_at", None)
            else None,
        }

    def _create_session_record(
        self,
        session_id,
        drone_id,
        session_type,
        csv_path,
        backend_name,
        source_path=None,
        meta=None,
    ):
        self._persistence.create_session_record(
            session_id=session_id,
            drone_id=drone_id,
            session_type=session_type,
            csv_path=csv_path,
            backend_name=backend_name,
            schema_version=self._schema_version,
            source_path=source_path,
            meta=meta,
        )

    def _delete_session_record(self, session_id):
        self._persistence.delete_session_record(session_id=session_id)

    def _finalize_session(
        self, session_id, duration, frame_count, fault_count, replay_metrics=None
    ):
        self._persistence.finalize_session(
            session_id=session_id,
            duration=duration,
            frame_count=frame_count,
            fault_count=fault_count,
            replay_metrics=replay_metrics,
        )

    def _update_session_detection(self, session_id, has_detection):
        self._persistence.update_session_detection(
            session_id=session_id,
            has_detection=has_detection,
        )

    def _write_log(self, drone_id, log_type, content):
        self._persistence.write_log(
            drone_id=drone_id,
            log_type=log_type,
            content=content,
        )


# 全局单例
telemetry_manager = TelemetryProcessManager()
