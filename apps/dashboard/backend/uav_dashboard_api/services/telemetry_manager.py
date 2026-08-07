"""遥测会话协调模块。

通过 transport Adapter 协调采集、处理与检测，同一时间只允许一个会话。
"""

import copy
import json
import os
import queue
import random
import shutil
import subprocess
import sys
import threading
import time
from collections import deque
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[5]
UAV_FAULT_PACKAGE_ROOT = PROJECT_ROOT / "packages" / "uav_fault"
PYTHONPATH_DIRS = [PROJECT_ROOT, UAV_FAULT_PACKAGE_ROOT]
for _path in reversed(PYTHONPATH_DIRS):
    _path_text = str(_path)
    if _path.exists() and _path_text not in sys.path:
        sys.path.insert(0, _path_text)

from uav_dashboard_api.services.diagnosis_contract import (
    build_diagnosis_payload,
    build_time_range,
    canonical_severity_from_probability,
    legacy_severity_from_canonical,
    normalize_recommendations,
)
from uav_dashboard_api.services.model_registry import resolve_registered_model
from uav_dashboard_api.services.telemetry_backends import (
    PX4ScenarioBackend,
    RFLY_INFERENCE_LOCATIONS,
    ReplayScenarioBackend,
    RflyScenarioBackend,
)
from uav_dashboard_api.utils.path_store import resolve_local_path, serialize_local_path
from uav_fault.telemetry.replay_metrics import gt_event_metrics
from uav_fault.telemetry.schema_adapter import (
    CANONICAL_SCHEMA_VERSION,
    assess_canonical_input_contract,
    adapt_px4_live_frame,
    canonicalize_rfly_frame,
)
from uav_fault.online.event_aggregator import AggregatedEvent
from uav_fault.online.monitoring_session import OnlineMonitoringSession

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
        created_by=None,
    ) -> None:
        if not self.manager.app:
            return
        try:
            with self.manager.app.app_context():
                from uav_dashboard_api import db
                from uav_dashboard_api.models.telemetry_session import TelemetrySession

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
                    created_by=str(created_by) if created_by is not None else None,
                )
                session.set_meta(meta or {})
                db.session.add(session)
                db.session.commit()
        except Exception as e:
            db.session.rollback()
            print(f"[telemetry_manager] 创建会话记录失败: {e}")

    def delete_session_record(self, *, session_id) -> None:
        if not self.manager.app:
            return
        try:
            with self.manager.app.app_context():
                from uav_dashboard_api import db
                from uav_dashboard_api.models.telemetry_session import TelemetrySession

                session = db.session.get(TelemetrySession, session_id)
                if session:
                    db.session.delete(session)
                    db.session.commit()
        except Exception as e:
            db.session.rollback()
            print(f"[telemetry_manager] 删除会话记录失败: {e}")

    def finalize_session(
        self,
        *,
        session_id,
        duration,
        frame_count,
        fault_count,
        replay_metrics=None,
        fault_prob_curve=None,
        mec=None,
        inference=None,
        session_meta=None,
    ) -> None:
        if not self.manager.app:
            return
        try:
            with self.manager.app.app_context():
                from uav_dashboard_api import db
                from uav_dashboard_api.models.asset import Asset
                from uav_dashboard_api.models.telemetry_session import TelemetrySession

                session = db.session.get(TelemetrySession, session_id)
                if session:
                    meta = session.get_meta()
                    if replay_metrics:
                        meta["replay_metrics"] = replay_metrics
                    if mec:
                        meta["mec"] = mec
                        meta["inference_location"] = mec.get("location")
                    if inference:
                        meta["inference"] = inference
                        meta["inference_location"] = inference.get("location")
                    if session_meta:
                        meta.update(session_meta)
                    if fault_prob_curve:
                        curve_dir = Path(self.manager.app.config["RUNTIME_DIR"]) / "telemetry_fault_prob"
                        curve_dir.mkdir(parents=True, exist_ok=True)
                        curve_path = curve_dir / f"{session_id}.json"
                        curve_path.write_text(
                            json.dumps(fault_prob_curve, ensure_ascii=False),
                            encoding="utf-8",
                        )
                        meta["fault_prob_curve_path"] = serialize_local_path(
                            curve_path, self.manager.app.config
                        )
                        meta["fault_prob_curve_points"] = len(fault_prob_curve)
                    if (
                        replay_metrics
                        or fault_prob_curve
                        or mec
                        or inference
                    ):
                        session.set_meta(meta)
                    session.end_time = datetime.now()
                    session.duration_seconds = round(duration, 1)
                    session.frame_count = frame_count
                    session.fault_events_count = fault_count
                    session.status = "completed"
                    resolved_csv = resolve_local_path(session.csv_path, self.manager.app.config)
                    if (
                        session.scenario_backend != "replay"
                        and resolved_csv
                        and Path(resolved_csv).exists()
                    ):
                        serialized_csv = serialize_local_path(
                            resolved_csv,
                            self.manager.app.config,
                        )
                        asset = Asset.query.filter_by(stored_path=serialized_csv).first()
                        if asset is None:
                            asset = Asset(
                                filename=Path(resolved_csv).name,
                                kind="csv",
                                stored_path=serialized_csv,
                                size_bytes=Path(resolved_csv).stat().st_size,
                                sha256=f"telemetry-session-{session_id}",
                                created_by=int(session.created_by)
                                if session.created_by and str(session.created_by).isdigit()
                                else None,
                            )
                            db.session.add(asset)
                        else:
                            asset.size_bytes = Path(resolved_csv).stat().st_size
                        asset_meta = asset.get_meta()
                        asset_meta.update(
                            {
                                "prepared_csv_path": serialized_csv,
                                "source_type": "telemetry_session",
                                "session_id": session.id,
                            }
                        )
                        asset.set_meta(asset_meta)
                    db.session.commit()
        except Exception as e:
            db.session.rollback()
            print(f"[telemetry_manager] 更新会话记录失败: {e}")

    def update_session_detection(self, *, session_id, has_detection) -> None:
        if not self.manager.app:
            return
        try:
            with self.manager.app.app_context():
                from uav_dashboard_api import db
                from uav_dashboard_api.models.telemetry_session import TelemetrySession

                session = db.session.get(TelemetrySession, session_id)
                if session:
                    session.has_detection = has_detection
                    db.session.commit()
        except Exception as e:
            db.session.rollback()
            print(f"[telemetry_manager] 更新检测状态失败: {e}")

    def write_log(self, *, drone_id, log_type, content) -> None:
        if not self.manager.app:
            return
        try:
            with self.manager.app.app_context():
                from uav_dashboard_api import db
                from uav_dashboard_api.models.log import Log

                log = Log(
                    id=_generate_id("LOG"),
                    drone_id=drone_id,
                    type=log_type,
                    content=content,
                )
                db.session.add(log)
                db.session.commit()
        except Exception as e:
            db.session.rollback()
            print(f"[telemetry_manager] 写入日志失败: {e}")

    def record_fault_event(
        self, *, session_id, event, diagnosis, legacy_severity
    ) -> None:
        if not self.manager.app:
            return
        try:
            with self.manager.app.app_context():
                from uav_dashboard_api import db
                from uav_dashboard_api.models.log import Log
                from uav_dashboard_api.models.telemetry_session import TelemetryFaultEvent

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
            db.session.rollback()
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
        self._frame_queue: queue.Queue = queue.Queue(maxsize=4096)
        self._inference_queue: queue.Queue = queue.Queue(maxsize=1)
        self._frame_worker_thread: threading.Thread | None = None
        self._inference_worker_thread: threading.Thread | None = None
        self._frame_worker_stop = threading.Event()
        self._inference_worker_stop = threading.Event()
        self._inference_idle = threading.Event()
        self._inference_idle.set()
        self._canonical_window: deque = deque(maxlen=150)
        self._frame_queue_drops = 0
        self._inference_queue_drops = 0
        # 检测
        self._detector = None
        self._detecting = False
        self._detection_callback = None
        self._monitoring_session = OnlineMonitoringSession()
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
        self._fault_prob_curve: list[dict] = []
        # 路径
        self._active_model: dict | None = None
        self._telemetry_python_bin: str | None = None
        self._src_dir = PROJECT_ROOT
        self._pythonpath_dirs = [path for path in PYTHONPATH_DIRS if path.exists()]
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
        self._mec_status = self._initial_mec_status()
        self._inference_status = self._initial_inference_status()
        self._center_inference_samples: deque[float] = deque(maxlen=1000)
        self._center_inference_count = 0
        self._mec_last_inference_count = 0
        self._mec_last_event_seq = 0
        self._current_fault_prob: float = 0.0
        self._risk_score: float = 0.0
        self._risk_trend: str = "stable"
        self._warning_stage: str = "normal"
        self._warning_reason: str = "当前未发现风险演化信号"
        self._preview_started_at: float | None = None
        self._top_recommendations: list[dict] = []
        self._last_preview_lead_time_s: float | None = None
        self._stop_in_progress = False
        self._last_stop_result: dict | None = None
        self._last_stop_error: Exception | None = None
        self._stop_condition = threading.Condition(self._lock)
        self._persistence = TelemetryPersistence(self)
        self._sessions_dir = PROJECT_ROOT / "data" / "telemetry_sessions"
        self._sessions_dir.mkdir(parents=True, exist_ok=True)

    def init_app(self, app):
        if self.app is not None and self.app is not app and (
            self.is_collecting or self._backend is not None
        ):
            raise RuntimeError("遥测管理器仍有活动会话，不能切换 Flask app")
        self.app = app
        self._sessions_dir = Path(app.config["RUNTIME_DIR"]) / "telemetry_sessions"
        self._sessions_dir.mkdir(parents=True, exist_ok=True)

    def on_transport_lifecycle(self, snapshot, session_id=None, backend=None) -> None:
        """Receive lifecycle snapshots from the active transport Adapter."""
        with self._lock:
            if (
                backend is not None
                and self._backend is not None
                and self._backend is not backend
            ):
                return
            if (
                session_id is not None
                and self._session_id is not None
                and self._session_id != session_id
            ):
                return
            self._startup_phase = snapshot.phase
            self._timeout_reason = snapshot.failure_reason

    @staticmethod
    def _get_detection_runner_cls():
        global DetectionRunner
        if DetectionRunner is None:
            from uav_dashboard_api.services.detection_runner import DetectionRunner as _DetectionRunner

            DetectionRunner = _DetectionRunner
        return DetectionRunner

    def _get_online_inference_stride(self) -> int:
        if self.app is None:
            return 1
        return max(1, int(self.app.config.get("ONLINE_INFERENCE_STRIDE", 5)))

    def _reset_runtime_state(self) -> None:
        self._session_id = None
        self._start_time = None
        self._planned_duration_s = None
        self._frame_count = 0
        self._fault_events = []
        self._fault_prob_curve = []
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
        self._mec_status = self._initial_mec_status()
        self._inference_status = self._initial_inference_status()
        self._center_inference_samples.clear()
        self._center_inference_count = 0
        self._mec_last_inference_count = 0
        self._mec_last_event_seq = 0
        self._current_fault_prob = 0.0
        self._risk_score = 0.0
        self._risk_trend = "stable"
        self._warning_stage = "normal"
        self._warning_reason = "当前未发现风险演化信号"
        self._preview_started_at = None
        self._top_recommendations = []
        self._last_preview_lead_time_s = None
        self._monitoring_session.reset_collection()
        self._canonical_window.clear()
        self._frame_queue_drops = 0
        self._inference_queue_drops = 0

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
        self._fault_prob_curve = []
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
        inference_location = str(
            backend_options.get("inference_location") or "center"
        ).strip().lower()
        self._mec_status = self._initial_mec_status(
            enabled=backend_name == "rfly" and inference_location == "mec",
            location=inference_location,
            status="waiting"
            if backend_name == "rfly" and inference_location == "mec"
            else "disabled",
        )
        self._inference_status = self._initial_inference_status(
            location=inference_location,
            status="waiting",
        )
        self._center_inference_samples.clear()
        self._center_inference_count = 0
        self._mec_last_inference_count = 0
        self._mec_last_event_seq = 0
        self._current_fault_prob = 0.0
        self._risk_score = 0.0
        self._risk_trend = "stable"
        self._warning_stage = "normal"
        self._warning_reason = "当前未发现风险演化信号"
        self._preview_started_at = None
        self._top_recommendations = []
        self._last_preview_lead_time_s = None
        self._monitoring_session.reset_collection()
        self._canonical_window.clear()
        self._frame_queue_drops = 0
        self._inference_queue_drops = 0
        self._stop_in_progress = False
        self._last_stop_result = None
        self._last_stop_error = None
        self._backend = backend_instance
        if session_type != "blind_test":
            self._blind_test_active = False
            self._blind_test_ground_truth = None

    @staticmethod
    def _clear_work_queue(work_queue: queue.Queue) -> None:
        while True:
            try:
                work_queue.get_nowait()
                work_queue.task_done()
            except queue.Empty:
                return

    def start_transport_processing(self) -> None:
        """Start the processing pipeline used by a live transport Adapter."""
        self._start_live_processing_pipeline()

    def stop_transport_processing(self, *, drain: bool = True) -> None:
        """Stop the processing pipeline owned by this monitoring session."""
        self._stop_live_processing_pipeline(drain=drain)

    def submit_transport_frame(
        self,
        frame: dict,
        _window=None,
        *,
        source_backend: str,
        session_id: str,
        backend,
    ) -> None:
        """Accept a frame only from the currently active transport Adapter."""
        with self._lock:
            if (
                self._backend is not backend
                or self._session_id != session_id
                or self._backend_name != source_backend
            ):
                return
        self._submit_backend_frame(frame, _window, source_backend=source_backend)

    def on_transport_finished(self, session_id: str, backend) -> None:
        """Schedule finalization for the active transport only."""
        with self._lock:
            if self._backend is not backend or self._session_id != session_id:
                return
        threading.Thread(
            target=self._auto_finalize,
            args=(session_id, backend),
            name=f"telemetry-finalize-{session_id}",
            daemon=True,
        ).start()

    def _start_live_processing_pipeline(self) -> None:
        self._clear_work_queue(self._frame_queue)
        self._clear_work_queue(self._inference_queue)
        self._canonical_window.clear()
        self._frame_worker_stop.clear()
        self._inference_worker_stop.clear()
        self._inference_idle.set()
        self._frame_queue_drops = 0
        self._inference_queue_drops = 0
        self._frame_worker_thread = threading.Thread(
            target=self._frame_processing_worker,
            name="telemetry-frame-processing",
            daemon=True,
        )
        self._inference_worker_thread = threading.Thread(
            target=self._inference_worker,
            name="telemetry-model-inference",
            daemon=True,
        )
        self._frame_worker_thread.start()
        self._inference_worker_thread.start()

    def _stop_live_processing_pipeline(self, *, drain: bool = True) -> None:
        if not drain:
            self._clear_work_queue(self._frame_queue)
        self._frame_worker_stop.set()
        if self._frame_worker_thread is not None:
            self._frame_worker_thread.join(timeout=15)
            if self._frame_worker_thread.is_alive():
                print("[telemetry_manager] frame worker 未在 15s 内退出")
            else:
                self._frame_worker_thread = None

        if not drain:
            self._clear_work_queue(self._inference_queue)
        self._inference_worker_stop.set()
        if self._inference_worker_thread is not None:
            self._inference_worker_thread.join(timeout=15)
            if self._inference_worker_thread.is_alive():
                print("[telemetry_manager] inference worker 未在 15s 内退出")
            else:
                self._inference_worker_thread = None

    def _submit_backend_frame(
        self,
        frame: dict,
        _window=None,
        source_backend: str | None = None,
    ) -> None:
        item = (dict(frame), source_backend or self._backend_name)
        try:
            self._frame_queue.put_nowait(item)
        except queue.Full:
            try:
                self._frame_queue.get_nowait()
                self._frame_queue.task_done()
            except queue.Empty:
                pass
            self._frame_queue_drops += 1
            self._frame_queue.put_nowait(item)

    def _frame_processing_worker(self) -> None:
        while not self._frame_worker_stop.is_set() or not self._frame_queue.empty():
            try:
                frame, backend_name = self._frame_queue.get(timeout=0.05)
            except queue.Empty:
                continue
            try:
                canonical, coverage = self._canonicalize_frame(frame, backend_name)
                self._canonical_window.append(canonical)
                self._handle_canonical_frame(
                    frame=frame,
                    canonical=canonical,
                    detect_window=list(self._canonical_window),
                    coverage=coverage,
                    async_inference=True,
                    force_contract_assessment=False,
                )
            except Exception as exc:
                print(f"[telemetry_manager] frame worker 异常: {exc}")
            finally:
                self._frame_queue.task_done()

    def _inference_worker(self) -> None:
        while (
            not self._inference_worker_stop.is_set()
            or not self._inference_queue.empty()
        ):
            try:
                job = self._inference_queue.get(timeout=0.05)
            except queue.Empty:
                continue
            self._inference_idle.clear()
            try:
                if self._detecting and self._detector is not None:
                    self._execute_detection_job(*job, detector=self._detector)
            except Exception as exc:
                print(f"[telemetry_manager] inference worker 异常: {exc}")
            finally:
                self._inference_idle.set()
                self._inference_queue.task_done()

    def _enqueue_inference_job(self, job: tuple) -> None:
        try:
            self._inference_queue.put_nowait(job)
            return
        except queue.Full:
            pass
        try:
            self._inference_queue.get_nowait()
            self._inference_queue.task_done()
        except queue.Empty:
            pass
        self._inference_queue_drops += 1
        self._inference_queue.put_nowait(job)

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

    def resolve_telemetry_python_bin(self) -> str:
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
        backend = self._backend
        if backend is None:
            return False
        probe = getattr(backend, "is_collecting", None)
        return bool(probe()) if callable(probe) else False

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
                pass
        return {
            "px4_sitl": None,
            "collector": None,
            "receiver": "running" if self.is_collecting else None,
        }

    def _transport_status(self) -> str:
        processes = self._backend_process_status()
        if self._timeout_reason:
            return "degraded"
        if self._backend_name == "replay" and self.is_collecting:
            return "connected"
        if (
            self._backend_name == "rfly"
            and self.is_collecting
            and processes.get("rfly_bridge") == "running"
        ):
            if self._mec_status.get("enabled") and processes.get("mec") != "running":
                return "degraded"
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
                from uav_dashboard_api import db
                from uav_dashboard_api.models.telemetry_session import TelemetrySession

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

    def _initial_mec_status(
        self,
        *,
        enabled: bool = False,
        location: str = "center",
        status: str | None = None,
    ) -> dict:
        return {
            "enabled": bool(enabled),
            "location": location,
            "status": status or ("waiting" if enabled else "disabled"),
            "latest_inference_ms": None,
            "avg_inference_ms": None,
            "p95_inference_ms": None,
            "inference_count": 0,
            "frames_forwarded": 0,
            "inference_queue_drops": 0,
            "latest_fault_prob": None,
            "latest_label": None,
            "alert_active": False,
            "contract_status": "waiting",
            "model_status": "loading" if enabled else "disabled",
            "metrics_csv": None,
        }

    def _initial_inference_status(
        self,
        *,
        location: str = "center",
        status: str = "waiting",
    ) -> dict:
        return {
            "location": location,
            "status": status,
            "latest_inference_ms": None,
            "avg_inference_ms": None,
            "p95_inference_ms": None,
            "inference_count": 0,
            "inference_queue_drops": 0,
        }

    @staticmethod
    def _sample_summary(samples) -> tuple[float | None, float | None]:
        ordered = sorted(float(value) for value in samples)
        if not ordered:
            return None, None
        p95_idx = int(0.95 * (len(ordered) - 1))
        return sum(ordered) / len(ordered), ordered[p95_idx]

    def _record_center_inference_metrics(
        self,
        *,
        inference_ms: float,
    ) -> None:
        with self._lock:
            self._center_inference_samples.append(float(inference_ms))
            self._center_inference_count += 1
            avg_inference, p95_inference = self._sample_summary(
                self._center_inference_samples
            )
            self._inference_status.update(
                {
                    "location": "center",
                    "status": "active",
                    "latest_inference_ms": round(float(inference_ms), 3),
                    "avg_inference_ms": round(avg_inference, 3)
                    if avg_inference is not None
                    else None,
                    "p95_inference_ms": round(p95_inference, 3)
                    if p95_inference is not None
                    else None,
                    "inference_count": self._center_inference_count,
                    "inference_queue_drops": int(self._inference_queue_drops),
                }
            )

    def _update_mec_status_locked(self, frame: dict) -> None:
        if not self._mec_status.get("enabled") and not frame.get("mec_enabled"):
            return
        status = self._mec_status
        status["enabled"] = True
        status["location"] = str(
            frame.get("mec_inference_location") or status.get("location") or "mec"
        )
        status["status"] = "active"
        field_map = {
            "mec_inference_ms": "latest_inference_ms",
            "mec_avg_inference_ms": "avg_inference_ms",
            "mec_p95_inference_ms": "p95_inference_ms",
            "mec_inference_count": "inference_count",
            "mec_frames_forwarded": "frames_forwarded",
            "mec_inference_queue_drops": "inference_queue_drops",
            "mec_fault_prob": "latest_fault_prob",
            "mec_label": "latest_label",
            "mec_contract_status": "contract_status",
            "mec_model_status": "model_status",
        }
        for source, target in field_map.items():
            if frame.get(source) is not None:
                status[target] = frame.get(source)
        status["alert_active"] = bool(frame.get("mec_alert_active"))
        inference = self._inference_status
        inference["location"] = "mec"
        inference["status"] = "active"
        inference_field_map = {
            "mec_inference_ms": "latest_inference_ms",
            "mec_avg_inference_ms": "avg_inference_ms",
            "mec_p95_inference_ms": "p95_inference_ms",
            "mec_inference_count": "inference_count",
            "mec_inference_queue_drops": "inference_queue_drops",
        }
        for source, target in inference_field_map.items():
            if frame.get(source) is not None:
                inference[target] = frame.get(source)

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
            mec = self._snapshot_runtime_value(self._mec_status) or {}
            inference = self._snapshot_runtime_value(self._inference_status) or {}
            if mec.get("enabled"):
                proc_status = processes.get("mec")
                if proc_status == "running" and mec.get("status") == "waiting":
                    mec["status"] = "running"
                elif proc_status in {"stopped", "failed"}:
                    mec["status"] = proc_status
                if processes.get("mec_metrics_csv"):
                    mec["metrics_csv"] = processes.get("mec_metrics_csv")
                mec_metrics = processes.get("mec_metrics")
                if mec_metrics:
                    mec.update(mec_metrics)
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
                "mec": mec,
                "inference": inference,
                "processing_pipeline": {
                    "mode": "async_live"
                    if self._frame_worker_thread is not None
                    else "synchronous",
                    "frame_queue_depth": self._frame_queue.qsize(),
                    "frame_queue_drops": self._frame_queue_drops,
                    "inference_queue_depth": self._inference_queue.qsize(),
                    "inference_queue_drops": self._inference_queue_drops,
                    "inference_busy": not self._inference_idle.is_set(),
                },
                "processes": processes,
                "transport_lifecycle": self._snapshot_runtime_value(
                    processes.get("transport_lifecycle")
                ),
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
                    "pos_x",
                    "pos_y",
                    "pos_z",
                    "fault_prob",
                ]
            }

        # 下采样到 ~200 点
        step = max(1, len(buf) // 200)
        sampled = buf[::step]

        def series(field: str, digits: int, default: float = 0.0) -> list[float]:
            values = []
            for frame in sampled:
                raw_value = frame.get(field, default)
                try:
                    values.append(round(float(raw_value), digits))
                except (TypeError, ValueError):
                    values.append(round(float(default), digits))
            return values

        def optional_series(field: str, digits: int) -> list[float | None]:
            values = []
            for frame in sampled:
                try:
                    values.append(round(float(frame[field]), digits))
                except (KeyError, TypeError, ValueError):
                    values.append(None)
            return values

        return {
            "times": series("time_s", 3),
            "motor1": series("motor1", 4),
            "motor2": series("motor2", 4),
            "motor3": series("motor3", 4),
            "motor4": series("motor4", 4),
            "roll": series("roll", 3),
            "pitch": series("pitch", 3),
            "yaw": series("yaw", 3),
            "pos_x": series("pos_x", 3),
            "pos_y": series("pos_y", 3),
            "pos_z": series("pos_z", 3),
            "fault_prob": series("fault_prob", 4),
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
        self, drone_id: str | None = None, sim_time: float = 120.0, created_by=None
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
                drone_id=drone_id, session_type="blind_test", sim_time=sim_time, created_by=created_by
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
        replay_asset_id: int | None = None,
        rfly_options: dict | None = None,
        created_by=None,
    ) -> dict:
        if self.is_collecting:
            raise RuntimeError("采集已在运行中，请先停止当前会话")
        if backend == "px4":
            telemetry_python_bin = self.resolve_telemetry_python_bin()
            print(f"[telemetry_manager] 遥测子进程解释器: {telemetry_python_bin}")
        sim_time = float(sim_time)
        backend_name = (backend or "px4").strip().lower()
        if backend_name == "px4" and sim_time <= 0:
            raise RuntimeError("采集时长必须大于 0 秒")

        session_id = _generate_id("TS")
        csv_path = str(self._sessions_dir / f"{session_id}.csv")
        replay_path_resolved = replay_path
        if backend_name == "replay" and replay_path is not None:
            replay_path_resolved = (
                resolve_local_path(replay_path, self.app.config)
                if self.app
                else str(replay_path).strip()
            )
        backend_options = {"replay_path": replay_path_resolved}
        if backend_name == "rfly":
            rfly_options = dict(rfly_options or {})
            cfg = self.app.config if self.app else {}
            inference_location = str(
                rfly_options.get("inference_location")
                or cfg.get("RFLY_INFERENCE_LOCATION")
                or "center"
            ).strip().lower()
            if inference_location not in RFLY_INFERENCE_LOCATIONS:
                raise RuntimeError("RflySim 推理位置仅支持 center 或 mec")
            rfly_options["inference_location"] = inference_location
            backend_options.update(rfly_options or {})
        backend_cls = {
            "px4": PX4ScenarioBackend,
            "replay": ReplayScenarioBackend,
            "rfly": RflyScenarioBackend,
        }.get(backend_name)
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
            if (
                backend_name == "rfly"
                and getattr(backend_instance, "_manual_qgc_mode", lambda: False)()
            ):
                self._planned_duration_s = None

        self._create_session_record(
            session_id,
            drone_id,
            session_type,
            csv_path,
            backend_name,
            source_path=replay_path_resolved,
            created_by=created_by,
            meta={
                "backend": backend_name,
                "source_asset_id": replay_asset_id,
                "inference_location": backend_options.get("inference_location")
                if backend_name == "rfly"
                else "center",
            },
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
            "planned_duration_s": self._planned_duration_s,
            "scenario_backend": backend_name,
            "replay_path": serialize_local_path(
                replay_path_resolved, self.app.config
            ),
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
            ):
                raise RuntimeError("当前没有运行中的采集会话")
            self._stop_in_progress = True
            self._last_stop_result = None
            self._last_stop_error = None
        session_id = self._session_id
        try:
            stop_error = None
            if self._backend is not None:
                try:
                    self._backend.stop()
                except Exception as e:
                    stop_error = e
            if self._detecting:
                self.stop_detection()
            final_mec = self.get_status().get("mec") or {}
            with self._lock:
                for key, value in final_mec.items():
                    if key in self._mec_status and value is not None:
                        self._mec_status[key] = value
            self._replay_metrics = self._collect_replay_metrics(session_id)
            backend_meta = self._backend.session_meta() if self._backend is not None else {}
            duration = time.time() - self._start_time if self._start_time else 0
            result = {
                "session_id": session_id,
                "duration_seconds": round(duration, 1),
                "frame_count": self._frame_count,
                "fault_events_count": len(self._fault_events),
                "scenario_backend": self._backend_name,
                "replay_metrics": self._replay_metrics,
                "mec": final_mec,
            }
            self._finalize_session(
                session_id,
                duration,
                self._frame_count,
                len(self._fault_events),
                self._replay_metrics,
                session_meta=backend_meta,
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
        mec_enabled = bool(self._mec_status.get("enabled"))
        model_meta = (
            getattr(self._backend, "_mec_model_meta", None)
            if mec_enabled
            else self._resolve_online_model_meta(requested_model=requested_model)
        )
        if (
            mec_enabled
            and requested_model
            and model_meta
            and requested_model != model_meta.get("name")
        ):
            raise RuntimeError("MEC 模型已在采集启动时加载；切换模型请停止采集后重启")
        if model_meta is None:
            if requested_model:
                raise RuntimeError(f"未找到可用在线模型: {requested_model}")
            raise RuntimeError(
                "未找到可用在线模型，请先配置 DEFAULT_ONLINE_MODEL 或补齐模型目录"
            )
        try:
            self._clear_work_queue(self._inference_queue)
            self._inference_idle.wait(timeout=5)
            if mec_enabled:
                self._detector = None
            else:
                runner_cls = self._get_detection_runner_cls()
                self._detector = runner_cls(model_meta["path"], device="cpu")
            self._active_model = self._model_summary(model_meta)
            self._monitoring_session.start_detection(
                self._detector,
                model_summary=self._active_model,
            )
            self._detecting = True
            self._startup_phase = "detecting"
            self._update_session_detection(self._session_id, True)
            self._write_log(
                None,
                "realtime_detection",
                f"启动实时故障检测，关联会话 {self._session_id}",
            )
            return {
                "detector_status": "running",
                "model": self._active_model,
                "inference_location": "mec" if mec_enabled else "center",
            }
        except Exception as exc:
            self._monitoring_session.reset_collection()
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
        self._clear_work_queue(self._inference_queue)
        self._inference_idle.wait(timeout=10)
        if self._monitoring_session.snapshot().state != "running":
            self._monitoring_session.start_detection(
                self._detector,
                model_summary=self._active_model,
            )
        update = self._monitoring_session.stop_detection()
        self._apply_monitoring_update(
            canonical=None,
            frame_idx=None,
            update=update,
        )
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

    def _canonicalize_frame(
        self, frame: dict, backend_name: str | None
    ) -> tuple[dict, dict | None]:
        if backend_name == "px4":
            return adapt_px4_live_frame(frame)
        if backend_name in {"replay", "rfly"}:
            return canonicalize_rfly_frame(frame)
        return frame, None

    def process_transport_frame(
        self, frame: dict, window: deque, source_backend: str | None = None
    ) -> None:
        """Synchronous path retained for replay and focused unit tests."""
        backend_name = source_backend or self._backend_name
        canonical, coverage = self._canonicalize_frame(frame, backend_name)
        if backend_name == "px4":
            detect_window = [adapt_px4_live_frame(item)[0] for item in window]
        elif backend_name in {"replay", "rfly"}:
            detect_window = [canonicalize_rfly_frame(item)[0] for item in window]
        else:
            detect_window = list(window)
        self._handle_canonical_frame(
            frame=frame,
            canonical=canonical,
            detect_window=detect_window,
            coverage=coverage,
            async_inference=False,
            force_contract_assessment=True,
        )

    def _handle_canonical_frame(
        self,
        *,
        frame: dict,
        canonical: dict,
        detect_window: list[dict],
        coverage: dict | None,
        async_inference: bool,
        force_contract_assessment: bool,
    ) -> None:
        with self._lock:
            self._frame_count += 1
            frame_idx = self._frame_count
            self._last_frame_time_s = canonical.get("time_s", frame_idx / 50.0)
            self._last_frame_wall_time = time.time()
            self._startup_phase = "streaming"
            input_contract_status = self._input_contract_status
            if coverage is not None:
                self._feature_coverage = {
                    "mapped_count": len(coverage.get("mapped", [])),
                    "derived_count": len(coverage.get("derived", [])),
                    "default_count": len(coverage.get("default", [])),
                    "proxy_count": len(coverage.get("proxy", [])),
                    "mapped": coverage.get("mapped", []),
                    "derived": coverage.get("derived", []),
                    "default": coverage.get("default", []),
                    "proxy": coverage.get("proxy", []),
                }
                should_assess = len(detect_window) >= 150 and (
                    force_contract_assessment
                    or input_contract_status is None
                    or frame_idx % 50 == 0
                )
                if should_assess:
                    input_contract_status = assess_canonical_input_contract(
                        detect_window,
                        coverage=coverage,
                        actuator_source=canonical.get("actuator_source")
                        or frame.get("actuator_source"),
                    )
                    self._input_contract_status = input_contract_status
            self._update_mec_status_locked(frame)
            self._realtime_buffer.append(
                {
                    "frame_idx": frame_idx,
                    "time_s": canonical.get("time_s", frame_idx / 50.0),
                    "motor1": canonical.get("actuator_out_1", frame.get("motor1", 0)),
                    "motor2": canonical.get("actuator_out_2", frame.get("motor2", 0)),
                    "motor3": canonical.get("actuator_out_3", frame.get("motor3", 0)),
                    "motor4": canonical.get("actuator_out_4", frame.get("motor4", 0)),
                    "roll": canonical.get("roll", frame.get("roll", 0)),
                    "pitch": canonical.get("pitch", frame.get("pitch", 0)),
                    "yaw": canonical.get("yaw", frame.get("yaw", 0)),
                    "pos_x": canonical.get("pos_x", frame.get("pos_x", 0)),
                    "pos_y": canonical.get("pos_y", frame.get("pos_y", 0)),
                    "pos_z": canonical.get(
                        "pos_z",
                        canonical.get(
                            "baro_alt",
                            frame.get("baro_alt_meter", frame.get("pos_z", 0)),
                        ),
                    ),
                    "fault_prob": self._current_fault_prob,
                }
            )
            detector = self._detector
            detecting = self._detecting
            mec_enabled = bool(self._mec_status.get("enabled"))
            mec_inference_count = int(frame.get("mec_inference_count") or 0)
            should_consume_mec = (
                mec_enabled
                and detecting
                and mec_inference_count > self._mec_last_inference_count
                and frame.get("mec_fault_prob") is not None
            )
            if should_consume_mec:
                self._mec_last_inference_count = mec_inference_count
        if should_consume_mec:
            session_snapshot = self._monitoring_session.snapshot()
            if session_snapshot.state != "running":
                session_snapshot = self._monitoring_session.start_detection(
                    None,
                    model_summary=self._active_model,
                )
            event_seq = int(frame.get("mec_event_seq") or 0)
            new_events = []
            if event_seq > self._mec_last_event_seq:
                for item in frame.get("mec_events") or []:
                    new_events.append(
                        AggregatedEvent(
                            start_t=float(item["start_t"]),
                            end_t=float(item["end_t"]),
                            start_frame_idx=item.get("start_frame_idx"),
                            end_frame_idx=item.get("end_frame_idx"),
                            n_points=int(item.get("n_points") or 0),
                            max_fault_prob=float(item.get("max_fault_prob") or 0.0),
                            mean_fault_prob=float(item.get("mean_fault_prob") or 0.0),
                            meta={"inference_location": "mec"},
                        )
                    )
                self._mec_last_event_seq = event_seq
            update = self._monitoring_session.ingest_external(
                result={
                    "fault_prob": float(frame.get("mec_fault_prob") or 0.0),
                    "label": str(frame.get("mec_label") or "normal"),
                    "confidence": float(frame.get("mec_confidence") or 0.0),
                },
                new_events=new_events,
                frame_idx=frame_idx,
                time_s=float(canonical.get("time_s") or frame_idx / 50.0),
                onset_th=0.6,
                generation=session_snapshot.generation,
            )
            self._apply_monitoring_update(
                canonical=canonical,
                frame_idx=frame_idx,
                update=update,
            )
            return
        if not detecting or detector is None:
            return
        if input_contract_status and input_contract_status.get("detection_blocked"):
            return
        if len(detect_window) < detector.window_size:
            return
        if frame_idx % self._get_online_inference_stride() != 0:
            return
        session_snapshot = self._monitoring_session.snapshot()
        if session_snapshot.state != "running":
            session_snapshot = self._monitoring_session.start_detection(
                detector,
                model_summary=self._active_model,
            )
        job = (canonical, detect_window, frame_idx, session_snapshot.generation)
        if async_inference:
            self._enqueue_inference_job(job)
        else:
            self._execute_detection_job(*job, detector=detector)

    def _execute_detection_job(
        self,
        canonical: dict,
        detect_window: list[dict],
        frame_idx: int,
        generation: int,
        *,
        detector,
    ) -> None:
        started = time.perf_counter()
        update = self._monitoring_session.ingest(
            frame=canonical,
            window=detect_window,
            frame_idx=frame_idx,
            generation=generation,
        )
        inference_ms = (time.perf_counter() - started) * 1000.0
        if update.stale:
            return
        self._record_center_inference_metrics(
            inference_ms=inference_ms,
        )
        if update.result is None:
            return
        self._apply_monitoring_update(
            canonical=canonical,
            frame_idx=frame_idx,
            update=update,
        )

    def _apply_monitoring_update(
        self,
        *,
        canonical: dict | None,
        frame_idx: int | None,
        update,
    ) -> None:
        if update.stale:
            return
        snapshot = update.snapshot
        result = update.result
        canonical = canonical or {}
        self._current_fault_prob = snapshot.current_fault_prob
        self._risk_score = snapshot.risk_score
        self._risk_trend = snapshot.risk_trend
        self._warning_stage = snapshot.warning_stage
        self._warning_reason = snapshot.warning_reason
        self._preview_started_at = snapshot.preview_started_at
        self._last_preview_lead_time_s = snapshot.last_preview_lead_time_s
        self._top_recommendations = self._recommend_actions_for_stage(
            snapshot.warning_stage
        )

        if result is not None and frame_idx is not None:
            fault_prob = float(result.get("fault_prob", 0.0))
            frame_time_s = float(canonical.get("time_s") or frame_idx / 50.0)
            with self._lock:
                for buffered in reversed(self._realtime_buffer):
                    if buffered.get("frame_idx") == frame_idx:
                        buffered["fault_prob"] = fault_prob
                        break
                self._fault_prob_curve.append(
                    {
                        "time_s": round(frame_time_s, 4),
                        "fault_prob": round(fault_prob, 6),
                    }
                )

        if result is not None and snapshot.preview_active:
            fault_prob = float(result.get("fault_prob", 0.0))
            frame_time_s = canonical.get("time_s")
            self._preview_alert = build_diagnosis_payload(
                source="realtime_preview",
                label=str(result.get("label") or "fault"),
                confidence=result.get("confidence"),
                severity=canonical_severity_from_probability(fault_prob),
                recommendations=[],
                model=self._get_online_model_summary(),
                time_range=build_time_range(
                    start_time=frame_time_s,
                    end_time=frame_time_s,
                ),
                evidence_summary=self._build_online_evidence_summary(
                    label=str(result.get("label") or "fault"),
                    severity=canonical_severity_from_probability(fault_prob),
                    probability=fault_prob,
                    start_time=frame_time_s,
                    end_time=frame_time_s,
                    lead_time_s=None,
                ),
            )
        elif not snapshot.preview_active:
            self._preview_alert = None

        for event in update.new_events:
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

    def _auto_finalize(self, session_id=None, backend=None) -> None:
        time.sleep(1)
        with self._lock:
            if session_id is not None and self._session_id != session_id:
                return
            if backend is not None and self._backend is not backend:
                return
            should_finalize = bool(self._session_id) and not self.is_collecting
        if should_finalize:
            try:
                self.stop_collection()
            except Exception:
                pass

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
        created_by=None,
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
            created_by=created_by,
        )

    def _delete_session_record(self, session_id):
        self._persistence.delete_session_record(session_id=session_id)

    def _finalize_session(
        self, session_id, duration, frame_count, fault_count, replay_metrics=None, session_meta=None
    ):
        mec = self._snapshot_runtime_value(self._mec_status) or {}
        inference = self._snapshot_runtime_value(self._inference_status) or {}
        self._persistence.finalize_session(
            session_id=session_id,
            duration=duration,
            frame_count=frame_count,
            fault_count=fault_count,
            replay_metrics=replay_metrics,
            fault_prob_curve=list(self._fault_prob_curve),
            mec=mec,
            inference=inference,
            session_meta=session_meta,
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
