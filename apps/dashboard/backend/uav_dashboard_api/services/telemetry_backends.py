from __future__ import annotations

import os
import csv
import signal
import shutil
import socket
import subprocess
import sys
import threading
import time
from abc import ABC, abstractmethod
from collections import deque
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

from uav_fault.telemetry.receiver import TelemetryReceiver
DEFAULT_RFLY_PX4MAVCTRL_PATH = Path(
    r"D:\PX4PSP\RflySimAPIs\RflySimSDK\ctrl\PX4MavCtrlV4.py"
)

RFLY_INFERENCE_LOCATIONS = {"center", "mec"}
_KEEP_LIFECYCLE_VALUE = object()


from uav_fault.telemetry.schema_adapter import (
    CANONICAL_SCHEMA_VERSION,
    canonicalize_rfly_frame,
    normalize_time_columns,
)


@dataclass(frozen=True)
class TransportLifecycleSnapshot:
    state: str = "created"
    phase: str = "created"
    failure_reason: str | None = None
    detail: str | None = None
    updated_at: float = 0.0


class ScenarioBackend(ABC):
    name = "base"
    schema_version = CANONICAL_SCHEMA_VERSION
    ready_on_start = True

    def __init__(
        self, manager, *, session_id: str, csv_path: str, sim_time: float, options: dict
    ):
        self.manager = manager
        self.session_id = session_id
        self.csv_path = csv_path
        self.sim_time = sim_time
        self.options = options
        self._lifecycle_lock = threading.RLock()
        self._lifecycle = TransportLifecycleSnapshot(updated_at=time.time())
        self._receiver: TelemetryReceiver | None = None
        self._receiver_thread: threading.Thread | None = None
        self._processing_started = False

    def lifecycle(self) -> TransportLifecycleSnapshot:
        with self._lifecycle_lock:
            return self._lifecycle

    def _transition(
        self,
        *,
        state: str | None = None,
        phase: str | None = None,
        failure_reason: str | None | object = _KEEP_LIFECYCLE_VALUE,
        detail: str | None | object = _KEEP_LIFECYCLE_VALUE,
    ) -> TransportLifecycleSnapshot:
        with self._lifecycle_lock:
            self._lifecycle = replace(
                self._lifecycle,
                state=state or self._lifecycle.state,
                phase=phase or self._lifecycle.phase,
                failure_reason=(
                    self._lifecycle.failure_reason
                    if failure_reason is _KEEP_LIFECYCLE_VALUE
                    else failure_reason
                ),
                detail=(
                    self._lifecycle.detail
                    if detail is _KEEP_LIFECYCLE_VALUE
                    else detail
                ),
                updated_at=time.time(),
            )
            snapshot = self._lifecycle
        callback = getattr(self.manager, "on_transport_lifecycle", None)
        if callable(callback):
            callback(snapshot, self.session_id, self)
        return snapshot

    def report_failure(self, reason: str, exc: Exception | None = None) -> None:
        self._transition(
            state="failed",
            phase=reason,
            failure_reason=reason,
            detail=str(exc) if exc is not None else None,
        )

    @staticmethod
    def _ensure_process_alive(process, label: str) -> None:
        poll = getattr(process, "poll", None)
        if callable(poll) and poll() is not None:
            raise RuntimeError(f"{label} 启动后提前退出")

    @staticmethod
    def _process_status(process: subprocess.Popen | None) -> str | None:
        if process is None:
            return None
        return "running" if process.poll() is None else "exited"

    @staticmethod
    def _session_process_kwargs() -> dict:
        if os.name == "nt":
            creationflags = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
            return {"creationflags": creationflags} if creationflags else {}
        return {"start_new_session": True}

    def _spawn_process(
        self,
        command: list[str],
        *,
        cwd: str | Path | None = None,
        env: dict | None = None,
        stdout=None,
        stderr=None,
    ) -> subprocess.Popen:
        process = subprocess.Popen(
            command,
            cwd=str(cwd) if cwd is not None else None,
            env=env,
            stdout=subprocess.DEVNULL if stdout is None else stdout,
            stderr=subprocess.DEVNULL if stderr is None else stderr,
            **self._session_process_kwargs(),
        )
        print(
            f"[telemetry_backend:{self.name}] 子进程已启动 "
            f"PID={process.pid}: {' '.join(command[:3])}"
        )
        return process

    @staticmethod
    def _stop_process(process: subprocess.Popen | None, name: str) -> None:
        if process is None or process.poll() is not None:
            return
        try:
            if os.name == "nt":
                ctrl_break = getattr(signal, "CTRL_BREAK_EVENT", None)
                if ctrl_break is not None:
                    process.send_signal(ctrl_break)
                else:
                    process.terminate()
                process.wait(timeout=15)
            else:
                os.killpg(os.getpgid(process.pid), signal.SIGTERM)
                process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            if os.name == "nt":
                subprocess.run(
                    ["taskkill", "/PID", str(process.pid), "/T", "/F"],
                    capture_output=True,
                    timeout=10,
                )
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
            else:
                os.killpg(os.getpgid(process.pid), signal.SIGKILL)
                process.wait(timeout=5)
        except Exception as exc:
            print(f"[telemetry_backend] 停止 {name} 失败: {exc}")

    def _start_receiver(
        self, receiver: TelemetryReceiver, *, thread_name: str
    ) -> None:
        self._receiver = receiver
        try:
            self.manager.start_transport_processing()
            self._processing_started = True
            receiver.callbacks.append(self._submit_receiver_frame)
            thread = threading.Thread(
                target=self._receiver_worker,
                name=thread_name,
                daemon=True,
            )
            self._receiver_thread = thread
            thread.start()
        except Exception:
            self._receiver_thread = None
            receiver.stop()
            self._receiver = None
            if self._processing_started:
                self.manager.stop_transport_processing(drain=False)
                self._processing_started = False
            raise

    def _submit_receiver_frame(self, frame: dict, window) -> None:
        self.manager.submit_transport_frame(
            frame,
            window,
            source_backend=self.name,
            session_id=self.session_id,
            backend=self,
        )

    def _receiver_worker(self) -> None:
        receiver = self._receiver
        if receiver is None:
            return
        try:
            receiver.start()
            if (
                getattr(receiver, "exit_reason", None) == "timeout"
                and self.lifecycle().state not in {"stopping", "stopped"}
            ):
                self.report_failure("receiver_timeout")
        except Exception as exc:
            if self.lifecycle().state not in {"stopping", "stopped"}:
                self.report_failure("receiver_timeout", exc)
            print(f"[telemetry_backend:{self.name}] Receiver 异常退出: {exc}")
        finally:
            self.manager.on_transport_finished(self.session_id, self)

    def _stop_receiver(self) -> None:
        receiver = self._receiver
        thread = self._receiver_thread
        self._receiver = None
        if receiver is not None:
            receiver.stop()
        try:
            if (
                thread is not None
                and thread.is_alive()
                and threading.current_thread() is not thread
            ):
                thread.join(timeout=10)
                if thread.is_alive():
                    raise RuntimeError("receiver thread 未在 10s 内退出")
        finally:
            if thread is None or not thread.is_alive():
                self._receiver_thread = None
            if self._processing_started:
                self.manager.stop_transport_processing(drain=True)
                self._processing_started = False

    def is_collecting(self) -> bool:
        thread = self._receiver_thread
        return thread is not None and thread.is_alive()

    def prepare(self) -> None:
        if self.lifecycle().state != "created":
            raise RuntimeError(
                f"{self.name} transport cannot prepare from {self.lifecycle().state}"
            )
        self._transition(
            state="preparing",
            phase=f"preparing_{self.name}",
            failure_reason=None,
            detail=None,
        )
        try:
            self._prepare()
        except Exception as exc:
            self.report_failure(f"{self.name}_prepare_failed", exc)
            raise
        self._transition(state="prepared", phase=f"{self.name}_prepared")

    def _prepare(self) -> None:
        return None

    def start(self) -> None:
        if self.lifecycle().state == "created":
            self.prepare()
        if self.lifecycle().state != "prepared":
            raise RuntimeError(
                f"{self.name} transport cannot start from {self.lifecycle().state}"
            )
        self._transition(
            state="starting",
            phase=f"starting_{self.name}",
            failure_reason=None,
            detail=None,
        )
        try:
            self._start()
        except Exception as exc:
            if self.lifecycle().state != "failed":
                self.report_failure(f"{self.name}_start_failed", exc)
            raise
        if self.ready_on_start and self.lifecycle().state == "starting":
            self._transition(state="running", phase=f"{self.name}_running")

    @abstractmethod
    def _start(self) -> None:
        raise NotImplementedError

    def stop(self) -> None:
        if self.lifecycle().state == "stopped":
            return
        if self.lifecycle().state == "created":
            self._transition(state="stopped", phase=f"{self.name}_stopped")
            return
        self._transition(state="stopping", phase=f"stopping_{self.name}")
        try:
            self._stop()
        except Exception as exc:
            self.report_failure(f"{self.name}_stop_failed", exc)
            raise
        self._transition(state="stopped", phase=f"{self.name}_stopped")

    @abstractmethod
    def _stop(self) -> None:
        raise NotImplementedError

    def status(self) -> dict[str, Any]:
        payload = dict(self._status())
        payload["transport_lifecycle"] = asdict(self.lifecycle())
        return payload

    def session_meta(self) -> dict:
        """返回需要写入会话 meta 的额外信息，默认无。"""
        return {}

    @abstractmethod
    def _status(self) -> dict:
        raise NotImplementedError


class PX4ScenarioBackend(ScenarioBackend):
    name = "px4"
    ready_on_start = False

    def __init__(
        self, manager, *, session_id: str, csv_path: str, sim_time: float, options: dict
    ) -> None:
        super().__init__(
            manager,
            session_id=session_id,
            csv_path=csv_path,
            sim_time=sim_time,
            options=options,
        )
        self._startup_cancel = threading.Event()
        self._startup_lock = threading.RLock()
        self._startup_thread: threading.Thread | None = None
        self._startup_generation = 0
        self._px4_proc: subprocess.Popen | None = None
        self._collector_proc: subprocess.Popen | None = None
        self._flight_proc: subprocess.Popen | None = None

    def _config(self) -> dict:
        return self.manager.app.config if self.manager.app else {}

    def _repo_root(self) -> Path:
        return Path(__file__).resolve().parents[5]

    def _get_px4_root(self) -> str:
        configured = str(self._config().get("PX4_ROOT", "") or "").strip()
        if configured:
            return os.path.expanduser(configured)
        return os.path.expanduser(os.environ.get("PX4_ROOT", "~/PX4-Autopilot"))

    def _build_collector_cmd(self, telemetry_python_bin: str) -> list[str]:
        cfg = self._config()
        command = [
            telemetry_python_bin,
            "-m",
            "uav_fault.telemetry.collector",
            "--host",
            str(cfg.get("TELEMETRY_FORWARD_HOST", "127.0.0.1")),
            "--port",
            str(cfg.get("TELEMETRY_FORWARD_PORT", 15550)),
            "--emit-rate",
            str(cfg.get("TELEMETRY_EMIT_RATE_HZ", 50)),
            "--mavsdk-address",
            str(cfg.get("TELEMETRY_MAVSDK_ADDRESS", "udpin://0.0.0.0:14550")),
            "--grpc-port",
            str(cfg.get("TELEMETRY_MAVSDK_GRPC_PORT", 50052)),
            "--pymav-address",
            str(cfg.get("TELEMETRY_PYMAV_ADDRESS", "udpin:0.0.0.0:14030")),
        ]
        enable_pymav = cfg.get("TELEMETRY_ENABLE_PYMAV", True)
        if isinstance(enable_pymav, str):
            enable_pymav = enable_pymav.strip().lower() in {
                "1",
                "true",
                "yes",
                "on",
            }
        if not enable_pymav:
            command.append("--no-pymav")
        return command

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
                    ["pkill", "-9", "-f", "gz sim"],
                    timeout=5,
                    capture_output=True,
                )
                time.sleep(3)
        except Exception:
            pass
        cfg = self._config()
        env = os.environ.copy()
        env["HEADLESS"] = "1"
        env["PX4_SYS_AUTOSTART"] = str(cfg.get("PX4_SYS_AUTOSTART", "4001"))
        px4_sim_target = str(cfg.get("PX4_SIM_TARGET", "gz_x500_lawn"))
        px4_make = f"make -C {self._get_px4_root()} px4_sitl {px4_sim_target}"
        self._px4_proc = self._spawn_process(
            ["bash", "-c", px4_make],
            env=env,
        )
        print(f"[telemetry_backend:px4] PX4 SITL 启动中 (PID={self._px4_proc.pid})")

    def _wait_for_px4(self, *, timeout: int, generation: int) -> bool:
        cfg = self._config()
        ready_host = str(cfg.get("PX4_READY_HOST", "127.0.0.1"))
        ready_port = int(cfg.get("PX4_READY_PORT", 18570))
        deadline = time.time() + timeout
        while time.time() < deadline:
            if self._startup_cancelled(generation):
                return False
            if self._px4_proc is not None and self._px4_proc.poll() is not None:
                print("[telemetry_backend:px4] PX4 进程已退出")
                return False
            try:
                with socket.create_connection((ready_host, ready_port), timeout=2):
                    return True
            except (ConnectionRefusedError, OSError):
                self._startup_cancel.wait(2)
        return False

    def _start(self) -> None:
        self._transition(phase="starting_px4")
        with self._startup_lock:
            self._startup_cancel.clear()
            self._startup_generation += 1
            generation = self._startup_generation
        self._start_px4()
        try:
            receiver = TelemetryReceiver(
                port=int(self._config().get("TELEMETRY_FORWARD_PORT", 15550)),
                csv_path=self.csv_path,
                window_size=150,
                auto_stop=5.0,
                recv_timeout=60.0,
            )
            self._start_receiver(receiver, thread_name="telemetry-receiver")
            self._startup_thread = threading.Thread(
                target=self._start_pipeline_delayed,
                args=(generation,),
                name="px4-transport-startup",
                daemon=True,
            )
            self._startup_thread.start()
        except Exception:
            with self._startup_lock:
                self._startup_cancel.set()
                self._startup_generation += 1
            self._cleanup_pipeline_processes()
            self._stop_receiver()
            self._stop_process(self._px4_proc, "px4")
            self._px4_proc = None
            raise

    def _startup_cancelled(self, generation: int) -> bool:
        with self._startup_lock:
            return (
                self._startup_cancel.is_set()
                or generation != self._startup_generation
                or self.lifecycle().state not in {"starting", "running"}
            )

    def _cleanup_pipeline_processes(self) -> None:
        with self._startup_lock:
            flight = self._flight_proc
            collector = self._collector_proc
            self._flight_proc = None
            self._collector_proc = None
        self._stop_process(flight, "flight_controller")
        self._stop_process(collector, "collector")

    def _start_pipeline_delayed(self, generation: int) -> None:
        try:
            telemetry_python_bin = self.manager.resolve_telemetry_python_bin()
            if self._startup_cancelled(generation):
                return
            if not self._wait_for_px4(timeout=90, generation=generation):
                if not self._startup_cancelled(generation):
                    self.report_failure("px4_not_ready")
                    print("[telemetry_manager] PX4 启动超时，跳过后续流程")
                return
            if self._startup_cancelled(generation):
                return
            env = os.environ.copy()
            pythonpath_dirs = [
                self._repo_root(),
                self._repo_root() / "packages" / "uav_fault",
            ]
            env["PYTHONPATH"] = os.pathsep.join(
                [str(path) for path in pythonpath_dirs if path.exists()]
                + [env.get("PYTHONPATH", "")]
            )
            self._transition(phase="starting_collector")
            collector = self._spawn_process(
                self._build_collector_cmd(telemetry_python_bin),
                cwd=self._repo_root(),
                env=env,
            )
            self._ensure_process_alive(collector, "collector")
            with self._startup_lock:
                if self._startup_cancelled(generation):
                    keep_collector = False
                else:
                    self._collector_proc = collector
                    keep_collector = True
            if not keep_collector:
                self._stop_process(collector, "collector")
                return
            self._transition(phase="starting_flight_controller")
            flight = self._spawn_process(
                self._flight_cmd(telemetry_python_bin),
                cwd=self._repo_root(),
                env=env,
            )
            self._ensure_process_alive(flight, "flight controller")
            with self._startup_lock:
                if self._startup_cancelled(generation):
                    keep_flight = False
                else:
                    self._flight_proc = flight
                    keep_flight = True
            if not keep_flight:
                self._stop_process(flight, "flight_controller")
                self._cleanup_pipeline_processes()
                return
            self._transition(state="running", phase="flight_controller_running")
            if self.manager._blind_test_active:
                time.sleep(5)
                if self._startup_cancelled(generation):
                    return
                try:
                    self.manager.start_detection()
                except Exception as exc:
                    print(f"[telemetry_manager] 盲测自动启动检测失败: {exc}")
        except RuntimeError as exc:
            if not self._startup_cancelled(generation):
                reason = (
                    "telemetry_python_unavailable"
                    if self.lifecycle().phase == "starting_px4"
                    else "px4_pipeline_failed"
                )
                self.report_failure(reason, exc)
                print(f"[telemetry_manager] {exc}")
            self._cleanup_pipeline_processes()
        except Exception as exc:
            if not self._startup_cancelled(generation):
                self.report_failure("px4_pipeline_failed", exc)
                print(f"[telemetry_manager] PX4 pipeline 启动失败: {exc}")
            self._cleanup_pipeline_processes()

    def _flight_cmd(self, telemetry_python_bin: str) -> list[str]:
        cmd = [
            telemetry_python_bin,
            "-m",
            "uav_fault.telemetry.flight_controller",
            "--sim-time",
            str(self.sim_time),
        ]
        gt = self.manager._blind_test_ground_truth or {}
        fault_type = gt.get("fault_type", "none")
        cmd += ["--fault-type", fault_type]
        if fault_type != "none":
            cmd += ["--fault-delay", str(gt.get("fault_delay_s", 20))]
        if fault_type == "motor_failure":
            cmd += [
                "--fault-scale",
                str(gt.get("effectiveness", 0.1)),
                "--motor-index",
                str(gt.get("motor_index", 1)),
            ]
        elif fault_type == "thrust_loss":
            cmd += ["--fault-scale", str(gt.get("scale", 0.6))]
        elif fault_type == "prop_imbalance":
            cmd += [
                "--motor-index",
                str(gt.get("motor_index", 1)),
                "--vibration-amp",
                str(gt.get("vibration_amp", 0.1)),
                "--vibration-freq",
                str(gt.get("vibration_freq", 10.0)),
            ]
        elif fault_type == "battery_failure":
            cmd += ["--battery-drain", str(gt.get("battery_drain", 20.0))]
        return cmd

    def _stop(self) -> None:
        with self._startup_lock:
            self._startup_cancel.set()
            self._startup_generation += 1
        self._cleanup_pipeline_processes()
        self._stop_receiver()
        self._stop_process(self._px4_proc, "px4")
        self._px4_proc = None
        if (
            self._startup_thread is not None
            and self._startup_thread.is_alive()
            and threading.current_thread() is not self._startup_thread
        ):
            self._startup_thread.join(timeout=1.0)
        if self._startup_thread is not None and not self._startup_thread.is_alive():
            self._startup_thread = None

    def _status(self) -> dict:
        return {
            "px4_sitl": self._process_status(self._px4_proc),
            "collector": self._process_status(self._collector_proc),
            "flight_controller": self._process_status(self._flight_proc),
            "receiver": "running" if self.is_collecting() else None,
        }


def _as_bool(value, default: bool = False) -> bool:
    if value is None:
        return bool(default)
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


class RflyScenarioBackend(ScenarioBackend):
    name = "rfly"

    def __init__(
        self, manager, *, session_id: str, csv_path: str, sim_time: float, options: dict
    ):
        super().__init__(
            manager,
            session_id=session_id,
            csv_path=csv_path,
            sim_time=sim_time,
            options=options,
        )
        self._bridge_proc = None
        self._px4mavctrl_path: Path | None = None
        self._bridge_script: Path | None = None
        self._bridge_log_handle = None
        self._bridge_log_name: str | None = None
        self._mec_proc = None
        self._mec_log_handle = None
        self._mec_log_name: str | None = None
        self._mec_metrics_path: Path | None = None
        self._mec_model_meta: dict | None = None
        self._inference_location = self._resolve_inference_location()
        self._simulation_started_by_backend = False
        self._simulation_log_name: str | None = None

    def _config(self) -> dict:
        return self.manager.app.config if self.manager.app else {}

    def _repo_root(self) -> Path:
        cfg = self._config()
        return Path(cfg.get("REPO_ROOT") or Path(__file__).resolve().parents[5])

    def _resolve_path(self, raw_value: str | Path, *, base_dir: Path) -> Path:
        path = Path(str(raw_value)).expanduser()
        return path if path.is_absolute() else (base_dir / path).resolve()

    def _resolve_px4mavctrl_path(self) -> Path:
        cfg = self._config()
        raw_value = (
            self.options.get("px4mavctrl_path")
            or cfg.get("RFLY_PX4MAVCTRL_PATH")
            or ""
        )
        raw_value = str(raw_value).strip()
        if not raw_value:
            path = DEFAULT_RFLY_PX4MAVCTRL_PATH
        else:
            path = self._resolve_path(raw_value, base_dir=self._repo_root())
        if not path.exists():
            raise RuntimeError(f"RflySim 官方控制脚本不存在: {path}")
        return path

    def _resolve_bridge_script(self) -> Path:
        cfg = self._config()
        raw_value = str(cfg.get("RFLY_BRIDGE_SCRIPT") or "").strip()
        if raw_value:
            path = self._resolve_path(raw_value, base_dir=self._repo_root())
        elif self._bridge_mode() == "buaa":
            path = self._repo_root() / "packages" / "rfly_bridge" / "rfly_bridge" / "run_buaa_realtime_bridge.py"
        else:
            path = self._repo_root() / "packages" / "rfly_bridge" / "rfly_bridge" / "run_rfly_live_bridge.py"
        if not path.exists():
            raise RuntimeError(f"RflySim 桥接脚本不存在: {path}")
        return path

    def _bridge_mode(self) -> str:
        mode = str(self._config().get("RFLY_BRIDGE_MODE") or "attach").strip().lower()
        return mode if mode in {"attach", "buaa"} else "attach"

    def _manual_qgc_mode(self) -> bool:
        if self._bridge_mode() == "attach":
            return True
        return not _as_bool(self._config().get("RFLY_BUAA_AUTO_CONTROL"), False)

    def _auto_start_simulation(self) -> bool:
        return self._bridge_mode() == "attach" and _as_bool(
            self._config().get("RFLY_AUTO_START_SIM"), True
        )

    def _powershell_bin(self) -> str:
        configured = str(self._config().get("RFLY_POWERSHELL_BIN") or "").strip()
        if configured:
            return configured
        return shutil.which("pwsh") or shutil.which("powershell") or "powershell"

    def _resolve_sim_script(self, config_key: str, default_relative: str) -> Path:
        raw_value = str(self._config().get(config_key) or "").strip()
        path = (
            self._resolve_path(raw_value, base_dir=self._repo_root())
            if raw_value
            else self._repo_root() / default_relative
        )
        if not path.exists():
            raise RuntimeError(f"RflySim 启停脚本不存在: {path}")
        return path

    def _build_sim_start_cmd(self) -> list[str]:
        cfg = self._config()
        script = self._resolve_sim_script(
            "RFLY_SIM_START_SCRIPT", "scripts/start_rflysim_sitl_qgc.ps1"
        )
        return [
            self._powershell_bin(),
            "-NoProfile",
            "-ExecutionPolicy",
            "Bypass",
            "-File",
            str(script),
            "-VehicleNum",
            "1",
            "-StartIndex",
            str(int(cfg.get("RFLY_TEST_ID", 1))),
            "-Frame",
            str(cfg.get("RFLY_SITL_FRAME") or "iris"),
            "-Map",
            str(cfg.get("RFLY_SIM_MAP") or "LowGPU"),
            "-UdpSimMode",
            str(int(cfg.get("RFLY_UDP_MODE", 2))),
            "-QgcUdpPort",
            str(int(cfg.get("RFLY_QGC_UDP_PORT", 14600))),
            "-DashboardPymavUdpPort",
            str(int(cfg.get("RFLY_DASHBOARD_PYMAV_UDP_PORT", 14031))),
        ]

    def _simulation_log(self):
        runtime_dir = self._config().get("RUNTIME_DIR")
        if not runtime_dir:
            return subprocess.DEVNULL
        log_dir = Path(runtime_dir) / "telemetry" / "rfly_sim"
        log_dir.mkdir(parents=True, exist_ok=True)
        path = log_dir / f"{self.session_id}.log"
        self._simulation_log_name = path.name
        return path.open("a", encoding="utf-8")

    def _start_simulation(self) -> None:
        if not self._auto_start_simulation():
            return
        self._transition(phase="starting_rfly_simulation")
        log_handle = self._simulation_log()
        try:
            completed = subprocess.run(
                self._build_sim_start_cmd(),
                stdout=log_handle,
                stderr=subprocess.STDOUT,
                cwd=str(self._repo_root()),
                env=self._bridge_env(),
                timeout=float(self._config().get("RFLY_SIM_START_TIMEOUT_S", 180.0)),
                check=False,
            )
            if completed.returncode != 0:
                raise RuntimeError(
                    f"RflySim LowGPU 启动脚本退出码: {completed.returncode}"
                )
            self._simulation_started_by_backend = True
            wait_s = float(self._config().get("RFLY_SIM_ATTACH_WAIT_S", 2.0))
            if wait_s > 0:
                time.sleep(wait_s)
            self._transition(phase="rfly_simulation_running")
        finally:
            if log_handle is not subprocess.DEVNULL:
                log_handle.close()

    def _stop_simulation(self) -> None:
        if not self._simulation_started_by_backend:
            return
        script = self._resolve_sim_script(
            "RFLY_SIM_STOP_SCRIPT", "scripts/stop_rflysim_sitl.ps1"
        )
        try:
            subprocess.run(
                [
                    self._powershell_bin(),
                    "-NoProfile",
                    "-ExecutionPolicy",
                    "Bypass",
                    "-File",
                    str(script),
                ],
                cwd=str(self._repo_root()),
                env=self._bridge_env(),
                timeout=float(self._config().get("RFLY_SIM_STOP_TIMEOUT_S", 45.0)),
                check=False,
                capture_output=True,
            )
        finally:
            self._simulation_started_by_backend = False

    def _receiver_timeout_s(self) -> float:
        if self._manual_qgc_mode():
            return 0.0
        return float(self._config().get("RFLY_RECEIVER_TIMEOUT_S", 60.0))

    def _python_bin(self) -> str:
        cfg = self._config()
        return str(
            cfg.get("RFLY_PYTHON_BIN")
            or cfg.get("TELEMETRY_PYTHON_BIN")
            or cfg.get("ML_PYTHON_BIN")
            or sys.executable
        )

    def _forward_port(self) -> int:
        return int(self._config().get("RFLY_FORWARD_PORT", 15550))

    def _forward_host(self) -> str:
        return str(self._config().get("RFLY_FORWARD_HOST", "127.0.0.1"))

    def _resolve_inference_location(self) -> str:
        cfg = self._config()
        location = str(
            self.options.get("inference_location")
            or cfg.get("RFLY_INFERENCE_LOCATION")
            or "center"
        ).strip().lower()
        if location not in RFLY_INFERENCE_LOCATIONS:
            raise RuntimeError("RflySim 推理位置仅支持 center 或 mec")
        return location

    def _mec_enabled(self) -> bool:
        return self._inference_location == "mec"

    def _mec_listen_port(self) -> int:
        return int(self._config().get("RFLY_MEC_LISTEN_PORT", 15552))

    def _mec_forward_port(self) -> int:
        return int(self._config().get("RFLY_MEC_FORWARD_PORT", self._forward_port()))

    def _upstream_forward_port(self) -> int:
        return self._mec_listen_port() if self._mec_enabled() else self._forward_port()

    def _bridge_forward_port(self) -> int:
        return self._upstream_forward_port()

    def _resolve_mec_script(self) -> Path:
        cfg = self._config()
        raw_value = str(cfg.get("RFLY_MEC_SCRIPT") or "").strip()
        if raw_value:
            path = self._resolve_path(raw_value, base_dir=self._repo_root())
        else:
            path = (
                self._repo_root()
                / "packages"
                / "uav_fault"
                / "uav_fault"
                / "telemetry"
                / "mec_service.py"
            )
        if not path.exists():
            raise RuntimeError(f"MEC 服务脚本不存在: {path}")
        return path

    def _prepare(self) -> None:
        if self._bridge_mode() == "buaa":
            self._px4mavctrl_path = None
        else:
            self._px4mavctrl_path = self._resolve_px4mavctrl_path()
        self._bridge_script = self._resolve_bridge_script()
        self._inference_location = self._resolve_inference_location()
        if self._mec_enabled():
            self._resolve_mec_script()
            requested_model = str(self.options.get("inference_model") or "").strip() or None
            self._mec_model_meta = self.manager._resolve_online_model_meta(
                requested_model=requested_model
            )
            if self._mec_model_meta is None:
                raise RuntimeError("MEC 未找到可用在线检测模型")
        if self._auto_start_simulation():
            self._resolve_sim_script(
                "RFLY_SIM_START_SCRIPT", "scripts/start_rflysim_sitl_qgc.ps1"
            )
            self._resolve_sim_script(
                "RFLY_SIM_STOP_SCRIPT", "scripts/stop_rflysim_sitl.ps1"
            )

    def _build_bridge_cmd(self) -> list[str]:
        cfg = self._config()
        if self._bridge_script is None or (
            self._bridge_mode() == "attach" and self._px4mavctrl_path is None
        ):
            self.prepare()
        if self._bridge_mode() == "buaa":
            cmd = [
                self._python_bin(),
                str(self._bridge_script),
                "--autotest-dir",
                str(cfg.get("RFLY_BUAA_AUTOTEST_DIR") or r"D:\BuaaAutoTestPlatForm\AutoTestAPI"),
                "--unique-index",
                str(int(cfg.get("RFLY_BUAA_UNIQUE_INDEX", 1419000188))),
                "--udp-port",
                str(int(cfg.get("RFLY_UDP_PORT", 20100))),
                "--ip",
                str(cfg.get("RFLY_MAV_IP", "127.0.0.1")),
                "--sim-map",
                str(cfg.get("RFLY_SIM_MAP") or "LowGPU"),
                "--host",
                self._forward_host(),
                "--port",
                str(self._bridge_forward_port()),
                "--emit-rate",
                str(int(cfg.get("RFLY_EMIT_RATE_HZ", 50))),
                "--rpm-scale",
                str(float(cfg.get("RFLY_RPM_SCALE", 10000.0))),
                "--pymav-address",
                str(cfg.get("RFLY_PYMAV_ADDRESS", "udpin:0.0.0.0:14031")),
                "--startup-wait",
                str(float(cfg.get("RFLY_BUAA_STARTUP_WAIT_S", 25.0))),
                "--ready-timeout",
                str(float(cfg.get("RFLY_BUAA_READY_TIMEOUT_S", 90.0))),
                "--duration",
                "0.0" if self._manual_qgc_mode() else str(float(self.sim_time)),
            ]
            if not _as_bool(cfg.get("RFLY_ENABLE_PYMAV"), True):
                cmd.append("--no-pymav")
            if not _as_bool(cfg.get("RFLY_BUAA_AUTO_CONTROL"), False):
                cmd.append("--manual-control")
            return cmd
        cmd = [
            self._python_bin(),
            str(self._bridge_script),
            "--px4mavctrl",
            str(self._px4mavctrl_path),
            "--host",
            self._forward_host(),
            "--port",
            str(self._bridge_forward_port()),
            "--emit-rate",
            str(int(cfg.get("RFLY_EMIT_RATE_HZ", 50))),
            "--rpm-scale",
            str(float(cfg.get("RFLY_RPM_SCALE", 10000.0))),
            "--test-id",
            str(int(cfg.get("RFLY_TEST_ID", 1))),
            "--udp-port",
            str(int(cfg.get("RFLY_UDP_PORT", 20100))),
            "--udp-mode",
            str(int(cfg.get("RFLY_UDP_MODE", 2))),
            "--mav-ip",
            str(cfg.get("RFLY_MAV_IP", "127.0.0.1")),
            "--pymav-address",
            str(cfg.get("RFLY_PYMAV_ADDRESS", "udpin:0.0.0.0:14031")),
            "--duration",
            "0.0" if self._manual_qgc_mode() else str(float(self.sim_time)),
        ]
        if not _as_bool(cfg.get("RFLY_ENABLE_PYMAV"), True):
            cmd.append("--no-pymav")
        return cmd

    def _bridge_env(self) -> dict:
        env = os.environ.copy()
        paths = [
            str(self._repo_root() / "packages" / "rfly_bridge"),
            str(self._repo_root() / "packages" / "uav_fault"),
            env.get("PYTHONPATH", ""),
        ]
        env["PYTHONPATH"] = os.pathsep.join([item for item in paths if item])
        return env

    def _mec_paths(self) -> tuple[Path | None, Path | None]:
        runtime_dir = self._config().get("RUNTIME_DIR")
        if not runtime_dir:
            return None, None
        log_dir = Path(runtime_dir) / "telemetry" / "mec"
        log_dir.mkdir(parents=True, exist_ok=True)
        return log_dir / f"{self.session_id}.log", log_dir / f"{self.session_id}.csv"

    def _open_mec_log(self):
        log_path, metrics_path = self._mec_paths()
        self._mec_metrics_path = metrics_path
        if log_path is None:
            return subprocess.DEVNULL
        self._mec_log_name = log_path.name
        self._mec_log_handle = log_path.open("a", encoding="utf-8")
        return self._mec_log_handle

    def _build_mec_cmd(self) -> list[str]:
        cfg = self._config()
        if self._mec_model_meta is None:
            self.prepare()
        return [
            self._python_bin(),
            str(self._resolve_mec_script()),
            "--listen-host",
            str(cfg.get("RFLY_MEC_LISTEN_HOST", self._forward_host())),
            "--listen-port",
            str(self._mec_listen_port()),
            "--forward-host",
            str(cfg.get("RFLY_MEC_FORWARD_HOST", self._forward_host())),
            "--forward-port",
            str(self._mec_forward_port()),
            "--model-dir",
            str(self._mec_model_meta["path"]),
            "--window-size",
            str(int(self._mec_model_meta.get("window_size") or 150)),
            "--stride",
            str(int(cfg.get("RFLY_MEC_INFERENCE_STRIDE", 5))),
            "--metrics-csv",
            str(self._mec_metrics_path),
        ]

    def _open_bridge_log(self):
        runtime_dir = self._config().get("RUNTIME_DIR")
        if not runtime_dir:
            return subprocess.DEVNULL
        log_dir = Path(runtime_dir) / "telemetry" / "rfly_bridge"
        log_dir.mkdir(parents=True, exist_ok=True)
        log_path = log_dir / f"{self.session_id}.log"
        self._bridge_log_name = log_path.name
        self._bridge_log_handle = log_path.open("a", encoding="utf-8")
        return self._bridge_log_handle

    def _start(self) -> None:
        self._transition(phase="starting_rfly_bridge")
        receiver = TelemetryReceiver(
            port=self._forward_port(),
            csv_path=self.csv_path,
            window_size=150,
            auto_stop=0,
            recv_timeout=self._receiver_timeout_s(),
            trim_on_stop=False,
        )
        self._start_receiver(
            receiver,
            thread_name="telemetry-rfly-receiver",
        )
        if self._mec_enabled():
            self._transition(phase="starting_mec")
            mec_log = self._open_mec_log()
            try:
                self._mec_proc = self._spawn_process(
                    self._build_mec_cmd(),
                    stdout=mec_log,
                    stderr=mec_log,
                    cwd=str(self._repo_root()),
                    env=self._bridge_env(),
                )
                time.sleep(float(self._config().get("RFLY_MEC_STARTUP_WAIT_S", 0.2)))
                if self._mec_proc.poll() is not None:
                    raise RuntimeError("MEC 服务启动后提前退出")
                self._transition(phase="mec_running")
            except Exception as exc:
                self.report_failure("mec_failed", exc)
                if self._mec_log_handle is not None:
                    self._mec_log_handle.close()
                    self._mec_log_handle = None
                self._stop_receiver()
                raise
        try:
            self._start_simulation()
        except Exception as exc:
            self.report_failure("rfly_simulation_failed", exc)
            self._stop_process(self._mec_proc, "mec")
            self._mec_proc = None
            if self._mec_log_handle is not None:
                self._mec_log_handle.close()
                self._mec_log_handle = None
            self._stop_receiver()
            raise
        bridge_log = self._open_bridge_log()
        try:
            self._bridge_proc = self._spawn_process(
                self._build_bridge_cmd(),
                stdout=bridge_log,
                stderr=bridge_log,
                cwd=str(self._repo_root()),
                env=self._bridge_env(),
            )
            self._ensure_process_alive(self._bridge_proc, "Rfly bridge")
            self._transition(phase="rfly_bridge_running")
        except Exception as exc:
            self.report_failure("rfly_bridge_failed", exc)
            if self._bridge_log_handle is not None:
                self._bridge_log_handle.close()
                self._bridge_log_handle = None
            self._stop_process(self._mec_proc, "mec")
            self._mec_proc = None
            if self._mec_log_handle is not None:
                self._mec_log_handle.close()
                self._mec_log_handle = None
            self._stop_simulation()
            self._stop_receiver()
            raise

    def _stop(self) -> None:
        self._stop_process(self._bridge_proc, "rfly_bridge")
        self._bridge_proc = None
        if self._bridge_log_handle is not None:
            self._bridge_log_handle.close()
            self._bridge_log_handle = None
        self._stop_process(self._mec_proc, "mec")
        self._mec_proc = None
        if self._mec_log_handle is not None:
            self._mec_log_handle.close()
            self._mec_log_handle = None
        self._stop_simulation()
        self._stop_receiver()

    def _status(self) -> dict:
        mec_metrics = None
        if self._mec_metrics_path and self._mec_metrics_path.exists():
            try:
                with self._mec_metrics_path.open(
                    newline="", encoding="utf-8"
                ) as metrics_file:
                    rows = list(csv.DictReader(metrics_file))
                if rows:
                    latest = rows[-1]
                    mec_metrics = {
                        "frames_forwarded": int(
                            float(latest.get("frames_forwarded") or 0)
                        ),
                        "inference_count": int(
                            float(latest.get("inference_count") or 0)
                        ),
                        "inference_queue_drops": int(
                            float(latest.get("inference_queue_drops") or 0)
                        ),
                        "avg_inference_ms": float(
                            latest.get("inference_avg_ms") or 0.0
                        ),
                        "p95_inference_ms": float(
                            latest.get("inference_p95_ms") or 0.0
                        ),
                    }
            except (OSError, ValueError, csv.Error):
                mec_metrics = None
        return {
            "px4_sitl": None,
            "collector": None,
            "rfly_bridge": self._process_status(self._bridge_proc),
            "mec": self._process_status(self._mec_proc)
            if self._mec_enabled()
            else None,
            "receiver": "running" if self.is_collecting() else None,
            "rfly_bridge_log": self._bridge_log_name,
            "mec_log": self._mec_log_name,
            "rfly_sim_log": self._simulation_log_name,
            "mec_metrics_csv": str(self._mec_metrics_path)
            if self._mec_metrics_path is not None
            else None,
            "mec_metrics": mec_metrics,
        }


class ReplayScenarioBackend(ScenarioBackend):
    name = "replay"

    def __init__(
        self, manager, *, session_id: str, csv_path: str, sim_time: float, options: dict
    ):
        super().__init__(
            manager,
            session_id=session_id,
            csv_path=csv_path,
            sim_time=sim_time,
            options=options,
        )
        self.replay_path = str(options.get("replay_path") or "").strip()
        self._thread = None
        self._stop_event = threading.Event()
        self.replay_completed = False

    def _prepare(self) -> None:
        if not self.replay_path:
            raise RuntimeError("replay backend 需要 replay_path")
        if not Path(self.replay_path).exists():
            raise RuntimeError(f"replay 源不存在: {self.replay_path}")

    def _start(self) -> None:
        self._thread = threading.Thread(
            target=self._run, name="telemetry-replay", daemon=True
        )
        self._thread.start()

    def _run(self) -> None:
        try:
            import pandas as pd

            df = normalize_time_columns(pd.read_csv(self.replay_path))
            window = deque(maxlen=150)
            t0 = None
            wall0 = time.time()
            processed = 0
            for row in df.to_dict(orient="records"):
                if self._stop_event.is_set():
                    break
                canonical, _ = canonicalize_rfly_frame(row)
                window.append(canonical)
                self.manager.process_transport_frame(
                    canonical, window, source_backend=self.name
                )
                now_t = float(canonical.get("time_s") or 0.0)
                if t0 is None:
                    t0 = now_t
                delay = max(0.0, (now_t - t0) - (time.time() - wall0))
                if delay > 0:
                    time.sleep(min(delay, 0.05))
                processed += 1
            self.replay_completed = bool(not self._stop_event.is_set())
            if processed > 0:
                self._write_session_csv(df.iloc[:processed])
            self._transition(
                phase=("replay_completed" if self.replay_completed else "replay_stopped")
            )
        except Exception as exc:
            self.report_failure("replay_failed", exc)
            raise
        finally:
            self.manager.on_transport_finished(self.session_id, self)

    def session_meta(self) -> dict:
        return {"replay_completed": bool(self.replay_completed)}

    def _write_session_csv(self, df) -> None:
        df.to_csv(self.csv_path, index=False)

    def _stop(self) -> None:
        self._stop_event.set()
        if (
            self._thread
            and self._thread.is_alive()
            and threading.current_thread() is not self._thread
        ):
            self._thread.join(timeout=5)

    def is_collecting(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def _status(self) -> dict:
        running = self.is_collecting()
        return {
            "px4_sitl": None,
            "collector": None,
            "receiver": "running" if running else None,
        }
