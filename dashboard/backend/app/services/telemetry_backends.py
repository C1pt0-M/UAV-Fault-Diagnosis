from __future__ import annotations

import os
import subprocess
import sys
import threading
import time
from abc import ABC, abstractmethod
from collections import deque
from pathlib import Path

import pandas as pd

from telemetry.receiver import TelemetryReceiver
from telemetry.schema_adapter import (
    CANONICAL_SCHEMA_VERSION,
    canonicalize_rfly_frame,
    normalize_time_columns,
)


class ScenarioBackend(ABC):
    name = "base"
    schema_version = CANONICAL_SCHEMA_VERSION

    def __init__(
        self, manager, *, session_id: str, csv_path: str, sim_time: float, options: dict
    ):
        self.manager = manager
        self.session_id = session_id
        self.csv_path = csv_path
        self.sim_time = sim_time
        self.options = options

    def prepare(self) -> None:
        return None

    @abstractmethod
    def start(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def stop(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def status(self) -> dict:
        raise NotImplementedError


class PX4ScenarioBackend(ScenarioBackend):
    name = "px4"

    def start(self) -> None:
        self.manager._startup_phase = "starting_px4"
        self.manager._timeout_reason = None
        self.manager._start_px4()
        self.manager._receiver = TelemetryReceiver(
            port=15550,
            csv_path=self.csv_path,
            window_size=150,
            auto_stop=5.0,
            recv_timeout=60.0,
        )
        self.manager._receiver.callbacks.append(self.manager._on_backend_frame)
        self.manager._receiver_thread = threading.Thread(
            target=self.manager._receiver_worker, name="telemetry-receiver", daemon=True
        )
        self.manager._receiver_thread.start()
        threading.Thread(target=self._start_pipeline_delayed, daemon=True).start()

    def _start_pipeline_delayed(self) -> None:
        try:
            telemetry_python_bin = self.manager._resolve_telemetry_python_bin()
        except RuntimeError as e:
            self.manager._startup_phase = "telemetry_python_unavailable"
            self.manager._timeout_reason = "telemetry_python_unavailable"
            print(f"[telemetry_manager] {e}")
            return
        if not self.manager._wait_for_px4(timeout=90):
            self.manager._startup_phase = "px4_not_ready"
            self.manager._timeout_reason = "px4_not_ready"
            print("[telemetry_manager] PX4 启动超时，跳过后续流程")
            return
        env = os.environ.copy()
        env["PYTHONPATH"] = (
            str(self.manager._src_dir) + os.pathsep + env.get("PYTHONPATH", "")
        )
        self.manager._startup_phase = "starting_collector"
        self.manager._collector_proc = self.manager._spawn_process(
            [
                telemetry_python_bin,
                "-m",
                "telemetry.collector",
                "--host",
                "127.0.0.1",
                "--port",
                "15550",
                "--emit-rate",
                "50",
            ],
            env=env,
        )
        self.manager._startup_phase = "starting_flight_controller"
        self.manager._flight_proc = self.manager._spawn_process(
            self._flight_cmd(telemetry_python_bin), env=env
        )
        if self.manager._blind_test_active:
            time.sleep(5)
            try:
                self.manager.start_detection()
            except Exception as e:
                print(f"[telemetry_manager] 盲测自动启动检测失败: {e}")

    def _flight_cmd(self, telemetry_python_bin: str) -> list[str]:
        cmd = [
            telemetry_python_bin,
            "-m",
            "telemetry.flight_controller",
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

    def stop(self) -> None:
        self.manager._stop_process(self.manager._flight_proc, "flight_controller")
        self.manager._flight_proc = None
        self.manager._stop_process(self.manager._collector_proc, "collector")
        self.manager._collector_proc = None
        if self.manager._receiver:
            try:
                self.manager._receiver._sock.close()
            except Exception:
                pass
        if self.manager._receiver_thread:
            self.manager._receiver_thread.join(timeout=10)
            self.manager._receiver_thread = None
        self.manager._stop_process(self.manager._px4_proc, "px4")
        self.manager._px4_proc = None

    def status(self) -> dict:
        return {
            "px4_sitl": self.manager._proc_status(self.manager._px4_proc),
            "collector": self.manager._proc_status(self.manager._collector_proc),
            "receiver": "running" if self.manager.is_collecting else None,
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

    def _config(self) -> dict:
        return self.manager.app.config if self.manager.app else {}

    def _repo_root(self) -> Path:
        cfg = self._config()
        return Path(cfg.get("REPO_ROOT") or Path(__file__).resolve().parents[4])

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
            raise RuntimeError(
                "RflySim 接入需要配置 RFLY_PX4MAVCTRL_PATH，或在启动采集时传入 rfly_px4mavctrl_path"
            )
        path = self._resolve_path(raw_value, base_dir=self._repo_root())
        if not path.exists():
            raise RuntimeError(f"RflySim 官方控制脚本不存在: {path}")
        return path

    def _resolve_bridge_script(self) -> Path:
        cfg = self._config()
        raw_value = str(cfg.get("RFLY_BRIDGE_SCRIPT") or "").strip()
        if raw_value:
            path = self._resolve_path(raw_value, base_dir=self._repo_root())
        else:
            path = self._repo_root() / "realtime" / "run_rfly_live_bridge.py"
        if not path.exists():
            raise RuntimeError(f"RflySim 桥接脚本不存在: {path}")
        return path

    def _python_bin(self) -> str:
        cfg = self._config()
        return str(
            cfg.get("TELEMETRY_PYTHON_BIN")
            or cfg.get("ML_PYTHON_BIN")
            or sys.executable
        )

    def _forward_port(self) -> int:
        return int(self._config().get("RFLY_FORWARD_PORT", 15550))

    def _forward_host(self) -> str:
        return str(self._config().get("RFLY_FORWARD_HOST", "127.0.0.1"))

    def prepare(self) -> None:
        self._px4mavctrl_path = self._resolve_px4mavctrl_path()
        self._bridge_script = self._resolve_bridge_script()

    def _build_bridge_cmd(self) -> list[str]:
        cfg = self._config()
        if self._px4mavctrl_path is None or self._bridge_script is None:
            self.prepare()
        cmd = [
            self._python_bin(),
            str(self._bridge_script),
            "--px4mavctrl",
            str(self._px4mavctrl_path),
            "--host",
            self._forward_host(),
            "--port",
            str(self._forward_port()),
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
            str(cfg.get("RFLY_PYMAV_ADDRESS", "udpin:0.0.0.0:14030")),
        ]
        if not _as_bool(cfg.get("RFLY_ENABLE_PYMAV"), True):
            cmd.append("--no-pymav")
        return cmd

    def _bridge_env(self) -> dict:
        env = os.environ.copy()
        paths = [
            str(self._repo_root() / "realtime"),
            str(self._repo_root() / "src"),
            env.get("PYTHONPATH", ""),
        ]
        env["PYTHONPATH"] = os.pathsep.join([item for item in paths if item])
        return env

    def start(self) -> None:
        self.manager._startup_phase = "starting_rfly_bridge"
        self.manager._timeout_reason = None
        self.manager._receiver = TelemetryReceiver(
            port=self._forward_port(),
            csv_path=self.csv_path,
            window_size=150,
            auto_stop=0,
            recv_timeout=float(self._config().get("RFLY_RECEIVER_TIMEOUT_S", 60.0)),
            trim_on_stop=False,
        )
        self.manager._receiver.callbacks.append(self.manager._on_backend_frame)
        self.manager._receiver_thread = threading.Thread(
            target=self.manager._receiver_worker,
            name="telemetry-rfly-receiver",
            daemon=True,
        )
        self.manager._receiver_thread.start()
        self._bridge_proc = subprocess.Popen(
            self._build_bridge_cmd(),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            cwd=str(self._repo_root()),
            env=self._bridge_env(),
            **self.manager._popen_session_kwargs(),
        )

    def stop(self) -> None:
        self.manager._stop_process(self._bridge_proc, "rfly_bridge")
        self._bridge_proc = None
        if self.manager._receiver:
            try:
                self.manager._receiver._sock.close()
            except Exception:
                pass
        if self.manager._receiver_thread:
            self.manager._receiver_thread.join(timeout=10)
            self.manager._receiver_thread = None

    def status(self) -> dict:
        return {
            "px4_sitl": None,
            "collector": None,
            "rfly_bridge": self.manager._proc_status(self._bridge_proc),
            "receiver": "running" if self.manager.is_collecting else None,
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

    def prepare(self) -> None:
        if not self.replay_path:
            raise RuntimeError("replay backend 需要 replay_path")
        if not Path(self.replay_path).exists():
            raise RuntimeError(f"replay 源不存在: {self.replay_path}")

    def start(self) -> None:
        self._thread = threading.Thread(
            target=self._run, name="telemetry-replay", daemon=True
        )
        self._thread.start()

    def _run(self) -> None:
        try:
            df = normalize_time_columns(pd.read_csv(self.replay_path))
            self._write_session_csv(df)
            window = deque(maxlen=150)
            t0 = None
            wall0 = time.time()
            for row in df.to_dict(orient="records"):
                if self._stop_event.is_set():
                    break
                canonical, _ = canonicalize_rfly_frame(row)
                window.append(canonical)
                self.manager._on_backend_frame(
                    canonical, window, source_backend=self.name
                )
                now_t = float(canonical.get("time_s") or 0.0)
                if t0 is None:
                    t0 = now_t
                delay = max(0.0, (now_t - t0) - (time.time() - wall0))
                if delay > 0:
                    time.sleep(min(delay, 0.05))
        finally:
            threading.Thread(target=self.manager._auto_finalize, daemon=True).start()

    def _write_session_csv(self, df) -> None:
        df.to_csv(self.csv_path, index=False)

    def stop(self) -> None:
        self._stop_event.set()
        if (
            self._thread
            and self._thread.is_alive()
            and threading.current_thread() is not self._thread
        ):
            self._thread.join(timeout=5)

    def status(self) -> dict:
        running = self._thread is not None and self._thread.is_alive()
        return {
            "px4_sitl": None,
            "collector": None,
            "receiver": "running" if running else None,
        }
