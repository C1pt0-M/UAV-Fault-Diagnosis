from __future__ import annotations

import os
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
