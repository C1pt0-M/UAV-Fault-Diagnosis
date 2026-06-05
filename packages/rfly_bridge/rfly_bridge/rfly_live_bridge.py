from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
import threading
import time
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


from uav_fault.telemetry.forwarder import UDPForwarder
from uav_fault.telemetry.state import TelemetryState


DEFAULT_RFLY_SDK_ROOT = Path(r"D:\PX4PSP\RflySimAPIs\RflySimSDK")


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float(default)
    if not math.isfinite(out):
        return float(default)
    return float(out)


def _get_vector_component(source: dict[str, Any], key: str, index: int) -> float:
    values = source.get(key)
    if isinstance(values, (list, tuple)) and len(values) > index:
        return _safe_float(values[index])
    return 0.0


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _quat_from_euler(roll: float, pitch: float, yaw: float) -> tuple[float, float, float, float]:
    cr = math.cos(roll * 0.5)
    sr = math.sin(roll * 0.5)
    cp = math.cos(pitch * 0.5)
    sp = math.sin(pitch * 0.5)
    cy = math.cos(yaw * 0.5)
    sy = math.sin(yaw * 0.5)
    return (
        cr * cp * cy + sr * sp * sy,
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
    )


class RflyLiveBridge:
    """Bridge official RflySim live state into the project's telemetry JSON contract."""

    MINIMAL_FIELDS = (
        "seq",
        "time_s",
        "motor1",
        "motor2",
        "motor3",
        "motor4",
        "roll",
        "pitch",
        "yaw",
        "rollspeed",
        "pitchspeed",
        "yawspeed",
        "accel_x",
        "accel_y",
        "accel_z",
        "gyro_x",
        "gyro_y",
        "gyro_z",
        "vx",
        "vy",
        "vz",
        "pos_z",
    )

    ENHANCED_FIELDS = (
        "baro_alt_meter",
        "baro_temp",
        "baro_pressure_pa",
        "mag_x",
        "mag_y",
        "mag_z",
        "battery_voltage_v",
        "battery_remaining",
        "actuator_ctrl_1",
        "actuator_ctrl_2",
        "actuator_ctrl_3",
        "actuator_ctrl_4",
        "rates_sp_roll",
        "rates_sp_pitch",
        "rates_sp_yaw",
        "thrust_z",
        "accel_vibration_metric",
        "gyro_vibration_metric",
        "esc_rpm_1",
        "esc_rpm_2",
        "esc_rpm_3",
        "esc_rpm_4",
        "nav_state",
        "battery_warning",
        "sys_status_health",
        "system_status",
        "pymav_highres_imu_seen",
        "pymav_attitude_target_seen",
        "pymav_actuator_control_seen",
    )

    def __init__(
        self,
        *,
        forward_host: str,
        forward_port: int,
        emit_rate_hz: int = 50,
        rpm_scale: float = 10000.0,
    ) -> None:
        self.emit_rate_hz = int(emit_rate_hz)
        self.rpm_scale = float(rpm_scale)
        self._forwarder = UDPForwarder(forward_host, int(forward_port))
        self._start_wall = time.time()
        self._prev_recv_time_s: float | None = None
        self._prev_roll: float | None = None
        self._prev_pitch: float | None = None
        self._prev_yaw: float | None = None
        self._prev_vx: float | None = None
        self._prev_vy: float | None = None
        self._prev_vz: float | None = None

    def close(self) -> None:
        self._forwarder.close()

    def _normalize_motor_rpm(self, rpm: Any) -> float:
        value = _safe_float(rpm)
        if self.rpm_scale <= 1e-6:
            return value
        return max(0.0, min(1.0, value / self.rpm_scale))

    def _time_s_from_state(self, mav_state: dict[str, Any]) -> float:
        true_time = _safe_float(mav_state.get("trueTimeStmp"), default=-1.0)
        if true_time > 1e-6:
            return true_time
        uav_time = _safe_float(mav_state.get("uavTimeStmp"), default=-1.0)
        if uav_time > 1e-6:
            return uav_time
        recv_time = _safe_float(mav_state.get("recv_time_s"), default=-1.0)
        if recv_time >= 0.0:
            return recv_time
        return max(0.0, time.time() - self._start_wall)

    @staticmethod
    def _ctrl_to_motor(value: float) -> float:
        return max(0.0, min(1.0, (value + 1.0) / 2.0))

    @staticmethod
    def _proxy_actuator_outputs(frame: dict[str, float]) -> list[float]:
        ctrl_roll = _safe_float(frame.get("actuator_ctrl_1"), default=0.0)
        ctrl_pitch = _safe_float(frame.get("actuator_ctrl_2"), default=0.0)
        ctrl_yaw = _safe_float(frame.get("actuator_ctrl_3"), default=0.0)
        ctrl_thrust = _safe_float(frame.get("actuator_ctrl_4"), default=0.0)
        if abs(ctrl_thrust) <= 1e-9:
            ctrl_thrust = 0.32

        base = 1300.0 + 180.0 * _clamp(ctrl_thrust - 0.32, -0.8, 0.8)
        roll_delta = 110.0 * _clamp(ctrl_roll, -0.8, 0.8)
        pitch_delta = 110.0 * _clamp(ctrl_pitch, -0.8, 0.8)
        yaw_delta = 55.0 * _clamp(ctrl_yaw, -0.8, 0.8)
        return [
            _clamp(base + roll_delta + pitch_delta + yaw_delta, 900.0, 1900.0),
            _clamp(base - roll_delta + pitch_delta - yaw_delta, 900.0, 1900.0),
            _clamp(base - roll_delta - pitch_delta + yaw_delta, 900.0, 1900.0),
            _clamp(base + roll_delta - pitch_delta - yaw_delta, 900.0, 1900.0),
        ]

    def _apply_fallbacks(self, frame: dict[str, float], pymav_frame: dict[str, Any]) -> None:
        recv_time = frame.get("recv_time_s", 0.0)
        dt = None
        if self._prev_recv_time_s is not None:
            dt = recv_time - self._prev_recv_time_s
            if dt <= 1e-6:
                dt = None

        # Fallback motor outputs from ESC RPM first, then actuator controls.
        actuator_source = str(frame.get("actuator_source") or "none")
        for idx in range(1, 5):
            out_key = f"actuator_out_{idx}"
            if out_key not in frame:
                frame[out_key] = 0.0

        for idx in range(1, 5):
            motor_key = f"motor{idx}"
            if abs(frame.get(motor_key, 0.0)) > 1e-9:
                out_key = f"actuator_out_{idx}"
                if abs(frame.get(out_key, 0.0)) <= 1e-9:
                    frame[out_key] = frame[motor_key] * self.rpm_scale
                continue
            esc_key = f"esc_rpm_{idx}"
            esc_rpm = _safe_float(pymav_frame.get(esc_key), default=0.0)
            if esc_rpm > 0.0:
                frame[motor_key] = self._normalize_motor_rpm(esc_rpm)
                frame[f"actuator_out_{idx}"] = esc_rpm
                if actuator_source == "none":
                    actuator_source = "esc_rpm"
                continue
            ctrl_key = f"actuator_ctrl_{idx}"
            ctrl_val = _safe_float(frame.get(ctrl_key), default=0.0)
            if abs(ctrl_val) <= 1e-9:
                ctrl_val = _safe_float(pymav_frame.get(ctrl_key), default=0.0)
            if abs(ctrl_val) > 1e-9:
                frame[motor_key] = self._ctrl_to_motor(ctrl_val)
                if actuator_source == "none":
                    actuator_source = "px4_controls"

        if all(abs(frame.get(f"actuator_out_{idx}", 0.0)) <= 1e-9 for idx in range(1, 5)):
            proxy_outputs = self._proxy_actuator_outputs(frame)
            for idx, value in enumerate(proxy_outputs, start=1):
                frame[f"actuator_out_{idx}"] = value
                if abs(frame.get(f"motor{idx}", 0.0)) <= 1e-9:
                    frame[f"motor{idx}"] = self._normalize_motor_rpm(value)
            if actuator_source == "none":
                actuator_source = "proxy"
        frame["actuator_source"] = actuator_source

        # Fallback body rates from Euler differences.
        if dt is not None and all(abs(frame.get(key, 0.0)) <= 1e-9 for key in ("rollspeed", "pitchspeed", "yawspeed")):
            if self._prev_roll is not None:
                frame["rollspeed"] = (frame.get("roll", 0.0) - self._prev_roll) / dt
            if self._prev_pitch is not None:
                frame["pitchspeed"] = (frame.get("pitch", 0.0) - self._prev_pitch) / dt
            if self._prev_yaw is not None:
                frame["yawspeed"] = (frame.get("yaw", 0.0) - self._prev_yaw) / dt

        # Keep gyro aligned with angular rates when unavailable.
        if all(abs(frame.get(key, 0.0)) <= 1e-9 for key in ("gyro_x", "gyro_y", "gyro_z")):
            frame["gyro_x"] = frame.get("rollspeed", 0.0)
            frame["gyro_y"] = frame.get("pitchspeed", 0.0)
            frame["gyro_z"] = frame.get("yawspeed", 0.0)

        # Fallback acceleration from velocity differences.
        if dt is not None and all(abs(frame.get(key, 0.0)) <= 1e-9 for key in ("accel_x", "accel_y", "accel_z")):
            if self._prev_vx is not None:
                frame["accel_x"] = (frame.get("vx", 0.0) - self._prev_vx) / dt
            if self._prev_vy is not None:
                frame["accel_y"] = (frame.get("vy", 0.0) - self._prev_vy) / dt
            if self._prev_vz is not None:
                frame["accel_z"] = (frame.get("vz", 0.0) - self._prev_vz) / dt

        self._prev_recv_time_s = recv_time
        self._prev_roll = frame.get("roll", 0.0)
        self._prev_pitch = frame.get("pitch", 0.0)
        self._prev_yaw = frame.get("yaw", 0.0)
        self._prev_vx = frame.get("vx", 0.0)
        self._prev_vy = frame.get("vy", 0.0)
        self._prev_vz = frame.get("vz", 0.0)

    @staticmethod
    def snapshot_mav_state(mav: Any) -> dict[str, Any]:
        px4_controls = [0.0, 0.0, 0.0, 0.0]
        px4ext = getattr(mav, "px4ext", None)
        if px4ext is not None:
            controls = list(getattr(px4ext, "controls", []) or [])
            if controls:
                px4_controls = controls[:4] + [0.0] * max(0, 4 - len(controls[:4]))
        return {
            "recv_time_s": time.perf_counter(),
            "trueTimeStmp": getattr(mav, "trueTimeStmp", 0.0),
            "uavTimeStmp": getattr(mav, "uavTimeStmp", 0.0),
            "uavAngEular": list(getattr(mav, "uavAngEular", [0.0, 0.0, 0.0])),
            "uavAngRate": list(getattr(mav, "uavAngRate", [0.0, 0.0, 0.0])),
            "uavPosNED": list(getattr(mav, "uavPosNED", [0.0, 0.0, 0.0])),
            "uavVelNED": list(getattr(mav, "uavVelNED", [0.0, 0.0, 0.0])),
            "trueAccB": list(getattr(mav, "trueAccB", [0.0, 0.0, 0.0])),
            "trueMotorRPMS": list(
                getattr(mav, "trueMotorRPMS", [0.0, 0.0, 0.0, 0.0])
            ),
            "px4_controls": px4_controls,
        }

    def build_frame(
        self,
        *,
        mav_state: dict[str, Any],
        pymav_frame: dict[str, Any] | None = None,
    ) -> dict[str, float]:
        pymav_frame = dict(pymav_frame or {})
        has_true_motor = any(
            abs(_get_vector_component(mav_state, "trueMotorRPMS", idx)) > 1e-9
            for idx in range(4)
        )
        frame: dict[str, float] = {
            "seq": float(self._forwarder._seq),
            "time_s": self._time_s_from_state(mav_state),
            "recv_time_s": _safe_float(mav_state.get("recv_time_s"), default=0.0),
            "motor1": self._normalize_motor_rpm(_get_vector_component(mav_state, "trueMotorRPMS", 0)),
            "motor2": self._normalize_motor_rpm(_get_vector_component(mav_state, "trueMotorRPMS", 1)),
            "motor3": self._normalize_motor_rpm(_get_vector_component(mav_state, "trueMotorRPMS", 2)),
            "motor4": self._normalize_motor_rpm(_get_vector_component(mav_state, "trueMotorRPMS", 3)),
            "actuator_out_1": _get_vector_component(mav_state, "trueMotorRPMS", 0),
            "actuator_out_2": _get_vector_component(mav_state, "trueMotorRPMS", 1),
            "actuator_out_3": _get_vector_component(mav_state, "trueMotorRPMS", 2),
            "actuator_out_4": _get_vector_component(mav_state, "trueMotorRPMS", 3),
            "actuator_source": "true_motor_rpm" if has_true_motor else "none",
            "roll": _get_vector_component(mav_state, "uavAngEular", 0),
            "pitch": _get_vector_component(mav_state, "uavAngEular", 1),
            "yaw": _get_vector_component(mav_state, "uavAngEular", 2),
            "rollspeed": _get_vector_component(mav_state, "uavAngRate", 0),
            "pitchspeed": _get_vector_component(mav_state, "uavAngRate", 1),
            "yawspeed": _get_vector_component(mav_state, "uavAngRate", 2),
            "accel_x": _get_vector_component(mav_state, "trueAccB", 0),
            "accel_y": _get_vector_component(mav_state, "trueAccB", 1),
            "accel_z": _get_vector_component(mav_state, "trueAccB", 2),
            "gyro_x": _get_vector_component(mav_state, "uavAngRate", 0),
            "gyro_y": _get_vector_component(mav_state, "uavAngRate", 1),
            "gyro_z": _get_vector_component(mav_state, "uavAngRate", 2),
            "vx": _get_vector_component(mav_state, "uavVelNED", 0),
            "vy": _get_vector_component(mav_state, "uavVelNED", 1),
            "vz": _get_vector_component(mav_state, "uavVelNED", 2),
            "vel_x": _get_vector_component(mav_state, "uavVelNED", 0),
            "vel_y": _get_vector_component(mav_state, "uavVelNED", 1),
            "vel_z": _get_vector_component(mav_state, "uavVelNED", 2),
            "pos_z": _get_vector_component(mav_state, "uavPosNED", 2),
            "baro_alt": -_get_vector_component(mav_state, "uavPosNED", 2),
            "baro_alt_meter": 0.0,
            "baro_temp": 25.0,
            "baro_pressure": 101325.0,
            "baro_pressure_pa": 101325.0,
            "mag_x": 0.0,
            "mag_y": 0.0,
            "mag_z": 0.0,
            "battery_voltage_v": 0.0,
            "battery_remaining": 0.0,
            "actuator_ctrl_1": 0.0,
            "actuator_ctrl_2": 0.0,
            "actuator_ctrl_3": 0.0,
            "actuator_ctrl_4": 0.0,
            "rates_sp_roll": 0.0,
            "rates_sp_pitch": 0.0,
            "rates_sp_yaw": 0.0,
            "thrust_z": 0.0,
            "accel_vibration_metric": 0.0,
            "gyro_vibration_metric": 0.0,
            "esc_rpm_1": 0.0,
            "esc_rpm_2": 0.0,
            "esc_rpm_3": 0.0,
            "esc_rpm_4": 0.0,
            "nav_state": 0.0,
            "battery_warning": 0.0,
            "sys_status_health": 0.0,
            "system_status": 0.0,
            "pymav_highres_imu_seen": 0.0,
            "pymav_attitude_target_seen": 0.0,
            "pymav_actuator_control_seen": 0.0,
        }
        q_w, q_x, q_y, q_z = _quat_from_euler(
            frame["roll"],
            frame["pitch"],
            frame["yaw"],
        )
        frame["att_q_w"] = q_w
        frame["att_q_x"] = q_x
        frame["att_q_y"] = q_y
        frame["att_q_z"] = q_z
        px4_controls = list(mav_state.get("px4_controls", []) or [])
        for idx in range(min(4, len(px4_controls))):
            frame[f"actuator_ctrl_{idx + 1}"] = _safe_float(px4_controls[idx], default=0.0)
        for key in self.ENHANCED_FIELDS:
            if key in pymav_frame:
                frame[key] = _safe_float(pymav_frame.get(key))
        self._apply_fallbacks(frame, pymav_frame)
        return frame

    def emit_frame(
        self,
        *,
        mav_state: dict[str, Any],
        pymav_frame: dict[str, Any] | None = None,
    ) -> dict[str, float]:
        frame = self.build_frame(mav_state=mav_state, pymav_frame=pymav_frame)
        self._forwarder.send(frame)
        return frame

    def emit_frame_from_mav(
        self, mav: Any, *, pymav_frame: dict[str, Any] | None = None
    ) -> dict[str, float]:
        return self.emit_frame(
            mav_state=self.snapshot_mav_state(mav),
            pymav_frame=pymav_frame,
        )


class PymavStateSource:
    def __init__(self, *, pymav_address: str = "udpin:0.0.0.0:14030") -> None:
        self.pymav_address = pymav_address
        self._state = TelemetryState()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        from uav_fault.telemetry.pymav_supplement import run_pymav_listener

        self._stop_event.clear()
        self._thread = threading.Thread(
            target=run_pymav_listener,
            args=(self._state, self.pymav_address, self._stop_event),
            daemon=True,
            name="rfly-live-pymav",
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=3.0)

    def snapshot(self) -> dict[str, float]:
        return self._state.snapshot()


class RflyLiveEmitter:
    def __init__(
        self,
        *,
        bridge: RflyLiveBridge,
        mav_provider,
        pymav_source: PymavStateSource | None = None,
    ) -> None:
        self.bridge = bridge
        self.mav_provider = mav_provider
        self.pymav_source = pymav_source

    def emit_once(self) -> dict[str, float]:
        mav = self.mav_provider()
        if mav is None:
            raise RuntimeError("mav_provider returned None")
        pymav_frame = self.pymav_source.snapshot() if self.pymav_source else {}
        return self.bridge.emit_frame_from_mav(mav, pymav_frame=pymav_frame)

    def run(self, stop_event: threading.Event) -> None:
        interval = 1.0 / max(self.bridge.emit_rate_hz, 1)
        while not stop_event.is_set():
            self.emit_once()
            if stop_event.wait(interval):
                break



class RflyMavProvider:
    def __init__(
        self,
        *,
        sdk_root: str | Path = DEFAULT_RFLY_SDK_ROOT,
        copter_id: int = 1,
        udp_mode: int = 2,
        true_data: bool = True,
    ) -> None:
        self.sdk_root = Path(sdk_root)
        self.copter_id = int(copter_id)
        self.udp_mode = int(udp_mode)
        self.true_data = bool(true_data)
        self._mav = None
        self._true_data_started = False

    def _load_ctrl_module(self):
        module_path = self.sdk_root / "ctrl" / "PX4MavCtrlV4.py"
        if not module_path.exists():
            raise FileNotFoundError(f"??? RflySim SDK ????: {module_path}")
        ctrl_dir = str(module_path.parent)
        sdk_dir = str(self.sdk_root)
        for path in (ctrl_dir, sdk_dir):
            if path not in sys.path:
                sys.path.insert(0, path)
        module_name = f"_rfly_px4_mav_ctrl_{abs(hash(str(module_path.resolve())))}"
        spec = importlib.util.spec_from_file_location(module_name, module_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"???? RflySim SDK ????: {module_path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    def start(self):
        if self._mav is not None:
            return self._mav
        ctrl_module = self._load_ctrl_module()
        self._mav = ctrl_module.PX4MavCtrler(self.copter_id)
        self._mav.InitMavLoop(self.udp_mode)
        if self.true_data:
            self._mav.InitTrueDataLoop()
            self._true_data_started = True
        return self._mav

    def __call__(self):
        return self.start()

    def close(self) -> None:
        if self._mav is None:
            return
        if self._true_data_started and hasattr(self._mav, "EndTrueDataLoop"):
            self._mav.EndTrueDataLoop()
        if hasattr(self._mav, "stopRun"):
            self._mav.stopRun()


def create_rfly_mav_provider(
    *,
    sdk_root: str | Path = DEFAULT_RFLY_SDK_ROOT,
    copter_id: int = 1,
    udp_mode: int = 2,
    true_data: bool = True,
) -> RflyMavProvider:
    provider = RflyMavProvider(
        sdk_root=sdk_root,
        copter_id=copter_id,
        udp_mode=udp_mode,
        true_data=true_data,
    )
    provider.start()
    return provider


def build_live_emitter(
    *,
    bridge: RflyLiveBridge,
    sdk_root: str | Path = DEFAULT_RFLY_SDK_ROOT,
    copter_id: int = 1,
    udp_mode: int = 2,
    true_data: bool = True,
    pymav_enabled: bool = False,
    pymav_address: str = "udpin:0.0.0.0:14030",
) -> tuple[RflyLiveEmitter, RflyMavProvider, PymavStateSource | None]:
    provider = create_rfly_mav_provider(
        sdk_root=sdk_root,
        copter_id=copter_id,
        udp_mode=udp_mode,
        true_data=true_data,
    )
    pymav_source = PymavStateSource(pymav_address=pymav_address) if pymav_enabled else None
    if pymav_source is not None:
        pymav_source.start()
    emitter = RflyLiveEmitter(
        bridge=bridge,
        mav_provider=provider,
        pymav_source=pymav_source,
    )
    return emitter, provider, pymav_source


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="RflySim live bridge")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=15550)
    parser.add_argument("--emit-rate", type=int, default=50)
    parser.add_argument("--rpm-scale", type=float, default=10000.0)
    parser.add_argument("--sdk-root", default=str(DEFAULT_RFLY_SDK_ROOT))
    parser.add_argument("--copter-id", type=int, default=1)
    parser.add_argument("--udp-mode", type=int, default=2)
    parser.add_argument("--no-true-data", action="store_true")
    parser.add_argument("--pymav", action="store_true", help="?? MAVLink ??????")
    parser.add_argument("--pymav-address", default="udpin:0.0.0.0:14030")
    parser.add_argument("--duration", type=float, default=0.0, help="?????0 ??????")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    bridge = RflyLiveBridge(
        forward_host=args.host,
        forward_port=args.port,
        emit_rate_hz=args.emit_rate,
        rpm_scale=args.rpm_scale,
    )
    try:
        print(
            json.dumps(
                {
                    "status": "ready",
                    "host": args.host,
                    "port": args.port,
                    "emit_rate_hz": args.emit_rate,
                },
                ensure_ascii=False,
            )
        )
        while True:
            time.sleep(1.0 / max(args.emit_rate, 1))
    except KeyboardInterrupt:
        pass
    finally:
        bridge.close()


if __name__ == "__main__":
    main()
