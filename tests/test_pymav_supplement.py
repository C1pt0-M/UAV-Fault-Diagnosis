import threading
import sys
import types

if "pymavlink" not in sys.modules:
    fake_pymavlink = types.ModuleType("pymavlink")
    fake_pymavlink.mavutil = types.SimpleNamespace(
        mavlink=types.SimpleNamespace(MAV_CMD_SET_MESSAGE_INTERVAL=511),
        mavlink_connection=lambda address: None,
    )
    sys.modules["pymavlink"] = fake_pymavlink

from uav_fault.telemetry import pymav_supplement
from uav_fault.telemetry.config import AUX_COLUMNS, FEATURE_COLUMNS
from uav_fault.telemetry.state import TelemetryState


class FakeMessage:
    def __init__(self, msg_type, **fields):
        self._msg_type = msg_type
        for key, value in fields.items():
            setattr(self, key, value)

    def get_type(self):
        return self._msg_type

    def get_srcComponent(self):
        return getattr(self, "src_component", 1)


class FakeConnection:
    def __init__(self, messages, stop_event):
        self.target_system = 1
        self.target_component = 1
        self._messages = list(messages)
        self._stop_event = stop_event
        self.closed = False

    def wait_heartbeat(self, timeout=30):
        return True

    def recv_match(self, type=None, blocking=True, timeout=1.0):
        if self._messages:
            msg = self._messages.pop(0)
            if not self._messages:
                self._stop_event.set()
            return msg
        self._stop_event.set()
        return None

    def close(self):
        self.closed = True


def test_run_pymav_listener_updates_highres_imu_pressure_temperature_and_magnetometer(
    monkeypatch,
):
    stop_event = threading.Event()
    state = TelemetryState()
    conn = FakeConnection(
        [
            FakeMessage(
                "HIGHRES_IMU",
                pressure_alt=123.4,
                abs_pressure=1001.5,
                temperature=24.5,
                xmag=0.1,
                ymag=-0.2,
                zmag=0.3,
            )
        ],
        stop_event,
    )

    monkeypatch.setattr(
        pymav_supplement.mavutil, "mavlink_connection", lambda address: conn
    )
    monkeypatch.setattr(
        pymav_supplement, "_request_stream", lambda *args, **kwargs: None
    )

    pymav_supplement.run_pymav_listener(
        state, address="udpin:0.0.0.0:14030", stop_event=stop_event
    )

    row = state.snapshot()
    assert row["baro_alt_meter"] == 123.4
    assert row["baro_pressure_pa"] == 100150.0
    assert row["baro_temp"] == 24.5
    assert row["mag_x"] == 0.1
    assert row["mag_y"] == -0.2
    assert row["mag_z"] == 0.3
    assert row["pymav_highres_imu_seen"] == 1
    assert conn.closed is True


def test_pressure_conversion_accepts_pascal_input_without_rescaling():
    assert pymav_supplement._pressure_to_pa(100713.5) == 100713.5
    assert pymav_supplement._pressure_to_pa(1007.135) == 100713.5


def test_run_pymav_listener_updates_attitude_target_and_actuator_control_seen_counters(
    monkeypatch,
):
    stop_event = threading.Event()
    state = TelemetryState()
    conn = FakeConnection(
        [
            FakeMessage(
                "ATTITUDE_TARGET",
                body_roll_rate=0.11,
                body_pitch_rate=-0.22,
                body_yaw_rate=0.33,
                thrust=0.44,
            ),
            FakeMessage(
                "ACTUATOR_CONTROL_TARGET",
                group_mlx=0,
                controls=[0.1, 0.2, 0.3, 0.4],
            ),
        ],
        stop_event,
    )

    monkeypatch.setattr(
        pymav_supplement.mavutil, "mavlink_connection", lambda address: conn
    )
    monkeypatch.setattr(
        pymav_supplement, "_request_stream", lambda *args, **kwargs: None
    )

    pymav_supplement.run_pymav_listener(
        state, address="udpin:0.0.0.0:14030", stop_event=stop_event
    )

    row = state.snapshot()
    assert row["rates_sp_roll"] == 0.11
    assert row["rates_sp_pitch"] == -0.22
    assert row["rates_sp_yaw"] == 0.33
    assert row["thrust_z"] == 0.44
    assert row["actuator_ctrl_1"] == 0.1
    assert row["actuator_ctrl_4"] == 0.4
    assert row["pymav_attitude_target_seen"] == 1
    assert row["pymav_actuator_control_seen"] == 1


def test_run_pymav_listener_updates_actual_attitude_and_local_position(monkeypatch):
    stop_event = threading.Event()
    state = TelemetryState()
    conn = FakeConnection(
        [
            FakeMessage(
                "ATTITUDE",
                roll=0.12,
                pitch=-0.08,
                yaw=1.25,
                rollspeed=0.01,
                pitchspeed=-0.02,
                yawspeed=0.03,
            ),
            FakeMessage(
                "LOCAL_POSITION_NED",
                vx=1.2,
                vy=-0.4,
                vz=0.7,
                z=-3.5,
            ),
        ],
        stop_event,
    )

    monkeypatch.setattr(
        pymav_supplement.mavutil, "mavlink_connection", lambda address: conn
    )
    monkeypatch.setattr(
        pymav_supplement, "_request_stream", lambda *args, **kwargs: None
    )

    pymav_supplement.run_pymav_listener(
        state, address="udpin:0.0.0.0:14031", stop_event=stop_event
    )

    row = state.snapshot()
    assert row["roll"] == 0.12
    assert row["pitch"] == -0.08
    assert row["yaw"] == 1.25
    assert row["rollspeed"] == 0.01
    assert row["pitchspeed"] == -0.02
    assert row["yawspeed"] == 0.03
    assert row["vx"] == 1.2
    assert row["vy"] == -0.4
    assert row["vz"] == 0.7
    assert row["pos_z"] == -3.5
    assert row["baro_alt"] == 3.5
    assert row["pymav_attitude_seen"] == 1
    assert row["pymav_local_position_seen"] == 1


def test_run_pymav_listener_updates_armed_state_from_heartbeat(monkeypatch):
    stop_event = threading.Event()
    state = TelemetryState()
    conn = FakeConnection(
        [
            FakeMessage(
                "HEARTBEAT",
                src_component=1,
                system_status=4,
                base_mode=128,
            )
        ],
        stop_event,
    )

    monkeypatch.setattr(
        pymav_supplement.mavutil, "mavlink_connection", lambda address: conn
    )
    monkeypatch.setattr(
        pymav_supplement, "_request_stream", lambda *args, **kwargs: None
    )

    pymav_supplement.run_pymav_listener(
        state, address="udpin:0.0.0.0:14031", stop_event=stop_event
    )

    row = state.snapshot()
    assert row["armed"] == 1.0
    assert row["pymav_heartbeat_seen"] == 1.0


def test_feature_columns_include_live_baro_temp_and_actuator_controls():
    assert "baro_temp" in FEATURE_COLUMNS
    assert "baro_pressure_pa" in FEATURE_COLUMNS
    assert "mag_x" in FEATURE_COLUMNS
    assert "mag_y" in FEATURE_COLUMNS
    assert "mag_z" in FEATURE_COLUMNS
    assert "actuator_ctrl_1" in FEATURE_COLUMNS
    assert "actuator_ctrl_2" in FEATURE_COLUMNS
    assert "actuator_ctrl_3" in FEATURE_COLUMNS
    assert "actuator_ctrl_4" in FEATURE_COLUMNS
    assert "pymav_highres_imu_seen" in AUX_COLUMNS
    assert "pymav_attitude_target_seen" in AUX_COLUMNS
    assert "pymav_actuator_control_seen" in AUX_COLUMNS
    assert "armed" in AUX_COLUMNS
    assert "pymav_heartbeat_seen" in AUX_COLUMNS
