from pathlib import Path
import math
import sys
import types
import threading

from rfly_bridge import rfly_live_bridge

def test_build_minimal_frame_maps_core_state_and_normalizes_motor_rpm():
    bridge = rfly_live_bridge.RflyLiveBridge(
        forward_host="127.0.0.1",
        forward_port=15550,
        emit_rate_hz=50,
    )
    frame = bridge.build_frame(
        mav_state={
            "trueTimeStmp": 12.5,
            "uavAngEular": [0.1, -0.2, 1.57],
            "uavAngRate": [0.01, -0.02, 0.03],
            "uavVelNED": [4.0, 5.0, -6.0],
            "uavPosNED": [10.0, 20.0, -30.0],
            "trueAccB": [1.1, 2.2, 3.3],
            "trueMotorRPMS": [5000.0, 5200.0, 5400.0, 5600.0],
        },
        pymav_frame={},
    )

    assert frame["seq"] == 0
    assert frame["time_s"] == 12.5
    assert frame["roll"] == 0.1
    assert frame["pitch"] == -0.2
    assert frame["yaw"] == 1.57
    assert frame["rollspeed"] == 0.01
    assert frame["pitchspeed"] == -0.02
    assert frame["yawspeed"] == 0.03
    assert frame["vx"] == 4.0
    assert frame["vy"] == 5.0
    assert frame["vz"] == -6.0
    assert frame["pos_z"] == -30.0
    assert frame["accel_x"] < 0.0
    assert frame["accel_y"] < 2.2
    assert frame["accel_z"] < -6.0
    assert math.isclose(frame["motor1"], 0.5, rel_tol=1e-6)
    assert math.isclose(frame["motor4"], 0.56, rel_tol=1e-6)


def test_build_minimal_frame_prefers_pymav_fields_when_present():
    bridge = rfly_live_bridge.RflyLiveBridge(
        forward_host="127.0.0.1",
        forward_port=15550,
        emit_rate_hz=50,
    )
    frame = bridge.build_frame(
        mav_state={
            "trueTimeStmp": 1.0,
            "uavAngEular": [0.0, 0.0, 0.0],
            "uavAngRate": [0.0, 0.0, 0.0],
            "uavVelNED": [0.0, 0.0, 0.0],
            "uavPosNED": [0.0, 0.0, 0.0],
            "trueAccB": [0.0, 0.0, 9.8],
            "trueMotorRPMS": [0.0, 0.0, 0.0, 0.0],
        },
        pymav_frame={
            "baro_alt_meter": 4.1,
            "baro_temp": 26.5,
            "baro_pressure_pa": 100812.0,
            "mag_x": 0.12,
            "mag_y": -0.03,
            "mag_z": 0.41,
            "rates_sp_roll": 0.1,
            "rates_sp_pitch": -0.2,
            "rates_sp_yaw": 0.3,
            "thrust_z": 0.48,
            "actuator_ctrl_1": 0.11,
            "actuator_ctrl_2": 0.22,
            "actuator_ctrl_3": 0.33,
            "actuator_ctrl_4": 0.44,
            "battery_warning": 2.0,
            "system_status": 4.0,
            "sys_status_health": 7.0,
            "pymav_highres_imu_seen": 1.0,
        },
    )

    assert frame["baro_alt_meter"] == 4.1
    assert frame["baro_temp"] == 26.5
    assert frame["baro_pressure_pa"] == 100812.0
    assert frame["mag_x"] == 0.12
    assert frame["mag_z"] == 0.41
    assert frame["rates_sp_roll"] == 0.1
    assert frame["rates_sp_pitch"] == -0.2
    assert frame["rates_sp_yaw"] == 0.3
    assert frame["thrust_z"] == 0.48
    assert frame["actuator_ctrl_4"] == 0.44
    assert frame["battery_warning"] == 2.0
    assert frame["system_status"] == 4.0
    assert frame["sys_status_health"] == 7.0
    assert frame["pymav_highres_imu_seen"] == 1.0


def test_build_frame_overrides_stale_official_state_with_pymav_actual_fields():
    bridge = rfly_live_bridge.RflyLiveBridge(
        forward_host="127.0.0.1",
        forward_port=15550,
        emit_rate_hz=50,
    )
    frame = bridge.build_frame(
        mav_state={
            "trueTimeStmp": 999999.0,
            "uavAngEular": [0.0, 0.0, 0.0],
            "uavAngRate": [0.0, 0.0, 0.0],
            "uavVelNED": [0.0, 0.0, 0.0],
            "uavPosNED": [0.0, 0.0, 0.0],
            "trueAccB": [0.0, 0.0, 0.0],
            "trueMotorRPMS": [5100.0, 5200.0, 5300.0, 5400.0],
        },
        pymav_frame={
            "time_s": 2.25,
            "roll": 0.12,
            "pitch": -0.08,
            "yaw": 1.25,
            "rollspeed": 0.01,
            "pitchspeed": -0.02,
            "yawspeed": 0.03,
            "accel_x": 0.4,
            "accel_y": -0.5,
            "accel_z": -9.7,
            "vx": 1.2,
            "vy": -0.4,
            "vz": 0.7,
            "pos_z": -3.5,
        },
    )

    assert frame["time_s"] == 2.25
    assert frame["roll"] == 0.12
    assert frame["pitch"] == -0.08
    assert frame["yaw"] == 1.25
    assert frame["rollspeed"] == 0.01
    assert frame["gyro_x"] == 0.01
    assert frame["accel_z"] == -9.7
    assert frame["vx"] == 1.2
    assert frame["vel_x"] == 1.2
    assert frame["pos_z"] == -3.5
    assert frame["baro_alt"] == 3.5
    assert frame["att_q_w"] != 1.0


def test_emit_frame_from_mav_reads_official_object_state_shape():
    bridge = rfly_live_bridge.RflyLiveBridge(
        forward_host="127.0.0.1",
        forward_port=15550,
        emit_rate_hz=50,
    )

    class DummyMav:
        trueTimeStmp = 3.2
        uavAngEular = [0.2, 0.3, 0.4]
        uavAngRate = [0.5, 0.6, 0.7]
        uavPosNED = [10.0, 11.0, -12.0]
        uavVelNED = [1.0, 2.0, 3.0]
        trueAccB = [4.0, 5.0, 6.0]
        trueMotorRPMS = [7000.0, 7100.0, 7200.0, 7300.0]

    frame = bridge.build_frame(
        mav_state=bridge.snapshot_mav_state(DummyMav()),
        pymav_frame={},
    )

    assert frame["time_s"] == 3.2
    assert frame["roll"] == 0.2
    assert frame["pitchspeed"] == 0.6
    assert frame["vx"] == 1.0
    assert frame["pos_z"] == -12.0
    assert math.isclose(frame["motor1"], 0.7, rel_tol=1e-6)
    assert math.isclose(frame["motor4"], 0.73, rel_tol=1e-6)


def test_build_minimal_frame_is_compatible_with_px4_live_schema_adapter():
    sys.modules.setdefault("torch", types.ModuleType("torch"))
    from uav_fault.telemetry.schema_adapter import adapt_px4_live_frame

    bridge = rfly_live_bridge.RflyLiveBridge(
        forward_host="127.0.0.1",
        forward_port=15550,
        emit_rate_hz=50,
    )
    frame = bridge.build_frame(
        mav_state={
            "trueTimeStmp": 2.5,
            "uavAngEular": [0.0, 0.0, 0.0],
            "uavAngRate": [0.1, 0.2, 0.3],
            "uavVelNED": [1.0, 2.0, 3.0],
            "uavPosNED": [0.0, 0.0, -4.0],
            "trueAccB": [3.0, 4.0, 0.0],
            "trueMotorRPMS": [5000.0, 5000.0, 5000.0, 5000.0],
        },
        pymav_frame={
            "baro_temp": 20.0,
            "baro_pressure_pa": 100100.0,
            "mag_x": 0.01,
            "mag_y": -0.02,
            "mag_z": 0.03,
        },
    )

    canonical, coverage = adapt_px4_live_frame(frame)

    assert canonical["actuator_out_1"] > 0.0
    assert canonical["vel_x"] == 1.0
    assert canonical["vel_y"] == 2.0
    assert canonical["vel_z"] == 3.0
    assert canonical["acc_norm"] > 9.8
    assert canonical["baro_temp"] == 20.0
    assert canonical["mag_x"] == 0.01
    assert "actuator_out_1" in coverage["mapped"]
    assert "att_q_w" in coverage["derived"]


def test_pymav_state_source_snapshot_reads_from_telemetry_state():
    source = rfly_live_bridge.PymavStateSource()
    source._state.update(baro_pressure_pa=100500.0, mag_x=0.02, pymav_highres_imu_seen=1.0)

    snapshot = source.snapshot()

    assert snapshot["baro_pressure_pa"] == 100500.0
    assert snapshot["mag_x"] == 0.02
    assert snapshot["pymav_highres_imu_seen"] == 1.0


def test_live_emitter_emit_once_uses_mav_and_pymav_sources():
    bridge = rfly_live_bridge.RflyLiveBridge(
        forward_host="127.0.0.1",
        forward_port=15550,
        emit_rate_hz=50,
    )
    sent = []
    bridge._forwarder = types.SimpleNamespace(send=lambda row: sent.append(dict(row)), _seq=0)

    class DummyMav:
        trueTimeStmp = 5.0
        uavAngEular = [0.1, 0.2, 0.3]
        uavAngRate = [0.4, 0.5, 0.6]
        uavPosNED = [7.0, 8.0, -9.0]
        uavVelNED = [1.0, 2.0, 3.0]
        trueAccB = [4.0, 5.0, 6.0]
        trueMotorRPMS = [5000.0, 5000.0, 5000.0, 5000.0]

    pymav_source = types.SimpleNamespace(
        snapshot=lambda: {
            "baro_pressure_pa": 100100.0,
            "rates_sp_roll": 0.11,
            "pymav_attitude_target_seen": 1.0,
        }
    )

    emitter = rfly_live_bridge.RflyLiveEmitter(
        bridge=bridge,
        mav_provider=lambda: DummyMav(),
        pymav_source=pymav_source,
    )

    frame = emitter.emit_once()

    assert sent
    assert frame["time_s"] == 5.0
    assert frame["baro_pressure_pa"] == 100100.0
    assert frame["rates_sp_roll"] == 0.11
    assert frame["pymav_attitude_target_seen"] == 1.0
    assert math.isclose(frame["motor1"], 0.5, rel_tol=1e-6)


def test_live_emitter_loop_stops_via_event():
    bridge = rfly_live_bridge.RflyLiveBridge(
        forward_host="127.0.0.1",
        forward_port=15550,
        emit_rate_hz=1000,
    )
    sent = []
    bridge._forwarder = types.SimpleNamespace(send=lambda row: sent.append(dict(row)), _seq=0)

    class DummyMav:
        trueTimeStmp = 1.0
        uavAngEular = [0.0, 0.0, 0.0]
        uavAngRate = [0.0, 0.0, 0.0]
        uavPosNED = [0.0, 0.0, 0.0]
        uavVelNED = [0.0, 0.0, 0.0]
        trueAccB = [0.0, 0.0, 0.0]
        trueMotorRPMS = [0.0, 0.0, 0.0, 0.0]

    emitter = rfly_live_bridge.RflyLiveEmitter(
        bridge=bridge,
        mav_provider=lambda: DummyMav(),
        pymav_source=None,
    )

    stop_event = threading.Event()
    thread = threading.Thread(target=emitter.run, args=(stop_event,), daemon=True)
    thread.start()
    stop_event.set()
    thread.join(timeout=1.0)

    assert not thread.is_alive()


def test_create_rfly_mav_provider_imports_official_sdk_and_starts_loops(tmp_path):
    sdk_ctrl = tmp_path / "RflySimSDK" / "ctrl"
    sdk_ctrl.mkdir(parents=True)
    (sdk_ctrl / "PX4MavCtrlV4.py").write_text(
        """
class PX4MavCtrler:
    def __init__(self, copter_id):
        self.copter_id = copter_id
        self.calls = []
        self.trueTimeStmp = 1.25
        self.uavAngEular = [0.1, 0.2, 0.3]
        self.uavAngRate = [0.4, 0.5, 0.6]
        self.uavPosNED = [7.0, 8.0, -9.0]
        self.uavVelNED = [1.0, 2.0, 3.0]
        self.trueAccB = [4.0, 5.0, 6.0]
        self.trueMotorRPMS = [5100.0, 5200.0, 5300.0, 5400.0]
    def InitMavLoop(self, udp_mode=2):
        self.calls.append(("InitMavLoop", udp_mode))
    def InitTrueDataLoop(self):
        self.calls.append(("InitTrueDataLoop",))
    def stopRun(self):
        self.calls.append(("stopRun",))
    def EndTrueDataLoop(self):
        self.calls.append(("EndTrueDataLoop",))
""",
        encoding="utf-8",
    )

    provider = rfly_live_bridge.create_rfly_mav_provider(
        sdk_root=tmp_path / "RflySimSDK",
        copter_id=3,
        udp_mode=2,
        true_data=True,
    )

    mav = provider()

    assert mav.copter_id == 3
    assert ("InitMavLoop", 2) in mav.calls
    assert ("InitTrueDataLoop",) in mav.calls

    provider.close()
    assert ("EndTrueDataLoop",) in mav.calls
    assert ("stopRun",) in mav.calls


def test_build_live_emitter_uses_rfly_provider_and_forwards_one_frame(tmp_path):
    sdk_ctrl = tmp_path / "RflySimSDK" / "ctrl"
    sdk_ctrl.mkdir(parents=True)
    (sdk_ctrl / "PX4MavCtrlV4.py").write_text(
        """
class PX4MavCtrler:
    def __init__(self, copter_id):
        self.trueTimeStmp = 2.5
        self.uavAngEular = [0.0, 0.0, 0.0]
        self.uavAngRate = [0.1, 0.2, 0.3]
        self.uavPosNED = [0.0, 0.0, -4.0]
        self.uavVelNED = [1.0, 2.0, 3.0]
        self.trueAccB = [4.0, 5.0, 6.0]
        self.trueMotorRPMS = [5000.0, 5000.0, 5000.0, 5000.0]
    def InitMavLoop(self, udp_mode=2):
        pass
    def InitTrueDataLoop(self):
        pass
""",
        encoding="utf-8",
    )

    bridge = rfly_live_bridge.RflyLiveBridge(
        forward_host="127.0.0.1",
        forward_port=15550,
        emit_rate_hz=50,
    )
    sent = []
    bridge._forwarder = types.SimpleNamespace(send=lambda row: sent.append(dict(row)), _seq=0)
    emitter, provider, pymav_source = rfly_live_bridge.build_live_emitter(
        bridge=bridge,
        sdk_root=tmp_path / "RflySimSDK",
        copter_id=1,
        udp_mode=2,
        true_data=True,
        pymav_enabled=False,
    )

    frame = emitter.emit_once()

    assert provider() is not None
    assert pymav_source is None
    assert sent
    assert frame["time_s"] == 2.5
    assert frame["actuator_source"] == "true_motor_rpm"
    assert frame["actuator_out_1"] == 1300.0
    assert math.isclose(frame["motor1"], 0.5, rel_tol=1e-6)
